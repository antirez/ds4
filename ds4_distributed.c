/* =========================================================================
 * ds4_distributed.c - Distributed inference runtime.
 * =========================================================================
 *
 * This module owns the DS4 distributed transport and orchestration layer. The
 * rest of the engine still sees a normal ds4_session: when distributed mode is
 * active, ds4.c delegates sync/eval/save/load to the coordinator session API in
 * this file.
 *
 * Workers execute contiguous model slices with the same graph-slice entry
 * points used by the local engine. KV snapshots remain topology-independent:
 * save gathers worker-owned layer tensors into the normal DSV4 payload, and
 * load splits a normal DSV4 payload across the currently registered route.
 */

#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* =========================================================================
 * Protocol Constants And Wire Records
 * ========================================================================= */

#define DS4_DIST_MAGIC 0x44533444u /* DS4D */
#define DS4_DIST_MSG_HELLO 1u
#define DS4_DIST_MSG_ERROR 2u
#define DS4_DIST_MSG_WORK 3u
#define DS4_DIST_MSG_RESULT 4u
#define DS4_DIST_MSG_SNAPSHOT_SAVE_REQ 5u
#define DS4_DIST_MSG_SNAPSHOT_BEGIN 6u
#define DS4_DIST_MSG_SNAPSHOT_CHUNK 7u
#define DS4_DIST_MSG_SNAPSHOT_DONE 8u
#define DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN 9u
#define DS4_DIST_PROTOCOL_VERSION 2u
#define DS4_DIST_MAX_ROUTE_BYTES (1024u * 1024u)
#include "ds4_transport.h"
#include "ds4_dist_v3.h"
#define DS4_DIST_MAX_MODEL_NAME 127u
/* Speculative WORK flag semantics and their capability contract live in
 * ds4_dist_v3.h so negotiation tests exercise the same wire constants used
 * here. */
#define DS4_DIST_ACTIVATION_BITS_DEFAULT 32u
#define DS4_DIST_ROUTE_F_OUTPUT_LOGITS 0x00000001u
#define DS4_DIST_ROUTE_RETURN_UPSTREAM 1u
#define DS4_DIST_RECV_TRANSPORT_ERROR 1
#define DS4_DIST_RECV_REMOTE_ERROR 2
#define DS4_DIST_SNAPSHOT_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_dist_frame_header;

typedef struct {
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    uint32_t model_name_len;
} ds4_dist_hello_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    uint32_t token_bytes;
    uint32_t input_hc_bytes;
    uint32_t input_hc_bits;
    uint32_t route_count;
    uint32_t route_index;
    uint32_t route_bytes;
} ds4_dist_work_fixed;

typedef struct {
    uint32_t host_len;
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
} ds4_dist_route_fixed;

typedef struct {
    uint32_t kind;
    uint32_t host_len;
    uint32_t port;
} ds4_dist_route_return_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t status;
    uint32_t result_kind;
    uint32_t telemetry_count;
    uint32_t telemetry_bytes;
    uint32_t payload_bytes;
    uint32_t payload_bits;
} ds4_dist_result_fixed;

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t route_index;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t eval_usec;
    uint32_t downstream_wait_usec;
    uint32_t forward_send_usec;
    uint32_t input_bytes;
    uint32_t output_bytes;
} ds4_dist_telemetry_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
} ds4_dist_snapshot_req_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t payload_hi;
    uint32_t payload_lo;
    uint32_t status;
    uint32_t token_bytes;
    uint32_t message_bytes;
} ds4_dist_snapshot_begin_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t chunk_bytes;
} ds4_dist_snapshot_chunk_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t status;
    uint32_t message_bytes;
} ds4_dist_snapshot_done_fixed;

/* =========================================================================
 * Runtime State
 * =========================================================================
 *
 * The coordinator registry is shared by the accept thread and by the session
 * calls made from the main inference thread. Workers keep per-session KV state
 * keyed by the coordinator-provided session ID so independent callers do not
 * share token timelines by accident.
 */

typedef struct ds4_dist_worker_entry {
    int fd;
    ds4_transport *transport;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    uint32_t protocol_version;
    uint32_t caps;
    uint64_t link_generation;
    struct ds4_dist_worker_entry *next;
} ds4_dist_worker_entry;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t local_start;
    uint32_t local_end;
    uint32_t ctx_size;
    bool local_has_output;
    bool local_can_output_head;
    bool replay_check;
    bool debug;
    bool use_control_for_work;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    ds4_distributed_transport transport_policy;
    const char *nhi_device;
    uint64_t generation;
    uint64_t next_link_generation;
    pthread_mutex_t mu;
    ds4_dist_worker_entry *workers;
    bool shutting_down;
} ds4_dist_coordinator_state;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_client_ctx;

typedef struct {
    ds4_dist_coordinator_state *state;
    int listen_fd;
} ds4_dist_accept_ctx;

typedef struct ds4_dist_worker_session {
    uint64_t session_id;
    uint64_t token_hash;
    bool token_hash_valid;
    ds4_session *session;
    ds4_dist_mtp_frontier frontier;
    struct ds4_dist_worker_session *next;
} ds4_dist_worker_session;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t layer_start;
    uint32_t layer_end;
    bool has_output;
    int ctx_size;
    uint64_t hidden_f32_values;
    int listen_fd;
    ds4_distributed_transport transport_policy;
    const char *nhi_device;
    pthread_mutex_t mu;
    ds4_dist_worker_session *sessions;
    bool spec_warned;
} ds4_dist_worker_state;

typedef struct ds4_dist_worker_upstream ds4_dist_worker_upstream;

typedef struct ds4_dist_pending_request {
    uint64_t request_id;
    double downstream_t0;
    ds4_dist_telemetry_fixed telemetry;
    ds4_dist_v3_result_limits result_limits;
    uint32_t activation_bits;
    uint32_t telemetry_count_limit;
    struct ds4_dist_pending_request *next;
} ds4_dist_pending_request;

typedef struct ds4_dist_worker_forwarder {
    ds4_dist_worker_upstream *upstream;
    char host[NI_MAXHOST];
    uint32_t port;
    int fd;
    ds4_transport *transport;
    pthread_t tid;
    bool thread_started;
    pthread_mutex_t send_mu;
    pthread_mutex_t queue_mu;
    pthread_cond_t queue_not_full;
    ds4_dist_pending_request *pending_head;
    ds4_dist_pending_request *pending_tail;
    uint32_t pending_count;
    uint32_t pending_depth;
    bool closing;
    struct ds4_dist_worker_forwarder *next;
} ds4_dist_worker_forwarder;

struct ds4_dist_worker_upstream {
    ds4_dist_worker_state *state;
    int fd;
    ds4_transport *transport;
    uint32_t selected_caps;
    pthread_mutex_t write_mu;
    pthread_mutex_t forward_mu;
    ds4_dist_worker_forwarder *forwarders;
};

typedef struct {
    ds4_transport_bulk_desc desc;
    ds4_transport_lease *lease;
    bool prepared;
} ds4_dist_tx_bulk_plan;

typedef struct {
    ds4_dist_work_fixed work;
    ds4_transport_bulk_desc bulk_desc;
    int *tokens;
    void *input_hc_wire;
    ds4_transport_lease *input_lease;
    void *route_blob;
    uint32_t frame_bytes;
    bool v3;
    uint64_t reject_request_id;
    char reject_message[160];
} ds4_dist_work_message;

enum {
    DS4_DIST_WORK_RECV_READY = 1,
    DS4_DIST_WORK_RECV_REJECTED = 2,
};


typedef struct ds4_dist_worker_job {
    ds4_dist_work_message message;
    struct ds4_dist_worker_job *next;
} ds4_dist_worker_job;

typedef struct {
    ds4_dist_worker_state *state;
    ds4_dist_worker_upstream *upstream;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    pthread_cond_t idle;
    ds4_dist_worker_job *head;
    ds4_dist_worker_job *tail;
    uint32_t queued;
    uint32_t active;
#ifdef DS4_TEST_HOOKS
    uint32_t control_waiters;
#endif
    uint32_t depth;
    bool closed;
    bool canceled;
    int rc;
} ds4_dist_worker_job_queue;

typedef struct {
    ds4_dist_worker_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_data_client_ctx;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t kind;
} ds4_dist_route_return;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    int fd;
    ds4_transport *transport;
    uint32_t caps; /* negotiated link caps; memory-only, never serialized */
} ds4_dist_route_entry;

typedef struct {
    ds4_dist_route_entry *entry;
    uint32_t count;
    void *blob;
    uint32_t blob_bytes;
} ds4_dist_route_plan;

typedef struct {
    uint32_t ctx;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t comp_cap;
    uint32_t token_count;
    uint32_t n_layers;
    uint32_t head_dim;
    uint32_t indexer_head_dim;
    uint32_t vocab;
    uint32_t raw_live;
} ds4_dist_kv_layout;

typedef struct {
    FILE *fp;
    uint64_t bytes;
    uint32_t layer_start;
    uint32_t layer_end;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
} ds4_dist_kv_shard_file;

struct ds4_dist_session {
    ds4_dist_coordinator_state state;
    int listen_fd;
    pthread_t accept_tid;
    bool accept_started;
    ds4_dist_accept_ctx accept_ctx;
    ds4_dist_route_plan plan;
    bool plan_ready;
    uint64_t plan_generation;
    uint64_t session_id;
    uint64_t request_id;
    uint64_t snapshot_request_id;
    /* Continuation drafts from the last fully-accepted verify span: valid
     * only for spec_pending_token decoded at timeline length
     * spec_pending_pos, letting the next speculative cycle fuse the base
     * decode into its verify span (one span, one round trip). */
    int spec_pending_drafts[16];
    int spec_pending_count;
    int spec_pending_token;
    uint32_t spec_pending_pos;
    bool spec_cap_warned;
    /* Aggregate counters for the dist speculative-decode telemetry line.
     * Emitted with the same gate as the single-node DSpark stats so an
     * external validator can turn the counters on with one env knob. */
    uint64_t spec_cycles;
    uint64_t spec_proposed;
    uint64_t spec_accepted;
};

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_dist_logprob;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    ds4_transport *transport;
    ds4_session *progress_session;
    uint64_t first_request_id;
    uint64_t *expected_hashes;
    uint32_t *expected_kinds;
    uint32_t count;
    uint32_t total_tokens;
    uint32_t chunk_cap;
    uint32_t progress_base;
    uint32_t progress_total;
    uint32_t progress_completed;
    bool progress_done;
    uint64_t hc_values;
    bool allow_hidden;
    uint32_t expected_final_kind;
    uint32_t final_kind;
    void *final_payload;
    uint32_t final_payload_bytes;
    int rc;
    char err[256];
    pthread_mutex_t progress_mu;
    pthread_cond_t progress_cv;
} ds4_dist_prefill_result_reader;

typedef struct {
    uint32_t pos;
    uint32_t n_tokens;
    uint32_t hidden_bytes;
    uint64_t request_id;
    uint64_t prefix_hash;
    uint64_t result_hash;
    bool reset_session;
    bool ack_only;
    float *hidden;
} ds4_dist_prefill_send_slot;

typedef struct {
    ds4_dist_coordinator_state *state;
    const ds4_dist_route_plan *plan;
    const ds4_tokens *prompt;
    uint64_t session_id;
    int fd;
    ds4_transport *transport;
    ds4_dist_prefill_send_slot *slots;
    uint32_t slot_count;
    uint32_t head;
    uint32_t tail;
    uint32_t queued;
    bool producer_done;
    bool stop;
    int rc;
    double send_sec;
    uint64_t send_bytes;
    char err[256];
    pthread_mutex_t mu;
    pthread_cond_t can_enqueue;
    pthread_cond_t can_dequeue;
} ds4_dist_prefill_sender;

/* =========================================================================
 * Small Utilities And Forward Declarations
 * ========================================================================= */

static uint32_t dist_prefill_send_depth(uint32_t chunk_count) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_PREFILL_SEND_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    if (chunk_count != 0 && depth > chunk_count) depth = chunk_count;
    return depth ? depth : 1;
}

static int dist_send_work_frame(
        int fd,
        ds4_transport *transport,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob);
static int dist_send_work_frame_prepared(
        int fd,
        ds4_transport *transport,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob,
        const ds4_transport_bulk_desc *prepared_desc,
        ds4_transport_lease *tx_lease);
static int dist_write_full(int fd, const void *buf, size_t len);
static int dist_send_snapshot_file_chunks(
        int fd,
        uint64_t request_id,
        FILE *fp,
        uint64_t bytes);

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static int dist_worker_handle_snapshot_save(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static int dist_worker_handle_snapshot_load(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static int dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd,
        ds4_transport *transport,
        uint32_t selected_caps);
static void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream);
static int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg);
static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen);
static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen);

static uint32_t dist_resolved_layer_end(const ds4_dist_options *opt, uint32_t n_layers) {
    if (opt->layers.has_output) return n_layers - 1u;
    return opt->layers.end;
}

/* Research-only trust-all speculation knob (measurement mode, never the
 * default).  When enabled the distributed speculative cycle emits every
 * drafted token without running any target-model verify span or acceptance
 * check, so the emitted completion is NOT guaranteed to match greedy decode.
 * Matches the DS4_DSPARK_STATS env convention: enabled for any non-empty
 * value other than "0". */
static bool ds4_dist_spec_trust_all_enabled(void) {
    const char *env = getenv("DS4_DIST_SPEC_TRUST_ALL");
    return env && env[0] && strcmp(env, "0") != 0;
}

static const char *dist_role_name(ds4_distributed_role role) {
    switch (role) {
    case DS4_DISTRIBUTED_NONE:        return "none";
    case DS4_DISTRIBUTED_COORDINATOR: return "coordinator";
    case DS4_DISTRIBUTED_WORKER:      return "worker";
    }
    return "unknown";
}

static void dist_sleep_reconnect(void) {
    sleep(1);
}

static double dist_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

/* =========================================================================
 * Local File And Size Helpers
 * ========================================================================= */

static int dist_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int dist_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *p = ptr;
    uint64_t original = bytes;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int dist_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4] = {
        (uint8_t)v,
        (uint8_t)(v >> 8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24),
    };
    return dist_payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int dist_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (dist_payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) return 1;
    *v = (uint32_t)b[0] |
         ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
    return 0;
}

static int dist_payload_copy_bytes(
        FILE *src,
        FILE *dst,
        uint64_t bytes,
        uint64_t *remaining,
        char *err,
        size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory copying distributed payload");
        return 1;
    }
    int rc = 0;
    uint64_t left = bytes;
    while (left != 0) {
        const size_t n = left > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (size_t)left;
        if (fread(buf, 1, n, src) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            rc = 1;
            break;
        }
        left -= n;
    }
    free(buf);
    if (rc == 0 && remaining) *remaining -= bytes;
    return rc;
}

static int dist_copy_file_range(
        FILE *src,
        uint64_t offset,
        uint64_t bytes,
        FILE *dst,
        char *err,
        size_t errlen) {
    if (offset > (uint64_t)LLONG_MAX || fseeko(src, (off_t)offset, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to seek distributed KV shard");
        return 1;
    }
    return dist_payload_copy_bytes(src, dst, bytes, NULL, err, errlen);
}

static int dist_rewind_file(FILE *fp, const char *what, char *err, size_t errlen) {
    if (fflush(fp) != 0 || fseeko(fp, 0, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to rewind %s", what);
        return 1;
    }
    return 0;
}

static int dist_measure_file(FILE *fp, uint64_t *bytes, const char *what, char *err, size_t errlen) {
    if (!bytes) return 1;
    if (fflush(fp) != 0) {
        if (errlen) snprintf(err, errlen, "failed to flush %s", what);
        return 1;
    }
    off_t pos = ftello(fp);
    if (pos < 0) {
        if (errlen) snprintf(err, errlen, "failed to measure %s", what);
        return 1;
    }
    *bytes = (uint64_t)pos;
    return 0;
}

static FILE *dist_tmpfile_or_err(const char *what, char *err, size_t errlen) {
    FILE *fp = tmpfile();
    if (!fp && errlen) snprintf(err, errlen, "failed to create %s temp file: %s",
                                what, strerror(errno));
    return fp;
}

static bool dist_u64_add(uint64_t *acc, uint64_t add) {
    if (!acc || *acc > UINT64_MAX - add) return false;
    *acc += add;
    return true;
}

static bool dist_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return false;
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

/* =========================================================================
 * Tunable Limits
 * ========================================================================= */

static int dist_socket_buffer_bytes(void) {
    int mb = 128;
    const char *env = getenv("DS4_DIST_SOCKET_BUFFER_MB");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 0 && v <= 512) {
            mb = (int)v;
        }
    }
    return mb > 0 ? mb * 1024 * 1024 : 0;
}

static uint32_t dist_worker_prefetch_depth(void) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_WORKER_PREFETCH_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static uint32_t dist_worker_forward_window(void) {
    uint32_t depth = 4;
    const char *env = getenv("DS4_DIST_WORKER_FORWARD_WINDOW");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 64) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static bool dist_decode_profile_enabled(void) {
    return getenv("DS4_DIST_DECODE_PROFILE") != NULL;
}

static bool dist_parse_positive_u32(
        const char *s,
        const char *name,
        uint32_t *out,
        char *err,
        size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a positive integer", name);
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v == 0 || v > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", name, s);
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

/* =========================================================================
 * Activation Transport
 * =========================================================================
 *
 * The graph-slice APIs exchange float buffers. Distributed transport can leave
 * those buffers as 32-bit floats or pack them to 16/8 bits on the wire; workers
 * decode back to float before executing the next slice.
 */

static uint32_t dist_activation_bits_or_default(uint32_t bits) {
    return bits ? bits : DS4_DIST_ACTIVATION_BITS_DEFAULT;
}

static bool dist_activation_bits_valid(uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    return bits == 32u || bits == 16u || bits == 8u;
}

static bool dist_activation_wire_bytes(uint32_t bits, uint64_t values, uint32_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint64_t bytes = values * (uint64_t)(bits / 8u);
    if (bytes > UINT32_MAX) return false;
    if (out) *out = (uint32_t)bytes;
    return true;
}

static bool dist_activation_values_from_wire_bytes(uint32_t bits, uint32_t bytes, uint64_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint32_t bytes_per_value = bits / 8u;
    if (bytes_per_value == 0 || (bytes % bytes_per_value) != 0) return false;
    if (out) *out = bytes / bytes_per_value;
    return true;
}

static bool dist_activation_wire_bytes_from_f32_bytes(uint32_t bits, uint32_t f32_bytes, uint32_t *out) {
    if ((f32_bytes % (uint32_t)sizeof(float)) != 0) return false;
    return dist_activation_wire_bytes(bits, f32_bytes / (uint32_t)sizeof(float), out);
}

static uint16_t dist_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float dist_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint8_t dist_f32_to_f8_e4m3(float f) {
    const uint8_t sign = signbit(f) ? 0x80u : 0u;
    float a = fabsf(f);
    if (a == 0.0f) return sign;
    if (!isfinite(a) || a >= 240.0f) return (uint8_t)(sign | 0x77u);

    if (a < 0.001953125f) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    int exp2 = 0;
    (void)frexpf(a, &exp2);
    int exp = exp2 - 1 + 7;
    if (exp <= 0) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    float base = ldexpf(1.0f, exp2 - 1);
    int mant = (int)floorf(((a / base) - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) {
        mant = 0;
        exp++;
    }
    if (exp >= 15) return (uint8_t)(sign | 0x77u);
    return (uint8_t)(sign | (uint8_t)(exp << 3) | (uint8_t)mant);
}

static float dist_f8_e4m3_to_f32(uint8_t h) {
    const float sign = (h & 0x80u) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 3) & 0x0fu;
    const uint32_t mant = h & 0x07u;
    if (exp == 0) {
        return sign * (float)mant * 0.001953125f;
    }
    if (exp >= 15u) {
        return sign * 240.0f;
    }
    return sign * ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
}

static int dist_write_activation_payload(
        ds4_transport *transport,
        const float *src,
        uint64_t values,
        uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) return -1;
    if (values == 0) return 0;
    if (!src) return -1;
    if (bits == 32u) {
        uint32_t bytes = 0;
        if (!dist_activation_wire_bytes(bits, values, &bytes)) return -1;
        return ds4_transport_send_bulk(transport, src, bytes);
    }

    const uint64_t max_values = 1024u * 1024u;
    uint64_t cap = values < max_values ? values : max_values;
    void *buf = malloc((size_t)cap * (size_t)(bits / 8u));
    if (!buf) return -1;
    uint64_t done = 0;
    int rc = 0;
    while (done < values) {
        uint64_t n = values - done;
        if (n > cap) n = cap;
        if (bits == 16u) {
            uint16_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f16(src[done + i]);
        } else {
            uint8_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f8_e4m3(src[done + i]);
        }
        if (ds4_transport_send_bulk(transport, buf, (size_t)n * (size_t)(bits / 8u)) != 0) {
            rc = -1;
            break;
        }
        done += n;
    }
    free(buf);
    return rc;
}

/* v3 names one logical tensor with one descriptor. Pack reduced-width
 * activations contiguously so one descriptor always maps to one transport
 * message (and therefore one NHI SUBMIT_TX). */
static int dist_pack_activation_payload(
        const float *src,
        uint64_t values,
        uint32_t bits,
        const void **wire,
        void **owned,
        uint32_t *wire_bytes) {
    if (wire) *wire = NULL;
    if (owned) *owned = NULL;
    if (wire_bytes) *wire_bytes = 0;
    bits = dist_activation_bits_or_default(bits);
    uint32_t bytes = 0;
    if (!src || values == 0 ||
        !dist_activation_wire_bytes(bits, values, &bytes))
        return -1;
    if (bits == 32u) {
        if (wire) *wire = src;
        if (wire_bytes) *wire_bytes = bytes;
        return 0;
    }
    void *buf = malloc(bytes);
    if (!buf) return -1;
    if (bits == 16u) {
        uint16_t *dst = buf;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f32_to_f16(src[i]);
    } else if (bits == 8u) {
        uint8_t *dst = buf;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f32_to_f8_e4m3(src[i]);
    } else {
        free(buf);
        return -1;
    }
    if (wire) *wire = buf;
    if (owned) *owned = buf;
    if (wire_bytes) *wire_bytes = bytes;
    return 0;
}

static bool dist_transport_uses_v3(const ds4_transport *transport) {
    return transport && ds4_transport_generation(transport) != 0;
}

static void dist_bulk_none_desc(
        const ds4_transport *transport,
        uint64_t session_id,
        uint64_t request_id,
        ds4_transport_bulk_desc *desc) {
    memset(desc, 0, sizeof(*desc));
    desc->mode = DS4_TRANSPORT_BULK_NONE;
    desc->generation = ds4_transport_generation(transport);
    desc->session_id = session_id;
    desc->request_id = request_id;
}

static int dist_validate_bulk_none_desc(
        const ds4_transport *transport,
        const ds4_transport_bulk_desc *desc,
        uint64_t session_id,
        uint64_t request_id) {
    if (!transport || !desc ||
        desc->mode != DS4_TRANSPORT_BULK_NONE || desc->kind != 0 ||
        desc->generation != ds4_transport_generation(transport) ||
        desc->sequence != 0 || desc->session_id != session_id ||
        desc->request_id != request_id || desc->payload_bytes != 0 ||
        desc->element_bits != 0 || desc->frame_count != 0 ||
        desc->flags != 0 || desc->reserved[0] != 0 ||
        desc->reserved[1] != 0) {
        errno = EPROTO;
        return -1;
    }
    return 0;
}

static bool dist_nhi_unsafe_fast_enabled(void) {
    const char *value = getenv("DS4_DIST_NHI_UNSAFE_FAST");
    const bool enabled = value &&
        (!strcmp(value, "1") || !strcmp(value, "true") ||
         !strcmp(value, "yes") || !strcmp(value, "on"));
    static int warned = 0;
    if (enabled && !__atomic_exchange_n(&warned, 1, __ATOMIC_RELAXED)) {
        fprintf(stderr,
                "ds4: WARNING: DS4_DIST_NHI_UNSAFE_FAST enabled; "
                "caller-proven GPU fences and dead graph-state copies are skipped\n");
    }
    return enabled;
}

static int dist_mapped_lease_commit_after_gpu_quiesce(
        ds4_transport_lease *lease) {
    return dist_nhi_unsafe_fast_enabled()
        ? ds4_transport_lease_commit_gpu_quiesced(lease)
        : ds4_transport_lease_commit(lease);
}

static const ds4_transport_bulk_desc *dist_tx_bulk_plan_desc(
        const ds4_dist_tx_bulk_plan *plan) {
    return plan && plan->prepared ? &plan->desc : NULL;
}

static void dist_tx_bulk_plan_release(ds4_dist_tx_bulk_plan *plan) {
    if (!plan) return;
    if (plan->lease) ds4_transport_lease_release(plan->lease);
    memset(plan, 0, sizeof(*plan));
}

static void dist_tx_bulk_plan_abort(
        ds4_transport *transport,
        ds4_dist_tx_bulk_plan *plan) {
    if (!plan || !plan->prepared) return;
    if (plan->lease) {
        (void)ds4_transport_lease_abort(plan->lease);
        ds4_transport_lease_release(plan->lease);
    } else if (transport && ds4_transport_control_fd(transport) >= 0) {
        /* A wrap fallback has no lease through which to roll back the already
         * prepared sequence. Abandon this generation if production fails. */
        shutdown(ds4_transport_control_fd(transport), SHUT_RDWR);
    }
    memset(plan, 0, sizeof(*plan));
}

/* Prepare a direct mapped TX slot only for 32-bit OOB tensors. A wrapped
 * payload returns ENOTSUP from the backend; the prepared descriptor is kept
 * so the ordinary NHI copy path can send the same sequence later. */
static int dist_tx_bulk_plan_prepare(
        ds4_transport *transport,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        ds4_dist_tx_bulk_plan *plan,
        char *err,
        size_t errlen) {
    if (!plan) return -1;
    memset(plan, 0, sizeof(*plan));
    if (!transport || element_bits != 32u || payload_bytes == 0 ||
        !dist_transport_uses_v3(transport) ||
        !ds4_transport_can_oob(transport, payload_bytes) ||
        !ds4_transport_mapped_leases_supported(transport))
        return 0;
    if (ds4_transport_prepare_bulk(transport, kind, session_id, request_id,
                                   payload_bytes, element_bits, &plan->desc,
                                   err, errlen) != 0)
        return -1;
    plan->prepared = true;
    if (plan->desc.mode != DS4_TRANSPORT_BULK_NHI_OOB) {
        if (errlen) snprintf(err, errlen,
                             "mapped TX lease did not prepare an OOB descriptor");
        errno = EPROTO;
        return -1;
    }
    if (ds4_transport_tx_lease_acquire(transport, &plan->desc,
                                       &plan->lease, err, errlen) == 0)
        return 1;
    if (errno == ENOTSUP) return 0;
    return -1;
}

static int dist_decode_activation_payload(
        const void *wire,
        uint32_t bits,
        uint32_t wire_bytes,
        float **out,
        uint32_t *out_f32_bytes,
        bool *out_uses_wire,
        char *err,
        size_t errlen) {
    if (out) *out = NULL;
    if (out_f32_bytes) *out_f32_bytes = 0;
    if (out_uses_wire) *out_uses_wire = false;
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation width: %u bits", bits);
        return 1;
    }
    if (wire_bytes != 0 && !wire) {
        if (errlen) snprintf(err, errlen, "missing distributed activation payload");
        return 1;
    }

    uint64_t values = 0;
    if (!dist_activation_values_from_wire_bytes(bits, wire_bytes, &values)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation payload size");
        return 1;
    }
    const uint64_t f32_bytes64 = values * sizeof(float);
    if (f32_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed activation payload is too large");
        return 1;
    }
    const uint32_t f32_bytes = (uint32_t)f32_bytes64;
    if (bits == 32u) {
        if (out) *out = (float *)(void *)wire;
        if (out_f32_bytes) *out_f32_bytes = f32_bytes;
        if (out_uses_wire) *out_uses_wire = true;
        return 0;
    }

    float *dst = f32_bytes ? malloc(f32_bytes) : NULL;
    if (f32_bytes && !dst) {
        if (errlen) snprintf(err, errlen, "out of memory decoding distributed activations");
        return 1;
    }
    if (bits == 16u) {
        const uint16_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f16_to_f32(src[i]);
    } else {
        const uint8_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f8_e4m3_to_f32(src[i]);
    }
    if (out) *out = dst;
    if (out_f32_bytes) *out_f32_bytes = f32_bytes;
    return 0;
}

/* =========================================================================
 * TCP Framing And Connections
 * ========================================================================= */

static int dist_set_socket_low_latency(int fd) {
    int one = 1;
    int rc = 0;
    int buffer_bytes = dist_socket_buffer_bytes();
    int send_timeout_sec = 60;
    const char *timeout_env = getenv("DS4_DIST_SOCKET_TIMEOUT_SEC");
    if (timeout_env && timeout_env[0]) {
        char *end = NULL;
        long v = strtol(timeout_env, &end, 10);
        if (end != timeout_env && *end == '\0' && v > 0 && v <= 3600)
            send_timeout_sec = (int)v;
    }
    struct timeval send_tv = {
        .tv_sec = send_timeout_sec,
        .tv_usec = 0,
    };
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_tv, sizeof(send_tv)) != 0) rc = -1;
    /* Do not install a receive timeout by default.  Worker control sockets can
     * be legitimately idle while a KV snapshot is transferred on a separate
     * data connection; timing out that read drops an otherwise healthy route. */
    const char *recv_timeout_env = getenv("DS4_DIST_SOCKET_RECV_TIMEOUT_SEC");
    if (recv_timeout_env && recv_timeout_env[0]) {
        char *end = NULL;
        long v = strtol(recv_timeout_env, &end, 10);
        if (end != recv_timeout_env && *end == '\0' && v > 0 && v <= 3600) {
            struct timeval recv_tv = {
                .tv_sec = (int)v,
                .tv_usec = 0,
            };
            if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                           &recv_tv, sizeof(recv_tv)) != 0) rc = -1;
        }
    }
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
#ifdef SO_NOSIGPIPE
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) rc = -1;
#endif
    return rc;
}

#ifdef DS4_DIST_TRACE
#define DIST_DEBUG(...) do { \
    fprintf(stderr, "ds4: distributed debug: " __VA_ARGS__); \
    fputc('\n', stderr); \
} while (0)
#else
#define DIST_DEBUG(...) ((void)0)
#endif

static int dist_write_full(int fd, const void *buf, size_t len) {
    return ds4_transport_tcp_write(fd, buf, len);
}

static int dist_read_full(int fd, void *buf, size_t len) {
    return ds4_transport_tcp_read(fd, buf, len);
}

static bool dist_connect_trace_enabled(void) {
    return getenv("DS4_DIST_CONNECT_TRACE") != NULL;
}

static void dist_sockaddr_string(const struct sockaddr *sa, socklen_t salen, char *buf, size_t buflen) {
    if (buflen) buf[0] = '\0';
    if (!sa || !buf || buflen == 0) return;
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getnameinfo(sa, salen,
                    host, sizeof(host),
                    service, sizeof(service),
                    NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
        snprintf(buf, buflen, "%s:%s", host, service);
        return;
    }
    snprintf(buf, buflen, "unknown");
}

static int dist_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    ds4_dist_frame_header h = {
        htonl(DS4_DIST_MAGIC),
        htonl(type),
        htonl(bytes)
    };
    return dist_write_full(fd, &h, sizeof(h));
}

static int dist_read_frame_header(int fd, uint32_t *type, uint32_t *bytes, char *err, size_t errlen) {
    ds4_dist_frame_header h;
    int rc = dist_read_full(fd, &h, sizeof(h));
    if (rc < 0 && errlen) snprintf(err, errlen, "failed to read frame header: %s", strerror(errno));
    if (rc <= 0) return rc;

    uint32_t magic = ntohl(h.magic);
    if (magic != DS4_DIST_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        return -1;
    }

    *type = ntohl(h.type);
    *bytes = ntohl(h.bytes);
    return 1;
}

static int dist_discard_bytes(int fd, uint32_t bytes) {
    unsigned char buf[4096];
    while (bytes > 0) {
        size_t n = bytes < sizeof(buf) ? bytes : sizeof(buf);
        int rc = dist_read_full(fd, buf, n);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        bytes -= (uint32_t)n;
    }
    return 1;
}

static int dist_send_error(int fd, const char *msg) {
    if (!msg) msg = "distributed protocol error";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_ERROR, (uint32_t)len) != 0) return -1;
    return dist_write_full(fd, msg, len);
}

static void dist_peer_name(int fd, char *host, size_t hostlen, char *port, size_t portlen) {
    if (hostlen) host[0] = '\0';
    if (portlen) port[0] = '\0';

    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &slen) == 0) {
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        host, (socklen_t)hostlen,
                        port, (socklen_t)portlen,
                        NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
            return;
        }
    }
    if (hostlen) snprintf(host, hostlen, "unknown");
    if (portlen) snprintf(port, portlen, "0");
}

static int dist_open_listener(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    const char *host_display = host ? host : "*";

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host_display, portbuf, gai_strerror(gai));
        return -1;
    }

    int listen_fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        dist_set_socket_low_latency(fd);

        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(fd, 64) == 0) {
            listen_fd = fd;
            break;
        }

        close(fd);
    }

    freeaddrinfo(res);
    if (listen_fd < 0 && errlen) {
        snprintf(err, errlen, "unable to listen on %s:%d: %s", host_display, port, strerror(errno));
    }
    return listen_fd;
}

static int dist_listener_port(int fd) {
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getsockname(fd, (struct sockaddr *)&ss, &slen) != 0) return 0;
    char service[NI_MAXSERV];
    if (getnameinfo((struct sockaddr *)&ss, slen,
                    NULL, 0,
                    service, sizeof(service),
                    NI_NUMERICSERV) != 0) {
        return 0;
    }
    char *end = NULL;
    unsigned long v = strtoul(service, &end, 10);
    if (end == service || *end != '\0' || v > 65535ul) return 0;
    return (int)v;
}

static bool dist_connect_errno_retryable(int e) {
    return e == ECONNREFUSED ||
           e == EHOSTUNREACH ||
           e == ENETUNREACH ||
           e == ETIMEDOUT ||
           e == EADDRNOTAVAIL;
}

static int dist_bind_connect_source(int fd, int family, const char *host, int *bind_errno) {
    if (bind_errno) *bind_errno = 0;
    if (!host || !host[0]) return 0;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = family;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, "0", &hints, &res);
    if (gai != 0) {
        if (bind_errno) *bind_errno = EINVAL;
        return -1;
    }

    int rc = -1;
    int saved_errno = EADDRNOTAVAIL;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            rc = 0;
            saved_errno = 0;
            break;
        }
        saved_errno = errno;
    }
    freeaddrinfo(res);

    if (bind_errno) *bind_errno = saved_errno;
    return rc;
}

static int dist_bind_connect_interface(int fd, int family, const char *ifname, int *bind_errno) {
    (void)fd;
    if (bind_errno) *bind_errno = 0;
    if (!ifname || !ifname[0]) return 0;

    unsigned int ifindex = if_nametoindex(ifname);
    if (ifindex == 0) {
        if (bind_errno) *bind_errno = errno ? errno : ENODEV;
        return -1;
    }

    int rc = 0;
    if (family == AF_INET) {
#ifdef IP_BOUND_IF
        rc = setsockopt(fd, IPPROTO_IP, IP_BOUND_IF, &ifindex, sizeof(ifindex));
#else
        errno = ENOTSUP;
        rc = -1;
#endif
    } else if (family == AF_INET6) {
#ifdef IPV6_BOUND_IF
        rc = setsockopt(fd, IPPROTO_IPV6, IPV6_BOUND_IF, &ifindex, sizeof(ifindex));
#else
        errno = ENOTSUP;
        rc = -1;
#endif
    }

    if (rc != 0) {
        if (bind_errno) *bind_errno = errno ? errno : EIO;
        return -1;
    }
    return 0;
}

static int dist_connect_endpoint_once(const char *host, int port, int *last_errno, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    if (last_errno) *last_errno = 0;
    const char *bind_host = getenv("DS4_DIST_CONNECT_BIND_HOST");
    if (bind_host && bind_host[0] == '\0') bind_host = NULL;
    const char *bind_if = getenv("DS4_DIST_CONNECT_BIND_IF");
    if (bind_if && bind_if[0] == '\0') bind_if = NULL;
    const bool trace = dist_connect_trace_enabled();

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host, portbuf, gai_strerror(gai));
        if (last_errno) *last_errno = EINVAL;
        return -1;
    }

    int fd = -1;
    int saved_errno = 0;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        char dst_desc[NI_MAXHOST + NI_MAXSERV + 8];
        if (trace) {
            dist_sockaddr_string(ai->ai_addr, ai->ai_addrlen, dst_desc, sizeof(dst_desc));
            fprintf(stderr, "ds4: distributed connect trace: candidate dst=%s family=%d bind=%s if=%s\n",
                    dst_desc, ai->ai_family, bind_host ? bind_host : "(none)",
                    bind_if ? bind_if : "(none)");
        }
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            saved_errno = errno;
            if (trace) {
                fprintf(stderr, "ds4: distributed connect trace: socket failed: %s\n",
                        strerror(saved_errno));
            }
            continue;
        }
        dist_set_socket_low_latency(fd);
        if (bind_if) {
            int bind_errno = 0;
            if (dist_bind_connect_interface(fd, ai->ai_family, bind_if, &bind_errno) != 0) {
                saved_errno = bind_errno ? bind_errno : ENODEV;
                if (trace) {
                    fprintf(stderr, "ds4: distributed connect trace: bind interface %s failed: %s\n",
                            bind_if, strerror(saved_errno));
                }
                close(fd);
                fd = -1;
                continue;
            }
            if (trace) {
                fprintf(stderr, "ds4: distributed connect trace: bound interface=%s\n", bind_if);
            }
        }
        if (bind_host) {
            int bind_errno = 0;
            if (dist_bind_connect_source(fd, ai->ai_family, bind_host, &bind_errno) != 0) {
                saved_errno = bind_errno ? bind_errno : EADDRNOTAVAIL;
                if (trace) {
                    fprintf(stderr, "ds4: distributed connect trace: bind source %s failed: %s\n",
                            bind_host, strerror(saved_errno));
                }
                close(fd);
                fd = -1;
                continue;
            }
            if (trace) {
                struct sockaddr_storage local_ss;
                socklen_t local_len = sizeof(local_ss);
                char local_desc[NI_MAXHOST + NI_MAXSERV + 8];
                if (getsockname(fd, (struct sockaddr *)&local_ss, &local_len) == 0) {
                    dist_sockaddr_string((const struct sockaddr *)&local_ss, local_len,
                                         local_desc, sizeof(local_desc));
                    fprintf(stderr, "ds4: distributed connect trace: bound source=%s\n",
                            local_desc);
                }
            }
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            if (trace) {
                struct sockaddr_storage local_ss;
                socklen_t local_len = sizeof(local_ss);
                char local_desc[NI_MAXHOST + NI_MAXSERV + 8];
                if (getsockname(fd, (struct sockaddr *)&local_ss, &local_len) == 0) {
                    dist_sockaddr_string((const struct sockaddr *)&local_ss, local_len,
                                         local_desc, sizeof(local_desc));
                    fprintf(stderr, "ds4: distributed connect trace: connected source=%s dst=%s\n",
                            local_desc, dst_desc);
                }
            }
            break;
        }
        saved_errno = errno;
        if (trace) {
            fprintf(stderr, "ds4: distributed connect trace: connect to %s failed: %s\n",
                    dst_desc, strerror(saved_errno));
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    if (fd < 0 && errlen) {
        if (saved_errno == 0) saved_errno = EIO;
        snprintf(err, errlen, "unable to connect to %s:%d: %s", host, port, strerror(saved_errno));
    }
    if (fd < 0 && last_errno) *last_errno = saved_errno ? saved_errno : EIO;
    return fd;
}

static int dist_connect_endpoint(const char *host, int port, char *err, size_t errlen) {
    int last_errno = 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        int fd = dist_connect_endpoint_once(host, port, &last_errno, err, errlen);
        if (fd >= 0) return fd;
        if (!dist_connect_errno_retryable(last_errno)) break;
        struct timespec ts = {0, 25 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    return -1;
}

/* =========================================================================
 * Wire Encoding Helpers
 * ========================================================================= */

static void dist_hello_to_wire(ds4_dist_hello_fixed *h) {
    h->model_id = htonl(h->model_id);
    h->quant_bits = htonl(h->quant_bits);
    h->layer_start = htonl(h->layer_start);
    h->layer_end = htonl(h->layer_end);
    h->has_output = htonl(h->has_output);
    h->has_hidden = htonl(h->has_hidden);
    h->ctx_size = htonl(h->ctx_size);
    h->n_layers = htonl(h->n_layers);
    h->listen_port = htonl(h->listen_port);
    h->model_name_len = htonl(h->model_name_len);
}

static void dist_hello_from_wire(ds4_dist_hello_fixed *h) {
    h->model_id = ntohl(h->model_id);
    h->quant_bits = ntohl(h->quant_bits);
    h->layer_start = ntohl(h->layer_start);
    h->layer_end = ntohl(h->layer_end);
    h->has_output = ntohl(h->has_output);
    h->has_hidden = ntohl(h->has_hidden);
    h->ctx_size = ntohl(h->ctx_size);
    h->n_layers = ntohl(h->n_layers);
    h->listen_port = ntohl(h->listen_port);
    h->model_name_len = ntohl(h->model_name_len);
}

static uint64_t dist_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}

/* FNV-1a over little-endian token IDs.  This is not a security primitive; it is
 * a compact session invariant so distributed workers can reject same-position
 * but different-prefix KV state before doing layer work. */
#define DS4_DIST_TOKEN_HASH_INIT 1469598103934665603ull
#define DS4_DIST_TOKEN_HASH_PRIME 1099511628211ull

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= DS4_DIST_TOKEN_HASH_PRIME;
    }
    return h;
}

static uint64_t dist_token_hash_update_span(uint64_t h, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) h = dist_token_hash_update(h, tokens[i]);
    return h;
}

static uint64_t dist_token_hash_prefix(const int *tokens, uint32_t n_tokens) {
    return dist_token_hash_update_span(DS4_DIST_TOKEN_HASH_INIT, tokens, n_tokens);
}

static int dist_session_token_hash_prefix(
        ds4_session *session,
        uint32_t n_tokens,
        uint64_t *hash,
        char *err,
        size_t errlen) {
    const ds4_tokens *checkpoint = ds4_session_tokens(session);
    if (!hash || !checkpoint || checkpoint->len < 0 || (uint32_t)checkpoint->len < n_tokens) {
        if (errlen) snprintf(err, errlen, "distributed session has no %u-token prefix", n_tokens);
        return 1;
    }
    *hash = dist_token_hash_prefix(checkpoint->v, n_tokens);
    return 0;
}

static bool dist_bytes_have_nul(const void *p, uint32_t len) {
    return len != 0 && memchr(p, '\0', len) != NULL;
}

static void dist_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static void dist_work_from_wire(ds4_dist_work_fixed *w) {
    w->model_id = ntohl(w->model_id);
    w->session_hi = ntohl(w->session_hi);
    w->session_lo = ntohl(w->session_lo);
    w->request_hi = ntohl(w->request_hi);
    w->request_lo = ntohl(w->request_lo);
    w->prefix_hash_hi = ntohl(w->prefix_hash_hi);
    w->prefix_hash_lo = ntohl(w->prefix_hash_lo);
    w->result_hash_hi = ntohl(w->result_hash_hi);
    w->result_hash_lo = ntohl(w->result_hash_lo);
    w->pos0 = ntohl(w->pos0);
    w->n_tokens = ntohl(w->n_tokens);
    w->layer_start = ntohl(w->layer_start);
    w->layer_end = ntohl(w->layer_end);
    w->flags = ntohl(w->flags);
    w->token_bytes = ntohl(w->token_bytes);
    w->input_hc_bytes = ntohl(w->input_hc_bytes);
    w->input_hc_bits = ntohl(w->input_hc_bits);
    w->route_count = ntohl(w->route_count);
    w->route_index = ntohl(w->route_index);
    w->route_bytes = ntohl(w->route_bytes);
}

static void dist_work_to_wire(ds4_dist_work_fixed *w) {
    w->model_id = htonl(w->model_id);
    w->session_hi = htonl(w->session_hi);
    w->session_lo = htonl(w->session_lo);
    w->request_hi = htonl(w->request_hi);
    w->request_lo = htonl(w->request_lo);
    w->prefix_hash_hi = htonl(w->prefix_hash_hi);
    w->prefix_hash_lo = htonl(w->prefix_hash_lo);
    w->result_hash_hi = htonl(w->result_hash_hi);
    w->result_hash_lo = htonl(w->result_hash_lo);
    w->pos0 = htonl(w->pos0);
    w->n_tokens = htonl(w->n_tokens);
    w->layer_start = htonl(w->layer_start);
    w->layer_end = htonl(w->layer_end);
    w->flags = htonl(w->flags);
    w->token_bytes = htonl(w->token_bytes);
    w->input_hc_bytes = htonl(w->input_hc_bytes);
    w->input_hc_bits = htonl(w->input_hc_bits);
    w->route_count = htonl(w->route_count);
    w->route_index = htonl(w->route_index);
    w->route_bytes = htonl(w->route_bytes);
}

static void dist_route_from_wire(ds4_dist_route_fixed *r) {
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
    r->layer_start = ntohl(r->layer_start);
    r->layer_end = ntohl(r->layer_end);
    r->flags = ntohl(r->flags);
}

static void dist_route_to_wire(ds4_dist_route_fixed *r) {
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
    r->layer_start = htonl(r->layer_start);
    r->layer_end = htonl(r->layer_end);
    r->flags = htonl(r->flags);
}

static void dist_route_return_from_wire(ds4_dist_route_return_fixed *r) {
    r->kind = ntohl(r->kind);
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
}

static void dist_route_return_to_wire(ds4_dist_route_return_fixed *r) {
    r->kind = htonl(r->kind);
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
}

static void dist_result_to_wire(ds4_dist_result_fixed *r) {
    r->request_hi = htonl(r->request_hi);
    r->request_lo = htonl(r->request_lo);
    r->result_hash_hi = htonl(r->result_hash_hi);
    r->result_hash_lo = htonl(r->result_hash_lo);
    r->status = htonl(r->status);
    r->result_kind = htonl(r->result_kind);
    r->telemetry_count = htonl(r->telemetry_count);
    r->telemetry_bytes = htonl(r->telemetry_bytes);
    r->payload_bytes = htonl(r->payload_bytes);
    r->payload_bits = htonl(r->payload_bits);
}

static void dist_result_from_wire(ds4_dist_result_fixed *r) {
    r->request_hi = ntohl(r->request_hi);
    r->request_lo = ntohl(r->request_lo);
    r->result_hash_hi = ntohl(r->result_hash_hi);
    r->result_hash_lo = ntohl(r->result_hash_lo);
    r->status = ntohl(r->status);
    r->result_kind = ntohl(r->result_kind);
    r->telemetry_count = ntohl(r->telemetry_count);
    r->telemetry_bytes = ntohl(r->telemetry_bytes);
    r->payload_bytes = ntohl(r->payload_bytes);
    r->payload_bits = ntohl(r->payload_bits);
}

static void dist_snapshot_req_to_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
}

static void dist_snapshot_req_from_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
}

static void dist_snapshot_begin_to_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
    s->payload_hi = htonl(s->payload_hi);
    s->payload_lo = htonl(s->payload_lo);
    s->status = htonl(s->status);
    s->token_bytes = htonl(s->token_bytes);
    s->message_bytes = htonl(s->message_bytes);
}

static void dist_snapshot_begin_from_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
    s->payload_hi = ntohl(s->payload_hi);
    s->payload_lo = ntohl(s->payload_lo);
    s->status = ntohl(s->status);
    s->token_bytes = ntohl(s->token_bytes);
    s->message_bytes = ntohl(s->message_bytes);
}

static void dist_snapshot_chunk_to_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->chunk_bytes = htonl(s->chunk_bytes);
}

static void dist_snapshot_chunk_from_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->chunk_bytes = ntohl(s->chunk_bytes);
}

static void dist_snapshot_done_to_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->status = htonl(s->status);
    s->message_bytes = htonl(s->message_bytes);
}

static void dist_snapshot_done_from_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->status = ntohl(s->status);
    s->message_bytes = ntohl(s->message_bytes);
}

static void dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = htonl(t->layer_start);
    t->layer_end = htonl(t->layer_end);
    t->route_index = htonl(t->route_index);
    t->pos0 = htonl(t->pos0);
    t->n_tokens = htonl(t->n_tokens);
    t->eval_usec = htonl(t->eval_usec);
    t->downstream_wait_usec = htonl(t->downstream_wait_usec);
    t->forward_send_usec = htonl(t->forward_send_usec);
    t->input_bytes = htonl(t->input_bytes);
    t->output_bytes = htonl(t->output_bytes);
}

static void dist_telemetry_from_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = ntohl(t->layer_start);
    t->layer_end = ntohl(t->layer_end);
    t->route_index = ntohl(t->route_index);
    t->pos0 = ntohl(t->pos0);
    t->n_tokens = ntohl(t->n_tokens);
    t->eval_usec = ntohl(t->eval_usec);
    t->downstream_wait_usec = ntohl(t->downstream_wait_usec);
    t->forward_send_usec = ntohl(t->forward_send_usec);
    t->input_bytes = ntohl(t->input_bytes);
    t->output_bytes = ntohl(t->output_bytes);
}

static uint32_t dist_usec_since(double t0, double t1) {
    if (t1 <= t0) return 0;
    const double usec = (t1 - t0) * 1000000.0;
    if (usec >= (double)UINT32_MAX) return UINT32_MAX;
    return (uint32_t)(usec + 0.5);
}

/* =========================================================================
 * Worker Registration
 * ========================================================================= */

static int dist_send_hello(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size, uint32_t listen_port, int fd,
                           const ds4_dist_v3_hello_ext *v3_ext) {
    uint32_t n_layers = (uint32_t)ds4_engine_layer_count(engine);
    const char *model_name = ds4_engine_model_name(engine);
    if (!model_name) model_name = "unknown";
    size_t model_name_len = strlen(model_name);
    if (model_name_len > DS4_DIST_MAX_MODEL_NAME) model_name_len = DS4_DIST_MAX_MODEL_NAME;

    ds4_dist_hello_fixed h = {
        (uint32_t)ds4_engine_model_id(engine),
        (uint32_t)ds4_engine_routed_quant_bits(engine),
        opt->layers.start,
        dist_resolved_layer_end(opt, n_layers),
        opt->layers.has_output ? 1u : 0u,
        1u,
        ctx_size > 0 ? (uint32_t)ctx_size : 0u,
        n_layers,
        listen_port,
        (uint32_t)model_name_len
    };
    ds4_dist_hello_fixed wire = h;
    dist_hello_to_wire(&wire);

    uint32_t bytes = (uint32_t)sizeof(wire) + (uint32_t)model_name_len;
    if (v3_ext) bytes += (uint32_t)sizeof(*v3_ext);
    if (dist_write_frame_header(fd,
                                v3_ext ? DS4_DIST_MSG_HELLO_V3
                                       : DS4_DIST_MSG_HELLO,
                                bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (v3_ext) {
        ds4_dist_v3_hello_ext ext_wire;
        ds4_dist_v3_hello_ext_to_wire(&ext_wire, v3_ext);
        if (dist_write_full(fd, &ext_wire, sizeof(ext_wire)) != 0) return -1;
    }
    if (model_name_len && dist_write_full(fd, model_name, model_name_len) != 0) return -1;
    return 0;
}

static int dist_recv_hello(int fd, ds4_dist_hello_fixed *hello,
                           ds4_dist_v3_hello_ext *v3_ext, bool *is_v3,
                           char *model_name, size_t model_name_cap, char *err, size_t errlen) {
    if (v3_ext) memset(v3_ext, 0, sizeof(*v3_ext));
    if (is_v3) *is_v3 = false;
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) return rc;
    if (type != DS4_DIST_MSG_HELLO && type != DS4_DIST_MSG_HELLO_V3) {
        if (errlen) snprintf(err, errlen, "expected HELLO frame, got type %u", type);
        dist_discard_bytes(fd, bytes);
        return -1;
    }
    const bool got_v3 = type == DS4_DIST_MSG_HELLO_V3;
    const uint32_t ext_bytes = got_v3
        ? (uint32_t)sizeof(ds4_dist_v3_hello_ext) : 0u;
    const uint64_t min_bytes = (uint64_t)sizeof(*hello) + ext_bytes;
    const uint64_t max_bytes = min_bytes + DS4_DIST_MAX_MODEL_NAME;
    if ((uint64_t)bytes < min_bytes || (uint64_t)bytes > max_bytes) {
        if (errlen) snprintf(err, errlen, "invalid HELLO payload length %u", bytes);
        dist_discard_bytes(fd, bytes);
        return -1;
    }

    ds4_dist_hello_fixed wire;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_hello_from_wire(&wire);

    uint32_t remaining = bytes - ((uint32_t)sizeof(wire) + ext_bytes);
    if (wire.model_name_len != remaining || wire.model_name_len > DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO model name length %u", wire.model_name_len);
        dist_discard_bytes(fd, remaining);
        return -1;
    }
    if (got_v3) {
        ds4_dist_v3_hello_ext ext_wire;
        rc = dist_read_full(fd, &ext_wire, sizeof(ext_wire));
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (v3_ext) ds4_dist_v3_hello_ext_from_wire(v3_ext, &ext_wire);
    }
    if (is_v3) *is_v3 = got_v3;

    if (model_name_cap) model_name[0] = '\0';
    if (wire.model_name_len) {
        char tmp[DS4_DIST_MAX_MODEL_NAME + 1u];
        rc = dist_read_full(fd, tmp, wire.model_name_len);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (dist_bytes_have_nul(tmp, wire.model_name_len)) {
            if (errlen) snprintf(err, errlen, "HELLO model family contains NUL bytes");
            return -1;
        }
        tmp[wire.model_name_len] = '\0';
        if (model_name_cap) {
            snprintf(model_name, model_name_cap, "%s", tmp);
        }
    }

    *hello = wire;
    return 1;
}

static uint32_t dist_nhi_offer_max_payload(const ds4_transport *transport) {
    if (!transport) return 0;
    const size_t payload = ds4_transport_max_oob_bytes(transport);
    return payload > UINT32_MAX ? UINT32_MAX : (uint32_t)payload;
}

static ds4_dist_v3_hello_ext dist_v3_local_offer(
        ds4_distributed_transport policy,
        const ds4_transport *nhi) {
    ds4_dist_v3_hello_ext offer;
    memset(&offer, 0, sizeof(offer));
    offer.protocol_min = DS4_DIST_V3_PROTOCOL_VERSION;
    offer.protocol_max = DS4_DIST_V3_PROTOCOL_VERSION;
    offer.capabilities = DS4_DIST_V3_CAP_BULK_DESC_V1 |
                         DS4_DIST_V3_CAP_SPEC_DECODE_V1 |
                         DS4_DIST_V3_CAP_SPEC_EXACT_V1;
    offer.transport_policy = policy == DS4_DIST_TRANSPORT_NHI
        ? DS4_DIST_V3_POLICY_REQUIRE_NHI : DS4_DIST_V3_POLICY_AUTO;
    if (nhi) {
        offer.capabilities |= DS4_DIST_V3_CAP_NHI_V1;
        offer.nhi_frame_size = ds4_transport_frame_size(nhi);
        offer.nhi_ring_size = ds4_transport_ring_size(nhi);
        offer.nhi_max_payload = dist_nhi_offer_max_payload(nhi);
    }
    return offer;
}

static int dist_send_hello_ack(
        int fd,
        const ds4_dist_v3_hello_ack *ack) {
    ds4_dist_v3_hello_ack wire;
    ds4_dist_v3_hello_ack_to_wire(&wire, ack);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_HELLO_ACK,
                                (uint32_t)sizeof(wire)) != 0)
        return -1;
    return dist_write_full(fd, &wire, sizeof(wire));
}

static int dist_recv_hello_ack(
        int fd,
        const ds4_dist_v3_hello_ext *local_offer,
        ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen) {
    uint32_t type = 0;
    uint32_t bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "peer closed before v3 HELLO ACK");
        return rc;
    }
    if (type == DS4_DIST_MSG_ERROR) {
        uint32_t n = bytes;
        if (err && errlen) {
            n = n < errlen - 1u ? n : (uint32_t)errlen - 1u;
            rc = dist_read_full(fd, err, n);
            if (rc <= 0) return rc;
            err[n] = '\0';
        } else {
            n = 0;
        }
        if (bytes > n && dist_discard_bytes(fd, bytes - n) <= 0) return -1;
        errno = EPROTONOSUPPORT;
        return -2;
    }
    if (type != DS4_DIST_MSG_HELLO_ACK || bytes != sizeof(*ack)) {
        (void)dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "expected v3 HELLO ACK, got type %u length %u",
                             type, bytes);
        errno = EPROTO;
        return -1;
    }
    ds4_dist_v3_hello_ack wire;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc;
    ds4_dist_v3_hello_ack_from_wire(ack, &wire);
    return ds4_dist_v3_hello_ack_validate(ack, local_offer, err, errlen) == 0
        ? 1 : -1;
}

static int dist_send_hello_ready(
        int fd,
        const ds4_dist_v3_hello_ack *ack) {
    ds4_dist_v3_hello_ready ready;
    ds4_dist_v3_hello_ready wire;
    memset(&ready, 0, sizeof(ready));
    ready.protocol_version = ack->protocol_version;
    ready.selected_transport = ack->selected_transport;
    ready.generation_hi = ack->generation_hi;
    ready.generation_lo = ack->generation_lo;
    ds4_dist_v3_hello_ready_to_wire(&wire, &ready);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_HELLO_READY,
                                (uint32_t)sizeof(wire)) != 0)
        return -1;
    return dist_write_full(fd, &wire, sizeof(wire));
}

static int dist_recv_hello_ready(
        int fd,
        const ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen) {
    uint32_t type = 0;
    uint32_t bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen)
            snprintf(err, errlen, "peer closed before v3 HELLO READY");
        return rc;
    }
    if (type != DS4_DIST_MSG_HELLO_READY ||
        bytes != DS4_DIST_V3_HELLO_READY_BYTES) {
        (void)dist_discard_bytes(fd, bytes);
        if (errlen)
            snprintf(err, errlen,
                     "expected v3 HELLO READY, got type %u length %u",
                     type, bytes);
        errno = EPROTO;
        return -1;
    }
    ds4_dist_v3_hello_ready wire;
    ds4_dist_v3_hello_ready ready;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc;
    ds4_dist_v3_hello_ready_from_wire(&ready, &wire);
    return ds4_dist_v3_hello_ready_validate(&ready, ack, err, errlen) == 0
        ? 1 : -1;
}

static uint64_t dist_coordinator_next_link_generation(
        ds4_dist_coordinator_state *state) {
    pthread_mutex_lock(&state->mu);
    if (state->next_link_generation == 0) {
        struct timespec realtime;
        struct timespec monotonic;
        clock_gettime(CLOCK_REALTIME, &realtime);
        clock_gettime(CLOCK_MONOTONIC, &monotonic);
        uint64_t seed = ((uint64_t)realtime.tv_sec << 32) ^
            (uint64_t)realtime.tv_nsec ^
            ((uint64_t)monotonic.tv_sec << 17) ^
            (uint64_t)monotonic.tv_nsec ^
            ((uint64_t)(uint32_t)getpid() << 7) ^
            (uint64_t)(uintptr_t)state;
        if (seed == 0 || seed == UINT64_MAX) seed = 1;
        state->next_link_generation = seed;
    }
    state->next_link_generation++;
    if (state->next_link_generation == 0)
        state->next_link_generation = 1;
    const uint64_t generation = state->next_link_generation;
    pthread_mutex_unlock(&state->mu);
    return generation;
}

static ds4_transport *dist_worker_negotiate_link(
        ds4_engine *engine,
        const ds4_dist_options *opt,
        int ctx_size,
        uint32_t listen_port,
        int fd,
        bool use_v3,
        bool suppress_nhi,
        bool *legacy_retry,
        bool *v3_tcp_retry,
        uint32_t *selected_caps_out,
        char *err,
        size_t errlen) {
    if (legacy_retry) *legacy_retry = false;
    if (v3_tcp_retry) *v3_tcp_retry = false;
    if (selected_caps_out) *selected_caps_out = 0;
    if (!use_v3) {
        if (dist_send_hello(engine, opt, ctx_size, listen_port, fd, NULL) != 0) {
            if (errlen) snprintf(err, errlen, "failed to send v2 HELLO: %s",
                                 strerror(errno));
            return NULL;
        }
        return ds4_transport_tcp_create(fd, err, errlen);
    }

    ds4_transport *candidate = NULL;
    if (opt->nhi_device && !suppress_nhi) {
        candidate = ds4_transport_nhi_create(fd, opt->nhi_device, err, errlen);
        if (!candidate && opt->transport == DS4_DIST_TRANSPORT_NHI)
            return NULL;
        if (!candidate) {
            fprintf(stderr,
                    "ds4: distributed worker: NHI candidate %s unavailable: %s; offering v3 TCP\n",
                    opt->nhi_device,
                    err && err[0] ? err : strerror(errno));
        }
    }
    ds4_dist_v3_hello_ext offer = dist_v3_local_offer(opt->transport, candidate);
    if (ds4_dist_v3_hello_ext_validate(&offer, err, errlen) != 0) {
        ds4_transport_release(candidate);
        return NULL;
    }
    if (dist_send_hello(engine, opt, ctx_size, listen_port, fd, &offer) != 0) {
        if (errlen) snprintf(err, errlen, "failed to send v3 HELLO: %s",
                             strerror(errno));
        ds4_transport_release(candidate);
        return NULL;
    }

    ds4_dist_v3_hello_ack ack;
    const int ack_rc = dist_recv_hello_ack(fd, &offer, &ack, err, errlen);
    if (ack_rc != 1) {
        if (legacy_retry && opt->transport == DS4_DIST_TRANSPORT_AUTO &&
            ack_rc != -2)
            *legacy_retry = true;
        ds4_transport_release(candidate);
        return NULL;
    }
    const uint64_t generation = ds4_dist_v3_u64_from_halves(
        ack.generation_hi, ack.generation_lo);
    ds4_transport *selected = NULL;
    if (ack.selected_transport == DS4_DIST_V3_TRANSPORT_NHI) {
        if (!candidate) {
            if (errlen) snprintf(err, errlen,
                                 "v3 peer selected NHI without a local NHI endpoint");
            errno = EPROTO;
            return NULL;
        }
        if (ds4_transport_configure_link(candidate, generation,
                                         ack.nhi_frame_size,
                                         ack.nhi_ring_size,
                                         ack.nhi_max_payload,
                                         err, errlen) != 0) {
            ds4_transport_release(candidate);
            if (v3_tcp_retry &&
                opt->transport == DS4_DIST_TRANSPORT_AUTO)
                *v3_tcp_retry = true;
            return NULL;
        }
        selected = candidate;
    } else {
        ds4_transport_release(candidate);
        selected = ds4_transport_tcp_create(fd, err, errlen);
        if (!selected) return NULL;
        if (ds4_transport_configure_link(selected, generation, 0, 0, 0,
                                         err, errlen) != 0) {
            ds4_transport_release(selected);
            return NULL;
        }
    }
    if (dist_send_hello_ready(fd, &ack) != 0) {
        if (errlen)
            snprintf(err, errlen, "failed to send v3 HELLO READY: %s",
                     strerror(errno));
        ds4_transport_release(selected);
        return NULL;
    }
    if (selected_caps_out) *selected_caps_out = ack.selected_caps;
    fprintf(stderr,
            "ds4: distributed worker: negotiated protocol v3 transport=%s generation=%llu\n",
            ds4_transport_name(selected),
            (unsigned long long)generation);
    return selected;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state);

static bool dist_coordinator_debug_enabled(const ds4_dist_coordinator_state *state) {
    return state && state->debug;
}

#define DIST_COORD_DEBUG(state, ...) do { \
    if (dist_coordinator_debug_enabled(state)) fprintf(stderr, __VA_ARGS__); \
} while (0)

/* =========================================================================
 * Coordinator Worker Registry And Route Planning
 * =========================================================================
 *
 * A route is a contiguous chain that starts after the coordinator's local
 * slice. The last hop can either return logits directly or return the final
 * hidden state so the coordinator can run its local output head.
 */

static ds4_transport *dist_coordinator_negotiate_v3(
        ds4_dist_coordinator_state *state,
        int fd,
        const ds4_dist_v3_hello_ext *worker_offer,
        uint32_t *selected_caps_out,
        char *err,
        size_t errlen) {
    if (selected_caps_out) *selected_caps_out = 0;
    ds4_transport *candidate = NULL;
    const bool try_nhi =
        state->transport_policy == DS4_DIST_TRANSPORT_AUTO ||
        state->transport_policy == DS4_DIST_TRANSPORT_NHI;
    if (try_nhi && state->nhi_device) {
        candidate = ds4_transport_nhi_dup(fd, state->nhi_device, err, errlen);
        if (!candidate) {
            if (state->debug) {
                fprintf(stderr,
                        "ds4: distributed coordinator: NHI candidate %s unavailable: %s\n",
                        state->nhi_device,
                        err && err[0] ? err : strerror(errno));
            }
        }
    }

    ds4_dist_v3_hello_ext local_offer =
        dist_v3_local_offer(state->transport_policy, candidate);
    const uint64_t generation = dist_coordinator_next_link_generation(state);
    ds4_dist_v3_hello_ack ack;
    if (ds4_dist_v3_negotiate(worker_offer, &local_offer, generation,
                              &ack, err, errlen) != 0) {
        ds4_transport_release(candidate);
        return NULL;
    }

    ds4_transport *selected = NULL;
    if (ack.selected_transport == DS4_DIST_V3_TRANSPORT_NHI) {
        if (!candidate ||
            ds4_transport_configure_link(candidate, generation,
                                         ack.nhi_frame_size,
                                         ack.nhi_ring_size,
                                         ack.nhi_max_payload,
                                         err, errlen) != 0) {
            ds4_transport_release(candidate);
            candidate = NULL;
            /* AUTO can still select descriptor-framed TCP, but the decision
             * is made before ACK and before any OOB descriptor is visible. */
            if (state->transport_policy == DS4_DIST_TRANSPORT_NHI)
                return NULL;
            local_offer = dist_v3_local_offer(DS4_DIST_TRANSPORT_AUTO, NULL);
            if (ds4_dist_v3_negotiate(worker_offer, &local_offer, generation,
                                      &ack, err, errlen) != 0)
                return NULL;
        } else {
            selected = candidate;
            candidate = NULL;
        }
    }
    if (!selected) {
        ds4_transport_release(candidate);
        selected = ds4_transport_tcp_dup(fd, err, errlen);
        if (!selected) return NULL;
        if (ds4_transport_configure_link(selected, generation, 0, 0, 0,
                                         err, errlen) != 0) {
            ds4_transport_release(selected);
            return NULL;
        }
    }
    if (dist_send_hello_ack(fd, &ack) != 0) {
        ds4_transport_release(selected);
        return NULL;
    }
    if (dist_recv_hello_ready(fd, &ack, err, errlen) != 1) {
        ds4_transport_release(selected);
        return NULL;
    }
    if (selected_caps_out) *selected_caps_out = ack.selected_caps;
    return selected;
}

static bool dist_coordinator_add_worker(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port,
        const ds4_dist_hello_fixed *hello,
        const char *model_name,
        ds4_transport *prepared_transport,
        uint32_t protocol_version,
        uint32_t selected_caps) {
    ds4_dist_worker_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory while registering worker\n");
        return false;
    }

    entry->fd = fd;
    entry->transport = prepared_transport
        ? ds4_transport_retain(prepared_transport)
        : ds4_transport_tcp_dup(fd, NULL, 0);
    if (!entry->transport) {
        free(entry);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: failed to create persistent worker transport\n");
        return false;
    }
    snprintf(entry->peer_host, sizeof(entry->peer_host), "%s", peer_host);
    snprintf(entry->peer_port, sizeof(entry->peer_port), "%s", peer_port);
    snprintf(entry->model_name, sizeof(entry->model_name), "%s", model_name ? model_name : "unknown");
    entry->model_id = hello->model_id;
    entry->quant_bits = hello->quant_bits;
    entry->layer_start = hello->layer_start;
    entry->layer_end = hello->layer_end;
    entry->has_output = hello->has_output;
    entry->has_hidden = hello->has_hidden;
    entry->ctx_size = hello->ctx_size;
    entry->n_layers = hello->n_layers;
    entry->listen_port = hello->listen_port;
    entry->protocol_version = protocol_version;
    entry->caps = selected_caps;
    entry->link_generation = prepared_transport
        ? ds4_transport_generation(prepared_transport) : 0;

    ds4_dist_worker_entry *stale = NULL;
    pthread_mutex_lock(&state->mu);
    if (state->shutting_down) {
        pthread_mutex_unlock(&state->mu);
        ds4_transport_release(entry->transport);
        free(entry);
        return false;
    }
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *old = *link;
        if (strcmp(old->peer_host, peer_host) == 0 &&
            old->model_id == hello->model_id &&
            old->layer_start == hello->layer_start &&
            old->layer_end == hello->layer_end &&
            old->has_output == hello->has_output)
        {
            *link = old->next;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: dropped stale worker %s:%s layers=%u:%u%s\n",
                             old->peer_host,
                             old->peer_port,
                             old->layer_start,
                             old->layer_end,
                             old->has_output ? "+output" : "");
            shutdown(old->fd, SHUT_RDWR);
            old->next = stale;
            stale = old;
            continue;
        }
        link = &old->next;
    }
    entry->next = state->workers;
    state->workers = entry;
    state->generation++;
    pthread_mutex_unlock(&state->mu);

    while (stale) {
        ds4_dist_worker_entry *next = stale->next;
        ds4_transport_release(stale->transport);
        free(stale);
        stale = next;
    }

    char layer_end[32];
    if (entry->has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", entry->layer_end);
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: registered worker %s:%s data_port=%u model_id=%u quant=Q%u layers=%u:%s hidden=%u ctx=%u protocol=v%u transport=%s generation=%llu\n",
                     entry->peer_host,
                     entry->peer_port,
                     entry->listen_port,
                     entry->model_id,
                     entry->quant_bits,
                     entry->layer_start,
                     layer_end,
                     entry->has_hidden,
                     entry->ctx_size,
                     entry->protocol_version,
                     ds4_transport_name(entry->transport),
                     (unsigned long long)entry->link_generation);
    if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
    return true;
}

static int dist_worker_route_cmp(const void *a, const void *b) {
    const ds4_dist_worker_entry *ea = *(const ds4_dist_worker_entry * const *)a;
    const ds4_dist_worker_entry *eb = *(const ds4_dist_worker_entry * const *)b;
    if (ea->layer_start < eb->layer_start) return -1;
    if (ea->layer_start > eb->layer_start) return 1;
    if (ea->has_output != eb->has_output) return ea->has_output ? -1 : 1;
    if (ea->layer_end > eb->layer_end) return -1;
    if (ea->layer_end < eb->layer_end) return 1;
    return 0;
}

static bool dist_worker_route_candidate_ok(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_worker_entry *w,
        uint32_t last) {
    const bool needs_hidden = w->layer_end < last || !w->has_output;
    if (needs_hidden && !w->has_hidden) return false;
    if (w->layer_end >= last && !w->has_output && !state->local_can_output_head) return false;
    return true;
}

static bool dist_route_search_workers(
        const ds4_dist_coordinator_state *state,
        ds4_dist_worker_entry **workers,
        uint32_t n,
        uint32_t next,
        uint32_t last,
        ds4_dist_worker_entry **path,
        uint32_t *path_len,
        uint32_t *missing_layer) {
    bool saw_start = false;
    for (uint32_t i = 0; i < n; i++) {
        ds4_dist_worker_entry *w = workers[i];
        if (w->layer_start < next) continue;
        if (w->layer_start > next) break;
        saw_start = true;
        if (!dist_worker_route_candidate_ok(state, w, last)) continue;

        path[(*path_len)++] = w;
        if (w->layer_end >= last) return true;
        uint32_t child_missing = w->layer_end + 1u;
        if (dist_route_search_workers(state,
                                      workers,
                                      n,
                                      child_missing,
                                      last,
                                      path,
                                      path_len,
                                      &child_missing)) {
            return true;
        }
        if (child_missing > *missing_layer) *missing_layer = child_missing;
        (*path_len)--;
    }
    if (!saw_start && next > *missing_layer) *missing_layer = next;
    return false;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state) {
    if (!dist_coordinator_debug_enabled(state)) return;
    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        fprintf(stderr, "ds4: distributed coordinator: out of memory building route plan\n");
        return;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    bool complete = state->local_start == 0;
    bool has_output = state->local_end == last &&
                      (state->local_has_output || state->local_can_output_head);
    uint32_t next = state->local_end + 1u;
    if (state->local_end >= last) next = state->n_layers;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (complete && !has_output) {
        complete = dist_route_search_workers(state,
                                             workers,
                                             n,
                                             next,
                                             last,
                                             path,
                                             &path_len,
                                             &missing);
        if (complete && path_len != 0) {
            ds4_dist_worker_entry *final = path[path_len - 1u];
            has_output = final->has_output || state->local_can_output_head;
            next = state->n_layers;
        }
    }

    char plan[1024];
    size_t used = 0;
    char local_end[32];
    if (state->local_has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", state->local_end);
    used += (size_t)snprintf(plan + used, used < sizeof(plan) ? sizeof(plan) - used : 0,
                             "local %u:%s",
                             state->local_start,
                             local_end);
    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        if (used < sizeof(plan)) {
            char end[32];
            if (w->has_output) snprintf(end, sizeof(end), "output");
            else snprintf(end, sizeof(end), "%u", w->layer_end);
            used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                     " -> %s:%u Q%u %u:%s",
                                     w->peer_host,
                                     w->listen_port,
                                     w->quant_bits,
                                     w->layer_start,
                                     end);
        }
    }
    if (complete && path_len != 0 &&
        !path[path_len - 1u]->has_output && state->local_can_output_head &&
        used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    if (complete && path_len == 0 &&
        state->local_end == last && !state->local_has_output &&
        state->local_can_output_head && used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    complete = complete && has_output && next == state->n_layers;
    pthread_mutex_unlock(&state->mu);

    if (complete) {
        fprintf(stderr, "ds4: distributed coordinator: complete route ready: %s\n", plan);
    } else {
        fprintf(stderr, "ds4: distributed coordinator: route incomplete; next needed layer %u\n", missing);
    }
    free(path);
    free(workers);
}

static void dist_route_plan_free(ds4_dist_route_plan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < plan->count; i++) {
        ds4_transport_release(plan->entry[i].transport);
    }
    free(plan->entry);
    free(plan->blob);
    memset(plan, 0, sizeof(*plan));
}

static bool dist_route_entry_matches_worker(
        const ds4_dist_route_entry *route,
        const ds4_dist_worker_entry *worker) {
    const bool route_has_output = (route->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    return route->port == worker->listen_port &&
           strcmp(route->host, worker->peer_host) == 0 &&
           route->layer_start == worker->layer_start &&
           route->layer_end == worker->layer_end &&
           route_has_output == (worker->has_output != 0);
}

static void dist_coordinator_forget_route_workers(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan) {
    bool removed_any = false;
    pthread_mutex_lock(&state->mu);
    for (uint32_t i = 0; i < plan->count; i++) {
        ds4_dist_worker_entry **link = &state->workers;
        while (*link) {
            ds4_dist_worker_entry *entry = *link;
            if (!dist_route_entry_matches_worker(&plan->entry[i], entry)) {
                link = &entry->next;
                continue;
            }
            *link = entry->next;
            close(entry->fd);
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: forgot failed route worker %s:%u layers=%u:%u%s\n",
                             plan->entry[i].host,
                             plan->entry[i].port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            free(entry);
            removed_any = true;
            break;
        }
    }
    if (removed_any) state->generation++;
    pthread_mutex_unlock(&state->mu);

    if (removed_any && dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static bool dist_route_plan_append_blob(
        ds4_dist_route_plan *plan,
        const ds4_dist_route_entry *entry,
        char *err,
        size_t errlen) {
    const size_t host_len = strlen(entry->host);
    if (host_len == 0 || host_len >= NI_MAXHOST) {
        if (errlen) snprintf(err, errlen, "invalid route host");
        return false;
    }
    const uint64_t add = sizeof(ds4_dist_route_fixed) + host_len;
    if (add > UINT32_MAX || plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    uint32_t old_bytes = plan->blob_bytes;
    uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    uint8_t *p = (uint8_t *)plan->blob + old_bytes;
    ds4_dist_route_fixed fixed = {
        (uint32_t)host_len,
        entry->port,
        entry->layer_start,
        entry->layer_end,
        entry->flags,
    };
    dist_route_to_wire(&fixed);
    memcpy(p, &fixed, sizeof(fixed));
    memcpy(p + sizeof(fixed), entry->host, host_len);
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_route_plan_append_return_upstream(
        ds4_dist_route_plan *plan,
        char *err,
        size_t errlen) {
    const uint64_t add = sizeof(ds4_dist_route_return_fixed);
    if (plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    const uint32_t old_bytes = plan->blob_bytes;
    const uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    ds4_dist_route_return_fixed fixed = {
        DS4_DIST_ROUTE_RETURN_UPSTREAM,
        0,
        0,
    };
    dist_route_return_to_wire(&fixed);
    memcpy((uint8_t *)plan->blob + old_bytes, &fixed, sizeof(fixed));
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_coordinator_build_route_plan(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    memset(plan, 0, sizeof(*plan));
    if (generation) *generation = 0;

    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        if (errlen) snprintf(err, errlen, "out of memory building route");
        return false;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    if (state->local_start != 0) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "coordinator route does not start at layer 0");
        return false;
    }
    if (state->local_end == last &&
        (state->local_has_output || state->local_can_output_head)) {
        if (generation) *generation = state->generation;
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        return true;
    }

    uint32_t next = state->local_end + 1u;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (!dist_route_search_workers(state,
                                   workers,
                                   n,
                                   next,
                                   last,
                                   path,
                                   &path_len,
                                   &missing)) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "distributed route incomplete: missing layer %u", missing);
        return false;
    }

    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        ds4_dist_route_entry entry;
        memset(&entry, 0, sizeof(entry));
        entry.fd = -1;
        snprintf(entry.host, sizeof(entry.host), "%s", w->peer_host);
        entry.port = w->listen_port;
        entry.layer_start = w->layer_start;
        entry.layer_end = w->layer_end;
        entry.flags = w->has_output ? DS4_DIST_ROUTE_F_OUTPUT_LOGITS : 0u;
        entry.caps = w->caps;
        entry.transport = ds4_transport_retain(w->transport);
        if (state->use_control_for_work && plan->count == 0) {
            entry.fd = ds4_transport_control_fd(entry.transport);
        }

        ds4_dist_route_entry *new_entries = realloc(plan->entry, (size_t)(plan->count + 1u) * sizeof(plan->entry[0]));
        if (!new_entries) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            if (entry.fd >= 0) close(entry.fd);
            dist_route_plan_free(plan);
            if (errlen) snprintf(err, errlen, "out of memory building route entries");
            return false;
        }
        plan->entry = new_entries;
        plan->entry[plan->count++] = entry;
        if (!dist_route_plan_append_blob(plan, &entry, err, errlen)) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            dist_route_plan_free(plan);
            return false;
        }
    }
    if (generation) *generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    free(workers);
    free(path);
    if (plan->count != 0 && !dist_route_plan_append_return_upstream(plan, err, errlen)) {
        dist_route_plan_free(plan);
        return false;
    }
    return true;
}

static int dist_logits_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static bool dist_coordinator_ensure_route(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    return dist_coordinator_build_route_plan(state, plan, generation, err, errlen);
}

static uint64_t dist_coordinator_generation(ds4_dist_coordinator_state *state) {
    if (!state) return 0;
    pthread_mutex_lock(&state->mu);
    uint64_t generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    return generation;
}

/* =========================================================================
 * Coordinator Work Dispatch
 * ========================================================================= */

static int dist_recv_result_alloc_leased(
        int fd,
        ds4_transport *transport,
        const ds4_dist_coordinator_state *state,
        uint64_t request_id,
        const ds4_dist_v3_result_limits *limits,
        uint32_t *kind,
        uint64_t *result_hash,
        void **payload,
        uint32_t *payload_bytes,
        ds4_transport_lease **payload_lease,
        char *err,
        size_t errlen) {
    *payload = NULL;
    *payload_bytes = 0;
    *kind = 0;
    if (result_hash) *result_hash = 0;
    if (payload_lease) *payload_lease = NULL;
    const bool v3 = dist_transport_uses_v3(transport);

    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (v3) shutdown(fd, SHUT_RDWR);
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
        if (v3)
            shutdown(fd, SHUT_RDWR);
        else
            dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid frame");
        return 1;
    }

    ds4_dist_result_fixed result;
    rc = dist_read_full(fd, &result, sizeof(result));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed result");
        return 1;
    }
    dist_result_from_wire(&result);
    if (ds4_dist_v3_result_payload_validate(
            result.status, result.result_kind, result.payload_bytes,
            result.payload_bits, limits,
            state ? state->activation_bits : 0u, err, errlen) != 0) {
        shutdown(fd, SHUT_RDWR);
        return 1;
    }
    const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
    const uint64_t got_hash = dist_u64_from_halves(result.result_hash_hi,
                                                  result.result_hash_lo);
    uint32_t body_bytes = bytes - (uint32_t)sizeof(result);
    ds4_transport_bulk_desc bulk_desc;
    memset(&bulk_desc, 0, sizeof(bulk_desc));
    if (v3) {
        if (body_bytes < sizeof(ds4_transport_bulk_desc_wire)) {
            shutdown(fd, SHUT_RDWR);
            if (errlen) snprintf(err, errlen,
                                 "distributed v3 result is missing its bulk descriptor");
            return 1;
        }
        ds4_transport_bulk_desc_wire desc_wire;
        rc = dist_read_full(fd, &desc_wire, sizeof(desc_wire));
        if (rc <= 0 ||
            ds4_transport_bulk_desc_decode(&desc_wire, &bulk_desc) != 0) {
            shutdown(fd, SHUT_RDWR);
            if (errlen) snprintf(err, errlen,
                                 "failed to read distributed v3 result descriptor");
            return 1;
        }
        body_bytes -= (uint32_t)sizeof(desc_wire);
    }
    const bool payload_is_bulk = result.status == 0 &&
        result.payload_bytes != 0 &&
        (result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE ||
         result.result_kind == DS4_DIST_RESULT_LOGITS);
    uint32_t tcp_payload_bytes = result.payload_bytes;
    if (v3 && payload_is_bulk &&
        bulk_desc.mode == DS4_TRANSPORT_BULK_NHI_OOB)
        tcp_payload_bytes = 0;
    if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
        result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
        !state || result.telemetry_count > state->n_layers + 1u ||
        result.telemetry_bytes > body_bytes ||
        tcp_payload_bytes != body_bytes - result.telemetry_bytes) {
        if (v3)
            shutdown(fd, SHUT_RDWR);
        else
            dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result telemetry metadata mismatch");
        return 1;
    }
    if (got_request != request_id) {
        if (v3)
            shutdown(fd, SHUT_RDWR);
        else
            dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result metadata mismatch");
        return 1;
    }

    if (v3) {
        int desc_rc = 0;
        if (!payload_is_bulk) {
            desc_rc = dist_validate_bulk_none_desc(transport, &bulk_desc,
                                                   0, request_id);
        } else {
            const uint32_t bulk_kind =
                result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE
                    ? DS4_TRANSPORT_BULK_RESULT_HIDDEN
                    : DS4_TRANSPORT_BULK_RESULT_LOGITS;
            if (bulk_desc.mode == DS4_TRANSPORT_BULK_TCP_INLINE) {
                desc_rc = ds4_transport_validate_inline_desc(
                    &bulk_desc, bulk_kind, 0, request_id,
                    result.payload_bytes, result.payload_bits,
                    err, errlen);
            } else if (bulk_desc.mode == DS4_TRANSPORT_BULK_NHI_OOB) {
                desc_rc = ds4_transport_validate_oob_desc(
                    transport, &bulk_desc, bulk_kind, 0, request_id,
                    result.payload_bytes, result.payload_bits,
                    err, errlen);
            } else {
                desc_rc = -1;
                errno = EPROTO;
            }
        }
        if (desc_rc != 0) {
            shutdown(fd, SHUT_RDWR);
            if (errlen && !err[0]) snprintf(err, errlen,
                                            "invalid distributed v3 result descriptor");
            return 1;
        }
    }

    if (result.telemetry_bytes != 0) {
        if (dist_coordinator_debug_enabled(state)) {
            ds4_dist_telemetry_fixed *telemetry = malloc(result.telemetry_bytes);
            if (!telemetry) {
                if (v3)
                    shutdown(fd, SHUT_RDWR);
                else
                    dist_discard_bytes(fd, body_bytes);
                if (errlen) snprintf(err, errlen, "out of memory reading distributed telemetry");
                return 1;
            }
            rc = dist_read_full(fd, telemetry, result.telemetry_bytes);
            if (rc <= 0) {
                free(telemetry);
                if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
                return 1;
            }
            for (uint32_t i = 0; i < result.telemetry_count; i++) {
                dist_telemetry_from_wire(&telemetry[i]);
                DIST_COORD_DEBUG(state,
                                 "ds4: distributed telemetry: request=%llu hop=%u layers=%u:%u route=%u pos=%u tokens=%u eval=%.3fms downstream_wait=%.3fms forward_send=%.3fms input=%.2fMiB output=%.2fMiB\n",
                                 (unsigned long long)got_request,
                                 i,
                                 telemetry[i].layer_start,
                                 telemetry[i].layer_end,
                                 telemetry[i].route_index,
                                 telemetry[i].pos0,
                                 telemetry[i].n_tokens,
                                 (double)telemetry[i].eval_usec / 1000.0,
                                 (double)telemetry[i].downstream_wait_usec / 1000.0,
                                 (double)telemetry[i].forward_send_usec / 1000.0,
                                 (double)telemetry[i].input_bytes / (1024.0 * 1024.0),
                                 (double)telemetry[i].output_bytes / (1024.0 * 1024.0));
            }
            free(telemetry);
        } else if (dist_discard_bytes(fd, result.telemetry_bytes) <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
            return 1;
        }
    }

    void *buf = NULL;
    ds4_transport_lease *lease = NULL;
    if (result.payload_bytes != 0) {
        const bool try_mapped = payload_lease && v3 && payload_is_bulk &&
            result.payload_bits == 32u &&
            bulk_desc.mode == DS4_TRANSPORT_BULK_NHI_OOB &&
            ds4_transport_mapped_leases_supported(transport);
        if (try_mapped) {
            if (ds4_transport_rx_lease_acquire(transport, &bulk_desc, &lease,
                                               err, errlen) == 0) {
                buf = ds4_transport_lease_host_ptr(lease);
            } else if (errno != ENOTSUP) {
                shutdown(fd, SHUT_RDWR);
                if (errlen && !err[0])
                    snprintf(err, errlen,
                             "failed to acquire mapped distributed result");
                return 1;
            }
        }
        if (!buf) {
            buf = malloc(result.payload_bytes);
            if (!buf) {
                if (v3)
                    shutdown(fd, SHUT_RDWR);
                else
                    dist_discard_bytes(fd, result.payload_bytes);
                if (errlen)
                    snprintf(err, errlen,
                             "out of memory reading distributed result");
                return 1;
            }
            if (v3 && payload_is_bulk) {
                rc = ds4_transport_recv_bulk_desc(transport, &bulk_desc,
                                                  buf, result.payload_bytes);
            } else {
                rc = payload_is_bulk
                    ? ds4_transport_recv_bulk(transport, buf,
                                              result.payload_bytes)
                    : dist_read_full(fd, buf, result.payload_bytes);
            }
            if (rc <= 0) {
                free(buf);
                if (errlen)
                    snprintf(err, errlen,
                             "failed to read distributed result payload");
                return 1;
            }
        }
    }

    if (result.status != 0) {
        if (errlen) {
            if (buf && result.payload_bytes) {
                size_t n = result.payload_bytes < errlen - 1 ? result.payload_bytes : errlen - 1;
                memcpy(err, buf, n);
                err[n] = '\0';
            } else {
                snprintf(err, errlen, "distributed worker returned an error");
            }
        }
        if (lease)
            ds4_transport_lease_release(lease);
        else
            free(buf);
        return DS4_DIST_RECV_REMOTE_ERROR;
    }

    if (result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE && result.payload_bytes != 0) {
        float *decoded = NULL;
        uint32_t decoded_bytes = 0;
        bool uses_wire = false;
        if (dist_decode_activation_payload(buf,
                                           result.payload_bits,
                                           result.payload_bytes,
                                           &decoded,
                                           &decoded_bytes,
                                           &uses_wire,
                                           err,
                                           errlen) != 0) {
            if (lease)
                ds4_transport_lease_release(lease);
            else
                free(buf);
            return 1;
        }
        if (!uses_wire) {
            if (lease) {
                if (dist_mapped_lease_commit_after_gpu_quiesce(lease) != 0) {
                    ds4_transport_lease_release(lease);
                    free(decoded);
                    return 1;
                }
                ds4_transport_lease_release(lease);
                lease = NULL;
            } else {
                free(buf);
            }
            buf = decoded;
        }
        result.payload_bytes = decoded_bytes;
    }

    *kind = result.result_kind;
    if (result_hash) *result_hash = got_hash;
    *payload = buf;
    *payload_bytes = result.payload_bytes;
    if (payload_lease) *payload_lease = lease;
    return 0;
}

static int dist_recv_result_alloc(
        int fd,
        ds4_transport *transport,
        const ds4_dist_coordinator_state *state,
        uint64_t request_id,
        const ds4_dist_v3_result_limits *limits,
        uint32_t *kind,
        uint64_t *result_hash,
        void **payload,
        uint32_t *payload_bytes,
        char *err,
        size_t errlen) {
    return dist_recv_result_alloc_leased(fd, transport, state, request_id,
                                         limits, kind, result_hash, payload,
                                         payload_bytes, NULL, err, errlen);
}

static int dist_result_payload_release(
        void *payload,
        ds4_transport_lease *lease,
        char *err,
        size_t errlen) {
    if (!lease) {
        free(payload);
        return 0;
    }
    const int rc = dist_mapped_lease_commit_after_gpu_quiesce(lease);
    ds4_transport_lease_release(lease);
    if (rc != 0) {
        if (errlen)
            snprintf(err, errlen,
                     "failed to release mapped distributed result: %s",
                     strerror(errno));
        return 1;
    }
    return 0;
}

static int dist_coordinator_send_remote_work_on_fd(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        int fd,
        ds4_transport *transport,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t result_hash,
        bool reset_session,
        bool ack_only,
        uint32_t spec_flags,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        const ds4_dist_tx_bulk_plan *tx_plan,
        char *err,
        size_t errlen) {
    if (plan->count == 0) {
        if (errlen) snprintf(err, errlen, "distributed route has no remote worker");
        return 1;
    }
    const ds4_dist_route_entry *first = &plan->entry[0];

    ds4_dist_work_fixed work;
    memset(&work, 0, sizeof(work));
    work.model_id = state->model_id;
    dist_u64_to_halves(session_id, &work.session_hi, &work.session_lo);
    dist_u64_to_halves(request_id, &work.request_hi, &work.request_lo);
    dist_u64_to_halves(prefix_hash, &work.prefix_hash_hi, &work.prefix_hash_lo);
    dist_u64_to_halves(result_hash, &work.result_hash_hi, &work.result_hash_lo);
    work.pos0 = pos0;
    work.n_tokens = n_tokens;
    work.layer_start = first->layer_start;
    work.layer_end = first->layer_end;
    work.flags = DS4_DIST_WORK_F_INPUT_HC | spec_flags;
    if (reset_session) work.flags |= DS4_DIST_WORK_F_RESET_SESSION;
    if (ack_only) work.flags |= DS4_DIST_WORK_F_ACK_ONLY;
    if ((first->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        work.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }
    uint32_t wire_hidden_hc_bytes = 0;
    if (!dist_activation_wire_bytes_from_f32_bytes(state->activation_bits,
                                                   hidden_hc_bytes,
                                                   &wire_hidden_hc_bytes)) {
        if (errlen) snprintf(err, errlen, "invalid distributed hidden-state size");
        return 1;
    }
    work.token_bytes = n_tokens * sizeof(uint32_t);
    work.input_hc_bytes = wire_hidden_hc_bytes;
    work.input_hc_bits = state->activation_bits;
    work.route_count = plan->count;
    work.route_index = 0;
    work.route_bytes = plan->blob_bytes;

    if (dist_send_work_frame_prepared(
            fd, transport, &work, tokens, hidden_hc, plan->blob,
            dist_tx_bulk_plan_desc(tx_plan), tx_plan ? tx_plan->lease : NULL) != 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed work");
        return 1;
    }
    return 0;
}

static int dist_result_limits_for_work(
        ds4_engine *engine,
        uint32_t activation_bits,
        uint32_t hidden_f32_bytes,
        uint32_t n_tokens,
        uint32_t work_flags,
        bool final_output_logits,
        ds4_dist_v3_result_limits *limits,
        char *err,
        size_t errlen) {
    if (!engine || !limits || n_tokens == 0u) {
        if (errlen) snprintf(err, errlen,
                             "invalid distributed result-limit request");
        return 1;
    }
    memset(limits, 0, sizeof(*limits));

    uint32_t expected_kind;
    const bool output_drafts =
        (work_flags & DS4_DIST_WORK_F_OUTPUT_DRAFTS) != 0;
    const bool output_all_logits =
        (work_flags & DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS) != 0;
    if ((work_flags & (DS4_DIST_WORK_F_ACK_ONLY |
                       DS4_DIST_WORK_F_SPEC_COMMIT)) != 0) {
        expected_kind = DS4_DIST_RESULT_ACK;
    } else if (output_drafts && output_all_logits) {
        expected_kind = DS4_DIST_RESULT_LOGITS_NROWS;
    } else if (output_drafts) {
        expected_kind = DS4_DIST_RESULT_LOGITS_DRAFTS;
    } else if (output_all_logits && n_tokens == 2u &&
               (work_flags & DS4_DIST_WORK_F_SPEC_ROLLBACK) == 0) {
        expected_kind = DS4_DIST_RESULT_LOGITS_DECODE2;
    } else if (output_all_logits) {
        expected_kind = DS4_DIST_RESULT_LOGITS_NROWS;
    } else {
        expected_kind = final_output_logits
            ? DS4_DIST_RESULT_LOGITS
            : DS4_DIST_RESULT_HIDDEN_STATE;
    }
    limits->allowed_kinds = DS4_DIST_RESULT_KIND_BIT(expected_kind);

    if (expected_kind == DS4_DIST_RESULT_ACK) return 0;
    if (expected_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        if (!dist_activation_wire_bytes_from_f32_bytes(
                activation_bits, hidden_f32_bytes,
                &limits->hidden_wire_bytes)) {
            if (errlen) snprintf(err, errlen,
                                 "distributed hidden result exceeds protocol limits");
            return 1;
        }
        return 0;
    }

    const uint64_t logits_bytes64 =
        (uint64_t)ds4_engine_vocab_size(engine) * sizeof(float);
    if (logits_bytes64 == 0 || logits_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen,
                             "distributed logits result exceeds protocol limits");
        return 1;
    }
    limits->logits_bytes = (uint32_t)logits_bytes64;
    if (expected_kind == DS4_DIST_RESULT_LOGITS_NROWS) {
        const uint64_t nrows_bytes64 = (uint64_t)n_tokens * logits_bytes64;
        if (nrows_bytes64 == 0 || nrows_bytes64 > UINT32_MAX) {
            if (errlen) snprintf(err, errlen,
                                 "distributed logits rows exceed protocol limits");
            return 1;
        }
        limits->nrows_bytes = (uint32_t)nrows_bytes64;
    }
    return 0;
}

static int dist_coordinator_eval_remote_on_fd(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        int fd,
        ds4_transport *transport,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t expected_result_hash,
        bool reset_session,
        uint32_t spec_flags,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        float *logits,
        int *drafts_out,
        int *n_drafts_out,
        int *top0_out,
        char *err,
        size_t errlen) {
    const bool profile = dist_decode_profile_enabled() && n_tokens == 1;
    const double total_t0 = profile ? dist_now_sec() : 0.0;
    const double send_t0 = profile ? dist_now_sec() : 0.0;
    const bool final_output_logits = plan && plan->count != 0u &&
        (plan->entry[plan->count - 1u].flags &
         DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    ds4_dist_v3_result_limits result_limits;
    if (dist_result_limits_for_work(
            state->engine, state->activation_bits, hidden_hc_bytes,
            n_tokens, spec_flags, final_output_logits,
            &result_limits, err, errlen) != 0)
        return 1;
    const uint32_t logits_bytes = result_limits.logits_bytes;
    if (n_drafts_out) *n_drafts_out = 0;
    if (top0_out) *top0_out = -1;
    int rc = dist_coordinator_send_remote_work_on_fd(state,
                                                     plan,
                                                     fd,
                                                     transport,
                                                     tokens,
                                                     n_tokens,
                                                     pos0,
                                                     session_id,
                                                     request_id,
                                                     prefix_hash,
                                                     expected_result_hash,
                                                     reset_session,
                                                     false,
                                                     spec_flags,
                                                     hidden_hc,
                                                     hidden_hc_bytes,
                                                     NULL,
                                                     err,
                                                     errlen);
    const double send_t1 = profile ? dist_now_sec() : 0.0;
    uint32_t kind = 0, payload_bytes = 0;
    uint64_t result_hash = 0;
    void *payload = NULL;
    ds4_transport_lease *payload_lease = NULL;
    if (rc == 0) {
        const double recv_t0 = profile ? dist_now_sec() : 0.0;
        rc = dist_recv_result_alloc_leased(fd,
                                           transport,
                                           state,
                                           request_id,
                                           &result_limits,
                                           &kind,
                                           &result_hash,
                                           &payload,
                                           &payload_bytes,
                                           &payload_lease,
                                           err,
                                           errlen);
        const double recv_t1 = profile ? dist_now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu pos=%u send=%.3fms wait_result=%.3fms kind=%u payload=%.2fMiB\n",
                    (unsigned long long)request_id,
                    pos0,
                    (send_t1 - send_t0) * 1000.0,
                    (recv_t1 - recv_t0) * 1000.0,
                    kind,
                    (double)payload_bytes / (1024.0 * 1024.0));
        }
    }
    if (rc != 0) return rc;
    if (result_hash != expected_result_hash) {
        (void)dist_result_payload_release(payload, payload_lease, NULL, 0);
        if (errlen) snprintf(err, errlen, "distributed result prefix hash mismatch");
        return 1;
    }

    if (kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes) {
        const double copy_t0 = profile ? dist_now_sec() : 0.0;
        memcpy(logits, payload, logits_bytes);
        const int release_rc = dist_result_payload_release(
            payload, payload_lease, err, errlen);
        if (profile) {
            const double copy_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu copy_logits=%.3fms total=%.3fms\n",
                    (unsigned long long)request_id,
                    (copy_t1 - copy_t0) * 1000.0,
                    (copy_t1 - total_t0) * 1000.0);
        }
        return release_rc;
    }
    if (kind == DS4_DIST_RESULT_LOGITS_DRAFTS && payload_bytes >= logits_bytes + 4u) {
        uint8_t *p = (uint8_t *)payload;
        memcpy(logits, p, logits_bytes);
        p += logits_bytes;
        uint32_t nd = 0;
        memcpy(&nd, p, 4u);
        p += 4u;
        if (nd > 16u || payload_bytes != logits_bytes + 4u + 4u * nd) {
            free(payload);
            if (errlen) snprintf(err, errlen, "distributed route returned invalid draft payload");
            return 1;
        }
        int n_copy = 0;
        if (drafts_out && n_drafts_out && nd != 0) {
            n_copy = (int)nd;
            for (uint32_t i = 0; i < nd; i++) {
                int32_t dv = 0;
                memcpy(&dv, p, 4u);
                p += 4u;
                drafts_out[i] = (int)dv;
            }
        }
        if (n_drafts_out) *n_drafts_out = n_copy;
        free(payload);
        if (profile) {
            const double copy_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu drafts=%d total=%.3fms\n",
                    (unsigned long long)request_id,
                    n_copy,
                    (copy_t1 - total_t0) * 1000.0);
        }
        return 0;
    }
    if (kind == DS4_DIST_RESULT_LOGITS_DECODE2 &&
        payload_bytes == 4u + logits_bytes) {
        if (top0_out) {
            int32_t tv = 0;
            memcpy(&tv, payload, 4u);
            *top0_out = (int)tv;
        }
        memcpy(logits, (uint8_t *)payload + 4u, logits_bytes);
        free(payload);
        if (profile) {
            const double copy_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu decode2 top=%d total=%.3fms\n",
                    (unsigned long long)request_id,
                    top0_out ? *top0_out : -1,
                    (copy_t1 - total_t0) * 1000.0);
        }
        return 0;
    }
    if (kind == DS4_DIST_RESULT_LOGITS_NROWS &&
        payload_bytes >= (uint64_t)n_tokens * logits_bytes) {
        const uint64_t rows_bytes = (uint64_t)n_tokens * logits_bytes;
        memcpy(logits, payload, (size_t)rows_bytes);
        if (payload_bytes > rows_bytes) {
            /* Fused verify spans append continuation drafts after the rows. */
            uint8_t *p = (uint8_t *)payload + rows_bytes;
            uint32_t nd = 0;
            if (payload_bytes < rows_bytes + 4u) {
                free(payload);
                if (errlen) snprintf(err, errlen, "distributed route returned invalid draft payload");
                return 1;
            }
            memcpy(&nd, p, 4u);
            p += 4u;
            if (nd > 16u || payload_bytes != rows_bytes + 4u + 4u * nd) {
                free(payload);
                if (errlen) snprintf(err, errlen, "distributed route returned invalid draft payload");
                return 1;
            }
            if (drafts_out && n_drafts_out) {
                for (uint32_t i = 0; i < nd; i++) {
                    int32_t dv = 0;
                    memcpy(&dv, p, 4u);
                    p += 4u;
                    drafts_out[i] = (int)dv;
                }
                *n_drafts_out = (int)nd;
            }
        }
        free(payload);
        if (profile) {
            const double copy_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu rows=%u total=%.3fms\n",
                    (unsigned long long)request_id,
                    n_tokens,
                    (copy_t1 - total_t0) * 1000.0);
        }
        return 0;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE && payload_bytes == hidden_hc_bytes) {
        const double head_t0 = profile ? dist_now_sec() : 0.0;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           payload,
                                                           n_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        (void)dist_result_payload_release(payload, payload_lease, NULL, 0);
        if (profile) {
            const double head_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu output_head=%.3fms total=%.3fms rc=%d\n",
                    (unsigned long long)request_id,
                    (head_t1 - head_t0) * 1000.0,
                    (head_t1 - total_t0) * 1000.0,
                    head_rc);
        }
        return head_rc;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        (void)dist_result_payload_release(payload, payload_lease, NULL, 0);
        if (errlen) snprintf(err, errlen, "distributed route returned invalid hidden-state size");
        return 1;
    }
    (void)dist_result_payload_release(payload, payload_lease, NULL, 0);
    if (errlen) snprintf(err, errlen, "distributed route did not return logits or hidden-state");
    return 1;
}

static int dist_coordinator_eval_span(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        bool reset_session,
        uint32_t spec_flags,
        bool decode2_span,
        int *top0_out,
        float *logits,
        int *drafts_out,
        int *n_drafts_out,
        char *err,
        size_t errlen) {
    const bool profile = dist_decode_profile_enabled() && n_tokens == 1;
    const double span_t0 = profile ? dist_now_sec() : 0.0;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t hidden_bytes64 = (uint64_t)n_tokens * hc_values * sizeof(float);
    if (hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    uint64_t prefix_hash = DS4_DIST_TOKEN_HASH_INIT;
    if (reset_session) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "distributed reset span must start at position 0");
            return 1;
        }
    } else if (dist_session_token_hash_prefix(session,
                                              pos0,
                                              &prefix_hash,
                                              err,
                                              errlen) != 0) {
        return 1;
    }
    const uint64_t result_hash = dist_token_hash_update_span(prefix_hash, tokens, n_tokens);
    const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
    float *hidden = NULL;
    if (plan->count != 0) {
        hidden = malloc(hidden_bytes);
        if (!hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating coordinator hidden-state");
            return 1;
        }
    }
    if (reset_session &&
        ds4_session_layer_slice_reset(session, err, errlen) != 0) {
        free(hidden);
        return 1;
    }

    const bool local_logits = plan->count == 0;
    int remote_fd = -1;
    if (plan->count != 0) {
        const ds4_dist_route_entry *first = &plan->entry[0];
        remote_fd = first->fd;
        if (remote_fd < 0) {
            if (errlen) snprintf(err, errlen, "distributed route has no live first-hop connection");
            free(hidden);
            return 1;
        }
    }

    const double local_t0 = profile ? dist_now_sec() : 0.0;
    int rc;
    const bool exact_required =
        (spec_flags & DS4_DIST_WORK_F_SPEC_EXACT_REQUIRED) != 0;
    const bool exact_rows =
        !decode2_span &&
        (spec_flags & DS4_DIST_WORK_F_SPEC_VERIFY) != 0 &&
        exact_required && n_tokens > 1u &&
        ds4_session_dist_spec_exact_verify(session, n_tokens);
    if (exact_required && !exact_rows) {
        free(hidden);
        if (errlen) snprintf(err, errlen,
                             "coordinator exact speculative verify gate is unavailable");
        return 1;
    }
    if (decode2_span) {
        const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
        rc = ds4_session_eval_layer_slice_decode2(session,
                                                  tokens[0],
                                                  tokens[1],
                                                  pos0,
                                                  state->local_start,
                                                  state->local_end,
                                                  NULL,
                                                  NULL,
                                                  hidden,
                                                  hidden + hc_values,
                                                  false,
                                                  NULL,
                                                  NULL,
                                                  err,
                                                  errlen);
    } else if (exact_rows) {
        rc = ds4_session_eval_layer_slice_exact_rows(session,
                                                     tokens,
                                                     n_tokens,
                                                     pos0,
                                                     state->local_start,
                                                     state->local_end,
                                                     NULL,
                                                     hidden,
                                                     NULL,
                                                     false,
                                                     err,
                                                     errlen);
    } else {
        rc = ds4_session_eval_layer_slice(session,
                                          tokens,
                                          n_tokens,
                                          pos0,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          local_logits ? NULL : hidden,
                                          local_logits,
                                          local_logits ? logits : NULL,
                                          err,
                                          errlen);
    }
    const double local_t1 = profile ? dist_now_sec() : 0.0;
    double remote_t0 = 0.0, remote_t1 = 0.0;
    if (rc == 0 && plan->count != 0) {
        remote_t0 = profile ? dist_now_sec() : 0.0;
        rc = dist_coordinator_eval_remote_on_fd(state,
                                                session,
                                                plan,
                                                remote_fd,
                                                plan->entry[0].transport,
                                                tokens,
                                                n_tokens,
                                                pos0,
                                                session_id,
                                                request_id,
                                                prefix_hash,
                                                result_hash,
                                                reset_session,
                                                spec_flags,
                                                hidden,
                                                hidden_bytes,
                                                logits,
                                                drafts_out,
                                                n_drafts_out,
                                                top0_out,
                                                err,
                                                errlen);
        remote_t1 = profile ? dist_now_sec() : 0.0;
    }
    if (profile) {
        const double span_t1 = dist_now_sec();
        fprintf(stderr,
                "ds4: dist decode profile: span request=%llu pos=%u layers=%u:%u local=%.3fms remote=%.3fms total=%.3fms hidden=%.2fMiB rc=%d\n",
                (unsigned long long)request_id,
                pos0,
                state->local_start,
                state->local_end,
                (local_t1 - local_t0) * 1000.0,
                (remote_t1 - remote_t0) * 1000.0,
                (span_t1 - span_t0) * 1000.0,
                (double)hidden_bytes / (1024.0 * 1024.0),
                rc);
    }
    free(hidden);
    return rc;
}

/* =========================================================================
 * One-Shot Coordinator Generation Utilities
 * ========================================================================= */

static bool dist_prompt_is_rendered_chat(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

static bool dist_json_utf8_valid(const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        int need = 0;
        if ((c & 0xe0) == 0xc0) need = 2;
        else if ((c & 0xf0) == 0xe0) need = 3;
        else if ((c & 0xf8) == 0xf0) need = 4;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i + 1u];
        if ((c1 & 0xc0) != 0x80) return false;
        if (need == 2 && c < 0xc2) return false;
        if (need == 3 && c == 0xe0 && c1 < 0xa0) return false;
        if (need == 3 && c == 0xed && c1 >= 0xa0) return false;
        if (need == 4 && c == 0xf0 && c1 < 0x90) return false;
        if (need == 4 && c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 2; j < need; j++) {
            if ((((unsigned char)s[i + (size_t)j]) & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void dist_json_write_string(FILE *fp, const char *s, size_t n) {
    const bool valid_utf8 = dist_json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void dist_json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    dist_json_write_string(fp, text ? text : "", text ? n : 0);
    fputs(",\"bytes\":[", fp);
    if (text) {
        for (size_t i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
        }
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int dist_write_logits_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        const ds4_dist_route_plan *plan,
        const float *logits) {
    FILE *fp = fopen(gen->dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(state->engine);
    const int argmax = dist_logits_argmax(logits, vocab);
    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"vocab\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"argmax_token\":",
            ds4_engine_routed_quant_bits(state->engine),
            prompt->len,
            gen->ctx_size,
            vocab,
            plan->count);
    dist_json_write_token(fp, state->engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }
    return 0;
}

static int dist_logits_top_logprobs(const float *logits, int vocab, ds4_dist_logprob *scores, int k) {
    if (k <= 0) return 0;
    for (int i = 0; i < k; i++) {
        scores[i].id = -1;
        scores[i].logit = -FLT_MAX;
        scores[i].logprob = -FLT_MAX;
    }

    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i]) && logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i])) sum += exp((double)logits[i] - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);

    int n = 0;
    for (int i = 0; i < vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == k && v <= scores[k - 1].logit) continue;
        int pos = n < k ? n++ : k - 1;
        while (pos > 0 && v > scores[pos - 1].logit) {
            scores[pos] = scores[pos - 1];
            pos--;
        }
        scores[pos].id = i;
        scores[pos].logit = v;
        scores[pos].logprob = (float)((double)v - logsum);
    }
    return n;
}

static int dist_coordinator_rebuild_from_transcript(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        ds4_dist_route_plan *plan,
        const ds4_tokens *transcript,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        uint64_t *plan_generation,
        bool forget_route,
        char *err,
        size_t errlen) {
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: replaying %d tokens after distributed %s\n",
                     transcript->len,
                     forget_route ? "route failure" : "KV mismatch");
    if (forget_route) {
        dist_coordinator_forget_route_workers(state, plan);
        dist_route_plan_free(plan);
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    } else if (plan->count == 0) {
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    }
    if (dist_coordinator_prefill_prompt(state,
                                        session,
                                        plan,
                                        transcript,
                                        session_id,
                                        request_id,
                                        logits,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    return 0;
}

static int dist_write_logprobs_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        ds4_dist_route_plan *plan,
        ds4_session *session,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits) {
    FILE *fp = fopen(gen->dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        return 1;
    }

    int k = gen->dump_logprobs_top_k > 0 ? gen->dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_dist_logprob *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        return 1;
    }

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, prompt);

    int max_tokens = gen->n_predict;
    int room = gen->ctx_size - prompt->len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"top_k\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"steps\":[\n",
            prompt->len,
            gen->ctx_size,
            k,
            plan->count);

    char err[256];
    int rc = 0;
    const int eos = ds4_token_eos(state->engine);
    for (int generated = 0; generated < max_tokens; generated++) {
        const int n = dist_logits_top_logprobs(logits, ds4_engine_vocab_size(state->engine), scores, k);
        const int token = dist_logits_argmax(logits, ds4_engine_vocab_size(state->engine));
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        dist_json_write_token(fp, state->engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            dist_json_write_token(fp, state->engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == eos) break;
        const uint32_t token_pos = (uint32_t)prompt->len + (uint32_t)generated;
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_eval_span(state, session, plan,
                                       &token, 1, token_pos,
                                       session_id, (*request_id)++,
                                       false, 0, false, NULL, logits, NULL,
                                       NULL, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed decode failed while dumping logprobs: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         plan,
                                                         &transcript,
                                                         session_id,
                                                         request_id,
                                                         logits,
                                                         NULL,
                                                         true,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed recovery failed while dumping logprobs: %s\n",
                        err);
                rc = 1;
                break;
            }
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        rc = 1;
    }
    ds4_tokens_free(&transcript);
    free(scores);
    return rc;
}

/* =========================================================================
 * Pipelined Prefill
 * =========================================================================
 *
 * Long prompt ingestion is chunked so the coordinator can compute its local
 * slice for chunk N+1 while downstream workers process chunk N. Intermediate
 * chunks are ACK-only; only the final chunk needs to return hidden state or
 * logits to the coordinator.
 */

static int dist_prefill_sender_init(
        ds4_dist_prefill_sender *sender,
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        int fd,
        ds4_transport *transport,
        uint32_t chunk_count,
        uint32_t max_hidden_bytes,
        char *err,
        size_t errlen) {
    memset(sender, 0, sizeof(*sender));
    sender->state = state;
    sender->plan = plan;
    sender->prompt = prompt;
    sender->session_id = session_id;
    sender->fd = fd;
    sender->transport = transport;
    sender->slot_count = dist_prefill_send_depth(chunk_count);
    pthread_mutex_init(&sender->mu, NULL);
    pthread_cond_init(&sender->can_enqueue, NULL);
    pthread_cond_init(&sender->can_dequeue, NULL);

    sender->slots = calloc(sender->slot_count, sizeof(sender->slots[0]));
    if (!sender->slots) {
        if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender slots");
        return 1;
    }
    for (uint32_t i = 0; i < sender->slot_count; i++) {
        sender->slots[i].hidden = malloc(max_hidden_bytes);
        if (!sender->slots[i].hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender hidden-state buffers");
            return 1;
        }
    }
    return 0;
}

static void dist_prefill_sender_destroy(ds4_dist_prefill_sender *sender) {
    if (!sender) return;
    if (sender->slots) {
        for (uint32_t i = 0; i < sender->slot_count; i++) {
            free(sender->slots[i].hidden);
        }
        free(sender->slots);
    }
    pthread_cond_destroy(&sender->can_dequeue);
    pthread_cond_destroy(&sender->can_enqueue);
    pthread_mutex_destroy(&sender->mu);
}

static ds4_dist_prefill_send_slot *dist_prefill_sender_acquire_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    while (!sender->stop && sender->queued == sender->slot_count) {
        pthread_cond_wait(&sender->can_enqueue, &sender->mu);
    }
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return NULL;
    }
    ds4_dist_prefill_send_slot *slot = &sender->slots[sender->tail];
    pthread_mutex_unlock(&sender->mu);
    return slot;
}

static int dist_prefill_sender_enqueue_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return 1;
    }
    sender->tail = (sender->tail + 1u) % sender->slot_count;
    sender->queued++;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    return 0;
}

static void dist_prefill_sender_finish(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
}

static void dist_prefill_sender_cancel(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    sender->stop = true;
    pthread_cond_broadcast(&sender->can_enqueue);
    pthread_cond_broadcast(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    shutdown(sender->fd, SHUT_RDWR);
}

static void *dist_prefill_sender_main(void *arg) {
    ds4_dist_prefill_sender *sender = arg;
    for (;;) {
        pthread_mutex_lock(&sender->mu);
        while (!sender->stop && sender->queued == 0 && !sender->producer_done) {
            pthread_cond_wait(&sender->can_dequeue, &sender->mu);
        }
        if (sender->stop || (sender->queued == 0 && sender->producer_done)) {
            pthread_mutex_unlock(&sender->mu);
            break;
        }
        ds4_dist_prefill_send_slot *slot = &sender->slots[sender->head];
        pthread_mutex_unlock(&sender->mu);

        char send_err[256];
        const double send_t0 = dist_now_sec();
        int rc = dist_coordinator_send_remote_work_on_fd(sender->state,
                                                         sender->plan,
                                                         sender->fd,
                                                         sender->transport,
                                                         sender->prompt->v + slot->pos,
                                                         slot->n_tokens,
                                                         slot->pos,
                                                         sender->session_id,
                                                         slot->request_id,
                                                         slot->prefix_hash,
                                                         slot->result_hash,
                                                         slot->reset_session,
                                                         slot->ack_only,
                                                         0,
                                                         slot->hidden,
                                                         slot->hidden_bytes,
                                                         NULL,
                                                         send_err,
                                                         sizeof(send_err));
        const double send_t1 = dist_now_sec();

        pthread_mutex_lock(&sender->mu);
        uint32_t slot_hidden_wire_bytes = slot->hidden_bytes;
        (void)dist_activation_wire_bytes_from_f32_bytes(sender->state->activation_bits,
                                                        slot->hidden_bytes,
                                                        &slot_hidden_wire_bytes);
        sender->send_sec += send_t1 - send_t0;
        sender->send_bytes += (uint64_t)sizeof(ds4_dist_work_fixed) +
                              (uint64_t)slot->n_tokens * sizeof(uint32_t) +
                              (uint64_t)slot_hidden_wire_bytes +
                              sender->plan->blob_bytes;
        if (rc != 0) {
            sender->rc = 1;
            snprintf(sender->err, sizeof(sender->err), "%s", send_err);
            sender->stop = true;
            pthread_cond_broadcast(&sender->can_enqueue);
            pthread_cond_broadcast(&sender->can_dequeue);
            pthread_mutex_unlock(&sender->mu);
            shutdown(sender->fd, SHUT_RDWR);
            break;
        }
        sender->head = (sender->head + 1u) % sender->slot_count;
        sender->queued--;
        pthread_cond_signal(&sender->can_enqueue);
        pthread_mutex_unlock(&sender->mu);
    }
    return NULL;
}

static void dist_prefill_reader_signal_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t completed,
        bool done) {
    pthread_mutex_lock(&reader->progress_mu);
    if (completed > reader->progress_completed) reader->progress_completed = completed;
    if (done) reader->progress_done = true;
    pthread_cond_broadcast(&reader->progress_cv);
    pthread_mutex_unlock(&reader->progress_mu);
}

static void dist_prefill_reader_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported || !reader->progress_session) return;

    pthread_mutex_lock(&reader->progress_mu);
    uint32_t completed = reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);

    while (*reported < completed) {
        (*reported)++;
        uint32_t rel = (*reported) * reader->chunk_cap;
        if (rel > reader->total_tokens) rel = reader->total_tokens;
        uint32_t current = reader->progress_base + rel;
        if (current > reader->progress_total) current = reader->progress_total;
        ds4_session_report_progress(reader->progress_session,
                                    "prefill_chunk",
                                    (int)current,
                                    (int)reader->progress_total);
    }
}

static bool dist_prefill_reader_wait_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported) return true;

    pthread_mutex_lock(&reader->progress_mu);
    while (reader->progress_completed <= *reported && !reader->progress_done) {
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
    }
    const bool done = reader->progress_done;
    pthread_mutex_unlock(&reader->progress_mu);

    dist_prefill_reader_emit_progress(reader, reported);
    pthread_mutex_lock(&reader->progress_mu);
    const bool finished = done && *reported >= reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);
    return finished;
}

static bool dist_prefill_reader_wait_flow_window(
        ds4_dist_prefill_result_reader *reader,
        uint32_t submitted,
        uint32_t window,
        uint32_t *reported) {
    if (!reader || window == 0) return true;

    for (;;) {
        pthread_mutex_lock(&reader->progress_mu);
        const uint32_t completed = reader->progress_completed;
        const bool done = reader->progress_done;
        const bool has_room = submitted < completed + window;
        if (done || has_room) {
            pthread_mutex_unlock(&reader->progress_mu);
            dist_prefill_reader_emit_progress(reader, reported);
            return !done && has_room;
        }
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
        pthread_mutex_unlock(&reader->progress_mu);
        dist_prefill_reader_emit_progress(reader, reported);
    }
}

static void *dist_prefill_result_reader_main(void *arg) {
    ds4_dist_prefill_result_reader *reader = arg;
    reader->rc = 0;
    reader->err[0] = '\0';
    reader->final_kind = 0;
    reader->final_payload = NULL;
    reader->final_payload_bytes = 0;

    const uint64_t logits_bytes64 =
        (uint64_t)ds4_engine_vocab_size(reader->state->engine) * sizeof(float);
    if (logits_bytes64 == 0 || logits_bytes64 > UINT32_MAX) {
        snprintf(reader->err, sizeof(reader->err),
                 "distributed pipelined prefill logits size overflow");
        reader->rc = 1;
        shutdown(reader->fd, SHUT_RDWR);
        dist_prefill_reader_signal_progress(reader, 0, true);
        return NULL;
    }
    const uint32_t logits_bytes = (uint32_t)logits_bytes64;
    for (uint32_t i = 0; i < reader->count; i++) {
        const uint64_t request_id = reader->first_request_id + (uint64_t)i;
        const uint32_t pos0 = i * reader->chunk_cap;
        const uint32_t remaining = reader->total_tokens - pos0;
        const uint32_t chunk = remaining < reader->chunk_cap
            ? remaining : reader->chunk_cap;
        const uint64_t hidden_bytes64 =
            (uint64_t)chunk * reader->hc_values * sizeof(float);
        uint32_t hidden_wire_bytes = 0;
        if (hidden_bytes64 == 0 || hidden_bytes64 > UINT32_MAX ||
            !dist_activation_wire_bytes_from_f32_bytes(
                reader->state->activation_bits,
                (uint32_t)hidden_bytes64,
                &hidden_wire_bytes)) {
            snprintf(reader->err, sizeof(reader->err),
                     "distributed pipelined prefill result bounds overflow");
            reader->rc = 1;
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        const bool final_chunk = i + 1u == reader->count;
        const uint32_t expected_kind = reader->expected_kinds[i];
        const uint32_t allowed_kinds =
            DS4_DIST_RESULT_KIND_BIT(expected_kind);
        const ds4_dist_v3_result_limits result_limits = {
            .allowed_kinds = allowed_kinds,
            .hidden_wire_bytes = hidden_wire_bytes,
            .logits_bytes = logits_bytes,
            .nrows_bytes = 0,
        };
        uint32_t kind = 0;
        uint32_t payload_bytes = 0;
        uint64_t result_hash = 0;
        void *payload = NULL;
        int recv_rc = dist_recv_result_alloc(reader->fd,
                                             reader->transport,
                                             reader->state,
                                             request_id,
                                             &result_limits,
                                             &kind,
                                             &result_hash,
                                             &payload,
                                             &payload_bytes,
                                             reader->err,
                                             sizeof(reader->err));
        if (recv_rc != 0) {
            reader->rc = recv_rc;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (reader->expected_hashes && result_hash != reader->expected_hashes[i]) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill prefix hash mismatch");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        const bool valid_ack = expected_kind == DS4_DIST_RESULT_ACK &&
                               kind == DS4_DIST_RESULT_ACK &&
                               payload_bytes == 0;
        const bool valid_logits = expected_kind == DS4_DIST_RESULT_LOGITS &&
                                  kind == DS4_DIST_RESULT_LOGITS &&
                                  payload_bytes == logits_bytes;
        const bool valid_hidden = expected_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
                                  reader->allow_hidden &&
                                  hidden_bytes64 <= UINT32_MAX &&
                                  kind == DS4_DIST_RESULT_HIDDEN_STATE &&
                                  payload_bytes == (uint32_t)hidden_bytes64;
        if (!valid_ack && !valid_logits && !valid_hidden) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill returned invalid result");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (final_chunk) {
            reader->final_kind = kind;
            reader->final_payload = payload;
            reader->final_payload_bytes = payload_bytes;
            payload = NULL;
        }
        free(payload);
        dist_prefill_reader_signal_progress(reader, i + 1u, final_chunk);
    }
    dist_prefill_reader_signal_progress(reader, reader->count, true);
    return NULL;
}

static bool dist_coordinator_can_pipeline_prefill(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        ds4_session *session,
        uint32_t n_tokens,
        uint32_t chunk_cap) {
    if (getenv("DS4_DIST_DISABLE_PREFILL_PIPELINE")) return false;
    if (!state || !plan) return false;
    (void)session;
    if (chunk_cap == 0 || n_tokens <= chunk_cap) return false;
    if (plan->count == 0) return false;
    if (plan->entry[0].fd < 0) return false;
    const ds4_dist_route_entry *final = &plan->entry[plan->count - 1u];
    if ((final->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) {
        return final->layer_end + 1u == state->n_layers &&
               state->local_can_output_head;
    }
    return true;
}

static int dist_coordinator_prefill_chunk_cap(
        const ds4_dist_coordinator_state *state,
        ds4_session *session,
        uint32_t *chunk_cap,
        char *err,
        size_t errlen) {
    if (!chunk_cap) return 1;
    const int prefill_cap_i = ds4_session_prefill_cap(session);
    if (prefill_cap_i <= 0) {
        if (errlen) snprintf(err, errlen, "distributed coordinator has no prefill capacity");
        return 1;
    }
    const uint32_t prefill_cap = (uint32_t)prefill_cap_i;
    uint32_t requested = state ? state->prefill_chunk : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_CHUNK");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_CHUNK", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested == 0) requested = prefill_cap;
    if (requested > prefill_cap) {
        if (errlen) {
            snprintf(err,
                     errlen,
                     "distributed prefill chunk %u exceeds session prefill cap %u",
                     requested,
                     prefill_cap);
        }
        return 1;
    }
    *chunk_cap = requested;
    return 0;
}

static int dist_coordinator_prefill_window(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        uint32_t chunk_count,
        uint32_t *window,
        char *err,
        size_t errlen) {
    if (!window) return 1;
    uint32_t requested = state ? state->prefill_window : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_WINDOW");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_WINDOW", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested > 64u) {
        if (errlen) snprintf(err, errlen, "distributed prefill window %u exceeds limit 64", requested);
        return 1;
    }
    if (requested == 0) {
        const uint32_t remote_stages = plan ? plan->count : 0u;
        requested = remote_stages + 2u;
        if (requested < 2u) requested = 2u;
        if (requested > 8u) requested = 8u;
    }
    if (chunk_count != 0 && requested > chunk_count) requested = chunk_count;
    *window = requested ? requested : 1u;
    return 0;
}

static void dist_report_prefill_progress(ds4_session *session, uint32_t current, uint32_t total) {
    if (!session) return;
    if (current > (uint32_t)INT_MAX) current = (uint32_t)INT_MAX;
    if (total > (uint32_t)INT_MAX) total = (uint32_t)INT_MAX;
    ds4_session_report_progress(session, "prefill_chunk", (int)current, (int)total);
}

static int dist_coordinator_prefill_prompt_pipelined(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint32_t span_start,
        uint32_t n_tokens,
        bool reset_first_chunk,
        uint32_t chunk_cap,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    const uint32_t total = n_tokens;
    if (!prompt ||
        span_start > (uint32_t)prompt->len ||
        n_tokens == 0 ||
        n_tokens > (uint32_t)prompt->len - span_start) {
        if (errlen) snprintf(err, errlen, "invalid distributed pipelined prefill span");
        return 1;
    }
    const uint32_t span_end = span_start + n_tokens;
    const uint32_t chunk_count = (total + chunk_cap - 1u) / chunk_cap;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t max_hidden_bytes64 = (uint64_t)chunk_cap * hc_values * sizeof(float);
    if (max_hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    const uint32_t max_hidden_bytes = (uint32_t)max_hidden_bytes64;
    uint32_t flow_window = 0;
    if (dist_coordinator_prefill_window(state,
                                        plan,
                                        chunk_count,
                                        &flow_window,
                                        err,
                                        errlen) != 0) {
        return 1;
    }

    ds4_dist_prefill_sender sender;
    if (dist_prefill_sender_init(&sender,
                                 state,
                                 plan,
                                 prompt,
                                 session_id,
                                 plan->entry[0].fd,
                                 plan->entry[0].transport,
                                 chunk_count,
                                 max_hidden_bytes,
                                 err,
                                 errlen) != 0) {
        dist_prefill_sender_destroy(&sender);
        return 1;
    }

    ds4_dist_prefill_result_reader reader;
    memset(&reader, 0, sizeof(reader));
    reader.state = state;
    reader.fd = plan->entry[0].fd;
    reader.transport = plan->entry[0].transport;
    reader.progress_session = session;
    reader.first_request_id = *request_id;
    reader.count = chunk_count;
    reader.total_tokens = total;
    reader.chunk_cap = chunk_cap;
    reader.progress_base = span_start;
    reader.progress_total = (uint32_t)prompt->len;
    reader.hc_values = hc_values;
    reader.allow_hidden =
        (plan->entry[plan->count - 1u].flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0;
    reader.expected_final_kind = reader.allow_hidden
        ? DS4_DIST_RESULT_HIDDEN_STATE : DS4_DIST_RESULT_LOGITS;
    pthread_mutex_init(&reader.progress_mu, NULL);
    pthread_cond_init(&reader.progress_cv, NULL);
    reader.expected_hashes = calloc(chunk_count, sizeof(reader.expected_hashes[0]));
    reader.expected_kinds = calloc(chunk_count, sizeof(reader.expected_kinds[0]));
    if (!reader.expected_hashes || !reader.expected_kinds) {
        free(reader.expected_kinds);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "out of memory allocating distributed prefill expectations");
        return 1;
    }
    const bool intermediate_ack_only =
        getenv("DS4_DIST_DISABLE_PREFILL_ACK_ONLY") == NULL;
    uint64_t chunk_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    for (uint32_t i = 0, hash_pos = span_start; i < chunk_count; i++) {
        const uint32_t remaining = span_end - hash_pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        chunk_prefix_hash = dist_token_hash_update_span(chunk_prefix_hash,
                                                        prompt->v + hash_pos,
                                                        chunk);
        reader.expected_hashes[i] = chunk_prefix_hash;
        reader.expected_kinds[i] = i + 1u == chunk_count
            ? reader.expected_final_kind
            : intermediate_ack_only
                ? DS4_DIST_RESULT_ACK
                : reader.expected_final_kind;
        hash_pos += chunk;
    }

    pthread_t reader_tid;
    if (pthread_create(&reader_tid, NULL, dist_prefill_result_reader_main, &reader) != 0) {
        free(reader.expected_kinds);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill result reader");
        return 1;
    }
    pthread_t sender_tid;
    if (pthread_create(&sender_tid, NULL, dist_prefill_sender_main, &sender) != 0) {
        dist_prefill_sender_cancel(&sender);
        pthread_join(reader_tid, NULL);
        free(reader.expected_kinds);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill sender");
        return 1;
    }

    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: pipelined prefill %u chunks of up to %u tokens through %u worker%s, first hop %s:%u, send depth %u, flow window %u\n",
                     chunk_count,
                     chunk_cap,
                     plan->count,
                     plan->count == 1u ? "" : "s",
                     plan->entry[0].host,
                     plan->entry[0].port,
                     sender.slot_count,
                     flow_window);

    int rc = 0;
    double local_eval_sec = 0.0;
    const double pipeline_t0 = dist_now_sec();
    uint32_t pos = span_start;
    uint64_t next_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    uint32_t reported_chunks = 0;
    uint32_t submitted_chunks = 0;
    while (pos < span_end) {
        if (!dist_prefill_reader_wait_flow_window(&reader,
                                                  submitted_chunks,
                                                  flow_window,
                                                  &reported_chunks)) {
            if (errlen) snprintf(err, errlen, "distributed prefill result reader stopped");
            rc = 1;
            break;
        }
        const uint32_t remaining = span_end - pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * hc_values * sizeof(float);
        const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
        ds4_dist_prefill_send_slot *slot =
            dist_prefill_sender_acquire_slot(&sender, err, errlen);
        if (!slot) {
            rc = 1;
            break;
        }
        if (pos == span_start &&
            reset_first_chunk &&
            ds4_session_layer_slice_reset(session, err, errlen) != 0) {
            rc = 1;
            break;
        }
        const double local_t0 = dist_now_sec();
        rc = ds4_session_eval_layer_slice(session,
                                          prompt->v + pos,
                                          chunk,
                                          pos,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          slot->hidden,
                                          false,
                                          NULL,
                                          err,
                                          errlen);
        const double local_t1 = dist_now_sec();
        local_eval_sec += local_t1 - local_t0;
        if (rc != 0) break;

        slot->pos = pos;
        slot->n_tokens = chunk;
        slot->hidden_bytes = hidden_bytes;
        slot->request_id = *request_id;
        slot->prefix_hash = next_prefix_hash;
        slot->result_hash = reader.expected_hashes[submitted_chunks];
        slot->reset_session = reset_first_chunk && pos == span_start;
        slot->ack_only = reader.expected_kinds[submitted_chunks] ==
                         DS4_DIST_RESULT_ACK;
        rc = dist_prefill_sender_enqueue_slot(&sender, err, errlen);
        if (rc != 0) break;

        dist_prefill_reader_emit_progress(&reader, &reported_chunks);
        (*request_id)++;
        submitted_chunks++;
        next_prefix_hash = slot->result_hash;
        pos += chunk;
    }

    if (rc == 0) dist_prefill_sender_finish(&sender);
    else dist_prefill_sender_cancel(&sender);
    pthread_join(sender_tid, NULL);
    if (rc == 0 && sender.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender.err[0] ? sender.err : "distributed prefill sender failed");
        rc = 1;
    }
    if (rc != 0) {
        shutdown(plan->entry[0].fd, SHUT_RDWR);
    }
    if (rc == 0) {
        while (!dist_prefill_reader_wait_emit_progress(&reader, &reported_chunks)) {
            ;
        }
    }
    pthread_join(reader_tid, NULL);
    const double pipeline_t1 = dist_now_sec();
    if (rc == 0 && reader.rc == 0) {
        const double total_sec = pipeline_t1 - pipeline_t0;
        DIST_COORD_DEBUG(state,
                         "ds4: distributed coordinator: pipelined prefill done tokens=%u chunks=%u total=%.3fs %.2f t/s local=%.3fs send=%.3fs %.2f MiB/s\n",
                         total,
                         chunk_count,
                         total_sec,
                         total_sec > 0.0 ? (double)total / total_sec : 0.0,
                         local_eval_sec,
                         sender.send_sec,
                         sender.send_sec > 0.0
                             ? ((double)sender.send_bytes / (1024.0 * 1024.0)) / sender.send_sec
                             : 0.0);
    }
    if (reader.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s", reader.err[0] ? reader.err : "distributed pipelined prefill failed");
        int reader_rc = reader.rc;
        free(reader.final_payload);
        free(reader.expected_kinds);
        free(reader.expected_hashes);
        dist_prefill_sender_destroy(&sender);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        return reader_rc;
    }
    dist_prefill_sender_destroy(&sender);
    free(reader.expected_kinds);
    free(reader.expected_hashes);
    pthread_cond_destroy(&reader.progress_cv);
    pthread_mutex_destroy(&reader.progress_mu);
    if (rc != 0) {
        free(reader.final_payload);
        return 1;
    }
    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (reader.final_kind == DS4_DIST_RESULT_LOGITS &&
        reader.final_payload_bytes == logits_bytes) {
        memcpy(logits, reader.final_payload, logits_bytes);
        free(reader.final_payload);
        return 0;
    }
    if (reader.final_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        reader.final_payload) {
        const uint32_t last_pos = (chunk_count - 1u) * chunk_cap;
        const uint32_t last_tokens = total - last_pos;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           reader.final_payload,
                                                           last_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(reader.final_payload);
        return head_rc;
    }
    free(reader.final_payload);
    if (errlen) snprintf(err, errlen, "distributed pipelined prefill did not return a final result");
    return 1;
}

static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    uint32_t chunk_cap = 0;
    if (dist_coordinator_prefill_chunk_cap(state, session, &chunk_cap, err, errlen) != 0) {
        return 1;
    }
    const uint32_t prompt_len = (uint32_t)prompt->len;
    if (dist_coordinator_can_pipeline_prefill(state, plan, session, prompt_len, chunk_cap)) {
        return dist_coordinator_prefill_prompt_pipelined(state,
                                                        session,
                                                        plan,
                                                        prompt,
                                                        0,
                                                        prompt_len,
                                                        true,
                                                        chunk_cap,
                                                        session_id,
                                                        request_id,
                                                        logits,
                                                        err,
                                                        errlen);
    }

    uint32_t pos = 0;
    while (pos < prompt_len) {
        uint32_t remaining = prompt_len - pos;
        uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        int eval_rc = dist_coordinator_eval_span(state, session, plan,
                                                 prompt->v + pos, chunk, pos,
                                                 session_id, (*request_id)++,
                                                 pos == 0, 0, false, NULL,
                                                 logits, NULL, NULL, err,
                                                 errlen);
        if (eval_rc != 0) {
            return eval_rc;
        }
        pos += chunk;
        dist_report_prefill_progress(session, pos, prompt_len);
    }
    return 0;
}

/* =========================================================================
 * Standalone Coordinator Recovery And Generation
 * ========================================================================= */

static int dist_replay_check_logits(
        ds4_dist_coordinator_state *state,
        const float *before,
        const float *after) {
    const int vocab = ds4_engine_vocab_size(state->engine);
    float max_abs = 0.0f;
    int max_i = 0;
    uint32_t mismatches = 0;
    for (int i = 0; i < vocab; i++) {
        if (before[i] != after[i]) {
            const float d = fabsf(before[i] - after[i]);
            if (d > max_abs) {
                max_abs = d;
                max_i = i;
            }
            mismatches++;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "ds4: distributed replay check failed: mismatches=%u max_abs=%g token=%d before=%g after=%g\n",
                mismatches,
                max_abs,
                max_i,
                before[max_i],
                after[max_i]);
        return 1;
    }
    fprintf(stderr, "ds4: distributed replay check passed: logits exact match across reset/replay\n");
    return 0;
}

static int dist_run_coordinator_generation(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen) {
    char err[256];
    ds4_dist_route_plan plan;
    uint64_t plan_generation = 0;
    if (!dist_coordinator_ensure_route(state, &plan, &plan_generation, err, sizeof(err))) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, state->engine, gen->ctx_size) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: failed to create local session\n");
        dist_route_plan_free(&plan);
        return 1;
    }

    ds4_tokens prompt = {0};
    if (dist_prompt_is_rendered_chat(gen->prompt)) {
        ds4_tokenize_rendered_chat(state->engine, gen->prompt, &prompt);
    } else {
        ds4_encode_chat_prompt(state->engine, gen->system, gen->prompt, gen->think_mode, &prompt);
    }
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4: distributed coordinator: empty prompt\n");
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    const uint64_t session_id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    uint64_t request_id = 1;
    float *logits = malloc((size_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ds4: distributed coordinator: out of memory allocating logits\n");
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     err,
                                                     sizeof(err));
    if (prefill_rc != 0) {
        fprintf(stderr,
                "ds4: distributed prompt processing failed: %s\n",
                err);
        if (dist_coordinator_rebuild_from_transcript(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     NULL,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed prompt recovery failed: %s\n",
                    err);
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (state->replay_check) {
        const size_t logits_bytes = (size_t)ds4_engine_vocab_size(state->engine) * sizeof(logits[0]);
        float *before = malloc(logits_bytes);
        if (!before) {
            fprintf(stderr, "ds4: distributed replay check: out of memory allocating logits copy\n");
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
        memcpy(before, logits, logits_bytes);
        int replay_prefill_rc = dist_coordinator_prefill_prompt(state,
                                                                session,
                                                                &plan,
                                                                &prompt,
                                                                session_id,
                                                                &request_id,
                                                                logits,
                                                                err,
                                                                sizeof(err));
        if (replay_prefill_rc != 0) {
            fprintf(stderr,
                    "ds4: distributed replay prompt processing failed: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &prompt,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         replay_prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed replay recovery failed: %s\n",
                        err);
                free(before);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
        int replay_rc = dist_replay_check_logits(state, before, logits);
        free(before);
        if (replay_rc != 0) {
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (gen->dump_logits_path) {
        int rc = dist_write_logits_dump(state, gen, &prompt, &plan, logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }
    if (gen->dump_logprobs_path) {
        int rc = dist_write_logprobs_dump(state,
                                          gen,
                                          &prompt,
                                          &plan,
                                          session,
                                          session_id,
                                          &request_id,
                                          logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }

    int generated = 0;
    int max_tokens = gen->n_predict > 0 ? gen->n_predict : 1;
    int room = gen->ctx_size - prompt.len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    uint64_t rng = gen->seed ? gen->seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    const int eos = ds4_token_eos(state->engine);
    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, &prompt);
    while (generated < max_tokens) {
        int token = ds4_sample_logits(logits,
                                      ds4_engine_vocab_size(state->engine),
                                      gen->temperature,
                                      0,
                                      gen->top_p,
                                      gen->min_p,
                                      &rng);
        if (token == eos) break;

        size_t len = 0;
        char *text = ds4_token_text(state->engine, token, &len);
        if (len) fwrite(text, 1, len, stdout);
        fflush(stdout);
        free(text);

        uint32_t token_pos = (uint32_t)prompt.len + (uint32_t)generated;
        generated++;
        ds4_tokens_push(&transcript, token);
        if (generated >= max_tokens) break;

        int decode_rc = dist_coordinator_eval_span(state, session, &plan,
                                                   &token, 1, token_pos,
                                                   session_id, request_id++,
                                                   false, 0, false, NULL,
                                                   logits, NULL, NULL, err,
                                                   sizeof(err));
        if (decode_rc != 0) {
            fprintf(stderr, "\nds4: distributed decode failed: %s\n", err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &transcript,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         decode_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr, "ds4: distributed decode recovery failed: %s\n", err);
                ds4_tokens_free(&transcript);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
    }
    fputc('\n', stdout);

    ds4_tokens_free(&transcript);
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    dist_route_plan_free(&plan);
    return 0;
}

/* =========================================================================
 * Coordinator Control Plane
 * ========================================================================= */

static void dist_coordinator_remove_worker(ds4_dist_coordinator_state *state, int fd) {
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *entry = *link;
        if (entry->fd == fd) {
            *link = entry->next;
            state->generation++;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: removed worker %s:%s layers=%u:%u%s\n",
                             entry->peer_host,
                             entry->peer_port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            pthread_mutex_unlock(&state->mu);
            ds4_transport_release(entry->transport);
            free(entry);
            if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
            return;
        }
        link = &entry->next;
    }
    pthread_mutex_unlock(&state->mu);
}

static void dist_coordinator_monitor_worker_fd(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port) {
    for (;;) {
        struct pollfd pfd = {
            .fd = fd,
            .events = 0,
            .revents = 0,
        };
        int rc = poll(&pfd, 1, 1000);
        if (rc < 0) {
            if (errno == EINTR) continue;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: worker %s:%s poll failed: %s\n",
                             peer_host,
                             peer_port,
                             strerror(errno));
            break;
        }
        if (rc == 0) continue;
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) break;
    }
}

static void *dist_coordinator_client_main(void *arg) {
    ds4_dist_client_ctx *ctx = arg;
    int fd = ctx->fd;
    ds4_dist_coordinator_state *state = ctx->state;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    ds4_dist_hello_fixed hello;
    ds4_dist_v3_hello_ext hello_v3;
    bool is_v3 = false;
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    char err[256];
    int rc = dist_recv_hello(fd, &hello, &hello_v3, &is_v3, model_name, sizeof(model_name), err, sizeof(err));
    if (rc <= 0) {
        if (rc < 0) DIST_COORD_DEBUG(state, "ds4: distributed coordinator: bad HELLO from %s:%s: %s\n", peer_host, peer_port, err);
        close(fd);
        return NULL;
    }

    if (hello.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: worker=%u coordinator=%u", hello.model_id, state->model_id);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    const char *expected_model_name = ds4_engine_model_name(state->engine);
    if (!expected_model_name) expected_model_name = "unknown";
    if (strcmp(model_name, expected_model_name) != 0) {
        snprintf(err,
                 sizeof(err),
                 "model family mismatch: worker=%s coordinator=%s",
                 model_name,
                 expected_model_name);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.n_layers != state->n_layers) {
        snprintf(err, sizeof(err), "layer count mismatch: worker=%u coordinator=%u", hello.n_layers, state->n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.quant_bits != 2u && hello.quant_bits != 4u) {
        snprintf(err, sizeof(err), "unsupported worker quant profile Q%u", hello.quant_bits);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output > 1u) {
        snprintf(err, sizeof(err), "invalid worker output-head flag %u", hello.has_output);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_hidden > 1u) {
        snprintf(err, sizeof(err), "invalid worker hidden-state flag %u", hello.has_hidden);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.layer_start >= hello.n_layers || hello.layer_end >= hello.n_layers || hello.layer_end < hello.layer_start) {
        snprintf(err, sizeof(err), "invalid worker layer range %u:%u for %u layers", hello.layer_start, hello.layer_end, hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output && hello.layer_end + 1u != hello.n_layers) {
        snprintf(err,
                 sizeof(err),
                 "worker output head requires final layer: range=%u:%u layers=%u",
                 hello.layer_start,
                 hello.layer_end,
                 hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (state->ctx_size != 0 && hello.ctx_size < state->ctx_size) {
        snprintf(err,
                 sizeof(err),
                 "worker context too small: worker=%u coordinator=%u",
                 hello.ctx_size,
                 state->ctx_size);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.listen_port == 0 || hello.listen_port > 65535u) {
        snprintf(err, sizeof(err), "invalid worker data listen port %u", hello.listen_port);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }

    ds4_transport *prepared_transport = NULL;
    uint32_t protocol_version = DS4_DIST_PROTOCOL_VERSION;
    uint32_t selected_caps = 0;
    if (is_v3) {
        if (ds4_dist_v3_hello_ext_validate(&hello_v3, err, sizeof(err)) != 0) {
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: rejecting v3 HELLO from %s:%s: %s\n",
                             peer_host, peer_port, err);
            dist_send_error(fd, err);
            close(fd);
            return NULL;
        }
        prepared_transport = dist_coordinator_negotiate_v3(
            state, fd, &hello_v3, &selected_caps, err, sizeof(err));
        if (!prepared_transport) {
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: v3 negotiation with %s:%s failed: %s\n",
                             peer_host, peer_port,
                             err[0] ? err : strerror(errno));
            dist_send_error(fd, err[0] ? err : "v3 transport negotiation failed");
            close(fd);
            return NULL;
        }
        protocol_version = DS4_DIST_V3_PROTOCOL_VERSION;
    } else if (state->transport_policy == DS4_DIST_TRANSPORT_NHI) {
        snprintf(err, sizeof(err),
                 "coordinator requires v3 NHI but worker sent a v2 HELLO");
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }

    if (!dist_coordinator_add_worker(state, fd, peer_host, peer_port,
                                     &hello, model_name, prepared_transport,
                                     protocol_version, selected_caps)) {
        ds4_transport_release(prepared_transport);
        dist_send_error(fd, "failed to create persistent worker transport");
        close(fd);
        return NULL;
    }
    ds4_transport_release(prepared_transport);

    if (state->use_control_for_work) {
        dist_coordinator_monitor_worker_fd(state, fd, peer_host, peer_port);
    } else {
        for (;;) {
            uint32_t type = 0, bytes = 0;
            rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
            if (rc == 0) break;
            if (rc < 0) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s protocol error: %s\n", peer_host, peer_port, err);
                break;
            }
            if (type == DS4_DIST_MSG_HELLO) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s sent duplicate HELLO\n", peer_host, peer_port);
                dist_discard_bytes(fd, bytes);
                break;
            }
            rc = dist_discard_bytes(fd, bytes);
            if (rc <= 0) break;
        }
    }

    dist_coordinator_remove_worker(state, fd);
    close(fd);
    return NULL;
}

static void *dist_coordinator_accept_main(void *arg) {
    ds4_dist_accept_ctx *accept_ctx = arg;
    int listen_fd = accept_ctx->listen_fd;
    ds4_dist_coordinator_state *state = accept_ctx->state;

    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (errno == EBADF || errno == EINVAL) break;
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory accepting worker\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_coordinator_client_main, ctx) != 0) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: pthread_create failed\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

static uint64_t dist_make_session_id(const void *ptr) {
    uint64_t id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    id ^= ((uint64_t)(uintptr_t)ptr << 17) ^ (uint64_t)(uintptr_t)ptr;
    id ^= (uint64_t)clock();
    return id ? id : 1u;
}

static int dist_session_ensure_route(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return 1;
    }
    uint64_t generation = dist_coordinator_generation(&d->state);
    if (d->plan_ready && d->plan_generation == generation) return 0;
    dist_route_plan_free(&d->plan);
    if (!dist_coordinator_ensure_route(&d->state, &d->plan, &generation, err, errlen)) {
        d->plan_ready = false;
        d->plan_generation = 0;
        return 1;
    }
    d->plan_ready = true;
    d->plan_generation = generation;
    return 0;
}

/* =========================================================================
 * Distributed KV Snapshot Transport
 * ========================================================================= */

static int dist_write_snapshot_load_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX ||
        begin->token_bytes != (uint32_t)token_bytes64 ||
        begin->message_bytes != 0)
        return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN,
                                (uint32_t)frame_bytes64) != 0)
        return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 1;
}

static int dist_read_snapshot_begin_frame(
        int fd,
        ds4_dist_snapshot_begin_fixed *begin,
        char *msg,
        size_t msg_cap,
        char *err,
        size_t errlen) {
    if (msg_cap) msg[0] = '\0';
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed snapshot connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_BEGIN ||
        bytes < sizeof(ds4_dist_snapshot_begin_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot frame");
        return 1;
    }
    rc = dist_read_full(fd, begin, sizeof(*begin));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot header");
        return 1;
    }
    dist_snapshot_begin_from_wire(begin);
    uint32_t body = bytes - (uint32_t)sizeof(*begin);
    uint64_t expected_token_bytes = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin->token_bytes != (uint32_t)expected_token_bytes ||
        begin->token_bytes > body ||
        begin->message_bytes > body - begin->token_bytes) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot response header");
        return 1;
    }
    if (begin->token_bytes != 0) {
        rc = dist_discard_bytes(fd, begin->token_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response tokens");
            return 1;
        }
        body -= begin->token_bytes;
    }
    if (begin->message_bytes != 0) {
        uint32_t n = begin->message_bytes;
        uint32_t copy = msg_cap && n < msg_cap ? n : (msg_cap ? (uint32_t)msg_cap - 1u : 0u);
        if (copy != 0) {
            rc = dist_read_full(fd, msg, copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to read distributed snapshot response message");
                return 1;
            }
            msg[copy] = '\0';
        }
        if (n > copy) {
            rc = dist_discard_bytes(fd, n - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response message");
                return 1;
            }
        }
        body -= n;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot response bytes");
            return 1;
        }
    }
    return 0;
}

static int dist_read_snapshot_done_frame(
        int fd,
        uint64_t request_id,
        char *err,
        size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed before snapshot completion");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_DONE ||
        bytes < sizeof(ds4_dist_snapshot_done_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot completion frame");
        return 1;
    }
    ds4_dist_snapshot_done_fixed done;
    rc = dist_read_full(fd, &done, sizeof(done));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion");
        return 1;
    }
    dist_snapshot_done_from_wire(&done);
    uint32_t body = bytes - (uint32_t)sizeof(done);
    char msg[256];
    msg[0] = '\0';
    if (done.message_bytes > body) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot completion message");
        return 1;
    }
    if (done.message_bytes != 0) {
        uint32_t copy = done.message_bytes < sizeof(msg) ?
            done.message_bytes : (uint32_t)sizeof(msg) - 1u;
        rc = dist_read_full(fd, msg, copy);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion message");
            return 1;
        }
        msg[copy] = '\0';
        if (done.message_bytes > copy) {
            rc = dist_discard_bytes(fd, done.message_bytes - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot completion message");
                return 1;
            }
        }
        body -= done.message_bytes;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot completion bytes");
            return 1;
        }
    }
    uint64_t got_request = dist_u64_from_halves(done.request_hi, done.request_lo);
    if (got_request != request_id) {
        if (errlen) snprintf(err, errlen, "distributed snapshot completion request mismatch");
        return 1;
    }
    if (done.status != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             msg[0] ? msg : "distributed worker failed snapshot request");
        return 1;
    }
    return 0;
}

static int dist_receive_snapshot_chunks_to_file(
        int fd,
        uint64_t request_id,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory receiving distributed KV shard");
        return 1;
    }
    uint64_t received = 0;
    int fail = 0;
    while (!fail && received < payload_bytes) {
        uint32_t type = 0, bytes = 0;
        int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
        if (rc <= 0) {
            if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed while sending KV shard");
            fail = 1;
            break;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(fd, bytes);
            if (errlen) snprintf(err, errlen, "expected distributed KV shard chunk");
            fail = 1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk header");
            fail = 1;
            break;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi,
                                                    chunk.request_lo);
        uint32_t chunk_bytes = bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(fd, chunk_bytes);
            if (errlen) snprintf(err, errlen, "invalid distributed KV shard chunk");
            fail = 1;
            break;
        }
        rc = dist_read_full(fd, buf, chunk_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk");
            fail = 1;
            break;
        }
        if (fwrite(buf, 1, chunk_bytes, fp) != chunk_bytes) {
            if (errlen) snprintf(err, errlen, "failed to write distributed KV shard");
            fail = 1;
            break;
        }
        received += chunk_bytes;
    }
    free(buf);
    return fail;
}

/* =========================================================================
 * DSV4 Payload Assembly And Sharding
 * =========================================================================
 *
 * Save and load operate on the normal single-node DSV4 payload format. This
 * block maps that payload to/from the layer slices advertised by the current
 * distributed route, keeping session files independent from cluster topology.
 */

static uint64_t dist_kv_state_bytes(uint32_t ratio, uint32_t head_dim, bool *ok) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    uint64_t bytes = 0;
    uint64_t tmp = 0;
    if (!dist_u64_mul(coff, head_dim, &tmp) ||
        !dist_u64_mul(tmp, coff, &tmp) ||
        !dist_u64_mul(tmp, ratio, &tmp) ||
        !dist_u64_mul(tmp, sizeof(float), &bytes)) {
        if (ok) *ok = false;
        return 0;
    }
    return bytes;
}

static bool dist_kv_layer_tensor_bytes(
        ds4_engine *engine,
        const ds4_dist_kv_layout *layout,
        uint32_t layer,
        uint32_t n_comp,
        uint32_t n_index_comp,
        uint64_t *out) {
    if (!layout || !out) return false;
    if (ds4_engine_is_glm_dsa(engine)) {
        return ds4_engine_glm_layer_payload_bytes(engine,
                                                  layer,
                                                  layout->raw_live,
                                                  layout->head_dim,
                                                  layout->indexer_head_dim,
                                                  n_comp,
                                                  n_index_comp,
                                                  out);
    }
    uint64_t bytes = 0;
    uint64_t tmp = 0;
    if (!dist_u64_mul(layout->raw_live, layout->head_dim, &tmp) ||
        !dist_u64_mul(tmp, sizeof(float), &tmp) ||
        !dist_u64_add(&bytes, tmp))
        return false;

    const uint32_t ratio = ds4_engine_layer_compress_ratio(engine, layer);
    if (ratio != 0) {
        if (!dist_u64_mul(n_comp, layout->head_dim, &tmp) ||
            !dist_u64_mul(tmp, sizeof(float), &tmp) ||
            !dist_u64_add(&bytes, tmp))
            return false;
        bool ok = true;
        const uint64_t attn_state = dist_kv_state_bytes(ratio, layout->head_dim, &ok);
        if (!ok ||
            !dist_u64_add(&bytes, attn_state) ||
            !dist_u64_add(&bytes, attn_state))
            return false;
        if (ratio == 4u) {
            if (!dist_u64_mul(n_index_comp, layout->indexer_head_dim, &tmp) ||
                !dist_u64_mul(tmp, sizeof(float), &tmp) ||
                !dist_u64_add(&bytes, tmp))
                return false;
            const uint64_t index_state =
                dist_kv_state_bytes(ratio, layout->indexer_head_dim, &ok);
            if (!ok ||
                !dist_u64_add(&bytes, index_state) ||
                !dist_u64_add(&bytes, index_state))
                return false;
        }
    }

    *out = bytes;
    return true;
}

static bool dist_kv_layout_matches(
        const ds4_dist_kv_layout *a,
        const ds4_dist_kv_layout *b) {
    return a->ctx == b->ctx &&
           a->prefill_cap == b->prefill_cap &&
           a->raw_cap == b->raw_cap &&
           a->raw_window == b->raw_window &&
           a->comp_cap == b->comp_cap &&
           a->token_count == b->token_count &&
           a->n_layers == b->n_layers &&
           a->head_dim == b->head_dim &&
           a->indexer_head_dim == b->indexer_head_dim &&
           a->raw_live == b->raw_live;
}

static bool dist_kv_raw_live_valid(ds4_engine *engine, const ds4_dist_kv_layout *layout) {
    if (!layout || layout->raw_window == 0 || layout->raw_cap == 0) return false;
    const uint32_t max_live =
        layout->token_count < layout->raw_window ? layout->token_count : layout->raw_window;
    if (ds4_engine_is_glm_dsa(engine)) {
        return layout->raw_live <= max_live &&
               layout->raw_live <= layout->raw_cap;
    }
    return layout->raw_live == max_live && layout->raw_live <= layout->raw_cap;
}

static int dist_kv_parse_layer_payload(
        ds4_engine *engine,
        FILE *fp,
        uint64_t bytes,
        uint32_t expected_start,
        uint32_t expected_end,
        ds4_dist_kv_layout *layout,
        bool *layout_set,
        uint32_t *n_comp,
        uint32_t *n_index_comp,
        ds4_dist_kv_shard_file *shard,
        char *err,
        size_t errlen) {
    if (dist_rewind_file(fp, "distributed KV shard", err, errlen) != 0)
        return 1;
    uint64_t remaining = bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
            return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        if (errlen) snprintf(err, errlen, "unsupported distributed KV layer payload");
        return 1;
    }
    ds4_dist_kv_layout got = {
        .ctx = h[2],
        .prefill_cap = h[3],
        .raw_cap = h[4],
        .raw_window = h[5],
        .comp_cap = h[6],
        .token_count = h[7],
        .n_layers = h[8],
        .head_dim = h[9],
        .indexer_head_dim = h[10],
        .vocab = layout ? layout->vocab : 0,
        .raw_live = h[13],
    };
    const uint32_t layer_start = h[11];
    const uint32_t layer_end = h[12];
    if (layer_start != expected_start ||
        layer_end != expected_end ||
        layer_start > layer_end ||
        layer_end >= got.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV shard range mismatch");
        return 1;
    }
    if (got.n_layers != (uint32_t)ds4_engine_layer_count(engine) ||
        !dist_kv_raw_live_valid(engine, &got)) {
        if (errlen) snprintf(err, errlen, "distributed KV shard layout is invalid");
        return 1;
    }
    if (layout_set && *layout_set) {
        if (!dist_kv_layout_matches(layout, &got)) {
            if (errlen) snprintf(err, errlen, "distributed KV shards use different layouts");
            return 1;
        }
    } else if (layout && layout_set) {
        const uint32_t vocab = layout->vocab;
        *layout = got;
        layout->vocab = vocab;
        *layout_set = true;
    }

    const uint32_t slice_layers = layer_end - layer_start + 1u;
    for (uint32_t i = 0; i < slice_layers; i++) {
        const uint32_t il = layer_start + i;
        if (dist_payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0)
            return 1;
        if (n_comp[il] > got.comp_cap) {
            if (errlen) snprintf(err, errlen, "distributed KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < slice_layers; i++) {
        const uint32_t il = layer_start + i;
        if (dist_payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0)
            return 1;
        if (n_index_comp[il] > got.comp_cap) {
            if (errlen) snprintf(err, errlen, "distributed KV shard has invalid indexer row count");
            return 1;
        }
    }

    off_t tensor_pos = ftello(fp);
    if (tensor_pos < 0) {
        if (errlen) snprintf(err, errlen, "failed to locate distributed KV shard tensors");
        return 1;
    }
    uint64_t expected_tensor_bytes = 0;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        uint64_t layer_bytes = 0;
        if (!dist_kv_layer_tensor_bytes(engine, &got, il,
                                        n_comp[il], n_index_comp[il],
                                        &layer_bytes) ||
            !dist_u64_add(&expected_tensor_bytes, layer_bytes)) {
            if (errlen) snprintf(err, errlen, "distributed KV shard tensor size overflow");
            return 1;
        }
    }
    if (remaining != expected_tensor_bytes) {
        if (errlen) snprintf(err, errlen, "distributed KV shard tensor byte count mismatch");
        return 1;
    }
    if (shard) {
        shard->layer_start = layer_start;
        shard->layer_end = layer_end;
        shard->tensor_offset = (uint64_t)tensor_pos;
        shard->tensor_bytes = remaining;
    }
    return 0;
}

static int dist_kv_write_session_header(
        FILE *fp,
        const ds4_dist_kv_layout *layout,
        char *err,
        size_t errlen) {
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        layout->ctx,
        layout->prefill_cap,
        layout->raw_cap,
        layout->raw_window,
        layout->comp_cap,
        layout->token_count,
        layout->n_layers,
        layout->head_dim,
        layout->indexer_head_dim,
        layout->vocab,
        layout->raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    return 0;
}

static int dist_kv_write_layer_header(
        FILE *fp,
        const ds4_dist_kv_layout *layout,
        uint32_t layer_start,
        uint32_t layer_end,
        const uint32_t *n_comp,
        const uint32_t *n_index_comp,
        char *err,
        size_t errlen) {
    const uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        layout->ctx,
        layout->prefill_cap,
        layout->raw_cap,
        layout->raw_window,
        layout->comp_cap,
        layout->token_count,
        layout->n_layers,
        layout->head_dim,
        layout->indexer_head_dim,
        layer_start,
        layer_end,
        layout->raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (dist_payload_write_u32(fp, n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (dist_payload_write_u32(fp, n_index_comp[il], err, errlen) != 0) return 1;
    }
    return 0;
}

static uint32_t dist_kv_route_shard_count(const ds4_dist_session *d) {
    return d ? 1u + d->plan.count : 0u;
}

static void dist_kv_route_shard(
        const ds4_dist_session *d,
        uint32_t shard,
        uint32_t *layer_start,
        uint32_t *layer_end,
        const ds4_dist_route_entry **entry) {
    if (entry) *entry = NULL;
    if (shard == 0) {
        if (layer_start) *layer_start = d->state.local_start;
        if (layer_end) *layer_end = d->state.local_end;
        return;
    }
    const ds4_dist_route_entry *e = &d->plan.entry[shard - 1u];
    if (layer_start) *layer_start = e->layer_start;
    if (layer_end) *layer_end = e->layer_end;
    if (entry) *entry = e;
}

static int dist_kv_route_validate(
        const ds4_dist_session *d,
        char *err,
        size_t errlen) {
    if (!d || d->state.n_layers == 0 ||
        d->state.local_start != 0 ||
        d->state.local_end >= d->state.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV route does not start at layer 0");
        return 1;
    }
    uint32_t prev = d->state.local_end;
    for (uint32_t i = 0; i < d->plan.count; i++) {
        const ds4_dist_route_entry *e = &d->plan.entry[i];
        if (prev == UINT32_MAX ||
            e->layer_start != prev + 1u ||
            e->layer_end < e->layer_start ||
            e->layer_end >= d->state.n_layers) {
            if (errlen) snprintf(err, errlen, "distributed KV route is not contiguous");
            return 1;
        }
        prev = e->layer_end;
    }
    if (prev + 1u != d->state.n_layers) {
        if (errlen) snprintf(err, errlen, "distributed KV route does not cover all layers");
        return 1;
    }
    return 0;
}

static void dist_kv_shards_close(ds4_dist_kv_shard_file *shards, uint32_t count) {
    if (!shards) return;
    for (uint32_t i = 0; i < count; i++) {
        if (shards[i].fp) fclose(shards[i].fp);
    }
}

static int dist_save_remote_shard_to_file(
        ds4_dist_session *d,
        const ds4_dist_route_entry *entry,
        const ds4_tokens *tokens,
        uint64_t token_hash,
        FILE *fp,
        uint64_t *payload_bytes_out,
        char *err,
        size_t errlen) {
    if (payload_bytes_out) *payload_bytes_out = 0;
    uint64_t request_id = d->snapshot_request_id++;
    int fd = dist_connect_endpoint(entry->host, (int)entry->port, err, errlen);
    if (fd < 0) return 1;

    ds4_dist_snapshot_req_fixed req;
    memset(&req, 0, sizeof(req));
    req.model_id = d->state.model_id;
    dist_u64_to_halves(d->session_id, &req.session_hi, &req.session_lo);
    dist_u64_to_halves(request_id, &req.request_hi, &req.request_lo);
    dist_u64_to_halves(token_hash, &req.token_hash_hi, &req.token_hash_lo);
    req.token_count = (uint32_t)tokens->len;
    req.layer_start = entry->layer_start;
    req.layer_end = entry->layer_end;
    ds4_dist_snapshot_req_fixed wire = req;
    dist_snapshot_req_to_wire(&wire);
    int rc = 1;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_SAVE_REQ,
                                (uint32_t)sizeof(wire)) != 0 ||
        dist_write_full(fd, &wire, sizeof(wire)) != 0) {
        if (errlen) snprintf(err, errlen, "failed to request distributed KV shard");
        goto cleanup;
    }

    ds4_dist_snapshot_begin_fixed begin;
    char msg[256];
    if (dist_read_snapshot_begin_frame(fd, &begin, msg, sizeof(msg),
                                       err, errlen) != 0)
        goto cleanup;
    if (begin.status != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             msg[0] ? msg : "distributed worker refused KV snapshot");
        goto cleanup;
    }
    const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi,
                                                        begin.payload_lo);
    uint64_t got_session = dist_u64_from_halves(begin.session_hi, begin.session_lo);
    uint64_t got_request = dist_u64_from_halves(begin.request_hi, begin.request_lo);
    uint64_t got_hash = dist_u64_from_halves(begin.token_hash_hi, begin.token_hash_lo);
    if (begin.model_id != d->state.model_id ||
        got_session != d->session_id ||
        got_request != request_id ||
        got_hash != token_hash ||
        (begin.token_count != 0 && begin.token_count != (uint32_t)tokens->len) ||
        begin.layer_start != entry->layer_start ||
        begin.layer_end != entry->layer_end) {
        if (errlen) snprintf(err, errlen, "distributed KV shard metadata mismatch");
        goto cleanup;
    }
    if (dist_receive_snapshot_chunks_to_file(fd, request_id, fp,
                                             payload_bytes, err, errlen) != 0)
        goto cleanup;
    if (dist_read_snapshot_done_frame(fd, request_id, err, errlen) != 0)
        goto cleanup;
    if (payload_bytes_out) *payload_bytes_out = payload_bytes;
    rc = 0;

cleanup:
    close(fd);
    return rc;
}

static int dist_prepare_shard_from_session_payload(
        ds4_dist_session *d,
        FILE *src,
        uint64_t *remaining,
        const ds4_dist_kv_layout *layout,
        const uint32_t *n_comp,
        const uint32_t *n_index_comp,
        uint32_t layer_start,
        uint32_t layer_end,
        FILE **tmp_out,
        uint64_t *payload_bytes_out,
        char *err,
        size_t errlen) {
    if (tmp_out) *tmp_out = NULL;
    if (payload_bytes_out) *payload_bytes_out = 0;
    FILE *tmp = dist_tmpfile_or_err("distributed KV shard", err, errlen);
    if (!tmp) return 1;

    int rc = 1;
    if (dist_kv_write_layer_header(tmp, layout, layer_start, layer_end,
                                   n_comp, n_index_comp, err, errlen) != 0)
        goto cleanup;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        uint64_t layer_bytes = 0;
        if (!dist_kv_layer_tensor_bytes(d->state.engine, layout, il,
                                        n_comp[il], n_index_comp[il],
                                        &layer_bytes)) {
            if (errlen) snprintf(err, errlen, "distributed KV layer byte count overflow");
            goto cleanup;
        }
        if (dist_payload_copy_bytes(src, tmp, layer_bytes, remaining, err, errlen) != 0)
            goto cleanup;
    }
    uint64_t bytes = 0;
    if (dist_measure_file(tmp, &bytes, "distributed KV shard", err, errlen) != 0 ||
        dist_rewind_file(tmp, "distributed KV shard", err, errlen) != 0)
        goto cleanup;
    *tmp_out = tmp;
    if (payload_bytes_out) *payload_bytes_out = bytes;
    rc = 0;

cleanup:
    if (rc != 0) fclose(tmp);
    return rc;
}

static int dist_load_remote_shard_from_payload(
        ds4_dist_session *d,
        const ds4_dist_route_entry *entry,
        const int *tokens,
        uint32_t token_count,
        uint64_t token_hash,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    uint64_t request_id = d->snapshot_request_id++;
    int fd = dist_connect_endpoint(entry->host, (int)entry->port, err, errlen);
    if (fd < 0) return 1;

    ds4_dist_snapshot_begin_fixed begin;
    memset(&begin, 0, sizeof(begin));
    begin.model_id = d->state.model_id;
    dist_u64_to_halves(d->session_id, &begin.session_hi, &begin.session_lo);
    dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
    dist_u64_to_halves(token_hash, &begin.token_hash_hi, &begin.token_hash_lo);
    begin.token_count = token_count;
    begin.layer_start = entry->layer_start;
    begin.layer_end = entry->layer_end;
    dist_u64_to_halves(payload_bytes, &begin.payload_hi, &begin.payload_lo);
    begin.token_bytes = token_count * sizeof(uint32_t);

    int rc = 1;
    if (dist_write_snapshot_load_begin(fd, &begin, tokens) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed KV shard restore request");
        goto cleanup;
    }
    if (dist_send_snapshot_file_chunks(fd, request_id, fp, payload_bytes) <= 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed KV shard restore payload");
        goto cleanup;
    }
    if (dist_read_snapshot_done_frame(fd, request_id, err, errlen) != 0)
        goto cleanup;
    rc = 0;

cleanup:
    close(fd);
    return rc;
}

/* =========================================================================
 * Coordinator KV Payload API
 * ========================================================================= */

int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen) {
    if (!d || !owner || !fp) {
        if (errlen) snprintf(err, errlen, "invalid distributed payload save");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;
    if (dist_kv_route_validate(d, err, errlen) != 0) return 1;

    const ds4_tokens *tokens = ds4_session_tokens(owner);
    if (!tokens || tokens->len < 0 || (uint64_t)tokens->len > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed session has no valid token timeline");
        return 1;
    }
    const uint32_t token_count = (uint32_t)tokens->len;
    const uint64_t token_hash = dist_token_hash_prefix(tokens->v, token_count);
    const uint32_t vocab = (uint32_t)ds4_engine_vocab_size(d->state.engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        if (errlen) snprintf(err, errlen, "out of memory saving distributed logits");
        return 1;
    }
    if (ds4_session_copy_logits(owner, logits, (int)vocab) != (int)vocab) {
        free(logits);
        if (errlen) snprintf(err, errlen, "failed to copy distributed logits");
        return 1;
    }

    const uint32_t shard_count = dist_kv_route_shard_count(d);
    ds4_dist_kv_shard_file *shards = calloc(shard_count, sizeof(shards[0]));
    uint32_t *n_comp = calloc(d->state.n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = calloc(d->state.n_layers, sizeof(n_index_comp[0]));
    if (!shards || !n_comp || !n_index_comp) {
        free(logits);
        free(shards);
        free(n_comp);
        free(n_index_comp);
        if (errlen) snprintf(err, errlen, "out of memory saving distributed KV payload");
        return 1;
    }

    int rc = 1;
    ds4_dist_kv_layout layout = {.vocab = vocab};
    bool layout_set = false;

    for (uint32_t shard = 0; shard < shard_count; shard++) {
        uint32_t layer_start = 0, layer_end = 0;
        const ds4_dist_route_entry *entry = NULL;
        dist_kv_route_shard(d, shard, &layer_start, &layer_end, &entry);
        shards[shard].fp = dist_tmpfile_or_err("distributed KV shard", err, errlen);
        if (!shards[shard].fp) goto cleanup;
        if (shard == 0) {
            if (ds4_session_save_layer_payload(owner, shards[shard].fp,
                                               layer_start, layer_end,
                                               err, errlen) != 0)
                goto cleanup;
            if (dist_measure_file(shards[shard].fp, &shards[shard].bytes,
                                  "distributed local KV shard", err, errlen) != 0)
                goto cleanup;
        } else {
            if (dist_save_remote_shard_to_file(d, entry, tokens, token_hash,
                                               shards[shard].fp,
                                               &shards[shard].bytes,
                                               err, errlen) != 0)
                goto cleanup;
        }
        if (shards[shard].bytes == 0) {
            if (errlen) snprintf(err, errlen, "distributed KV shard is empty");
            goto cleanup;
        }
        if (dist_kv_parse_layer_payload(d->state.engine,
                                        shards[shard].fp,
                                        shards[shard].bytes,
                                        layer_start,
                                        layer_end,
                                        &layout,
                                        &layout_set,
                                        n_comp,
                                        n_index_comp,
                                        &shards[shard],
                                        err,
                                        errlen) != 0)
            goto cleanup;
    }
    if (!layout_set || layout.token_count != token_count ||
        layout.n_layers != d->state.n_layers ||
        layout.vocab != vocab) {
        if (errlen) snprintf(err, errlen, "distributed KV shard metadata mismatch");
        goto cleanup;
    }

    if (dist_kv_write_session_header(fp, &layout, err, errlen) != 0)
        goto cleanup;
    for (uint32_t i = 0; i < token_count; i++) {
        if (dist_payload_write_u32(fp, (uint32_t)tokens->v[i], err, errlen) != 0)
            goto cleanup;
    }
    if (dist_payload_write_bytes(fp, logits,
                                 (uint64_t)vocab * sizeof(logits[0]),
                                 err, errlen) != 0)
        goto cleanup;
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_write_u32(fp, n_comp[il], err, errlen) != 0)
            goto cleanup;
    }
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_write_u32(fp, n_index_comp[il], err, errlen) != 0)
            goto cleanup;
    }
    for (uint32_t shard = 0; shard < shard_count; shard++) {
        if (dist_copy_file_range(shards[shard].fp,
                                 shards[shard].tensor_offset,
                                 shards[shard].tensor_bytes,
                                 fp,
                                 err,
                                 errlen) != 0)
            goto cleanup;
    }
    rc = 0;

cleanup:
    dist_kv_shards_close(shards, shard_count);
    free(shards);
    free(n_comp);
    free(n_index_comp);
    free(logits);
    return rc;
}

int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    if (!d || !owner || !fp) {
        if (errlen) snprintf(err, errlen, "invalid distributed payload load");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;
    if (dist_kv_route_validate(d, err, errlen) != 0) return 1;

    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (dist_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
            return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        if (errlen) snprintf(err, errlen, "unsupported DS4 KV payload version");
        return 1;
    }
    ds4_dist_kv_layout layout = {
        .ctx = h[2],
        .prefill_cap = h[3],
        .raw_cap = h[4],
        .raw_window = h[5],
        .comp_cap = h[6],
        .token_count = h[7],
        .n_layers = h[8],
        .head_dim = h[9],
        .indexer_head_dim = h[10],
        .vocab = h[11],
        .raw_live = h[12],
    };
    if (layout.n_layers != d->state.n_layers ||
        layout.ctx > (uint32_t)ds4_session_ctx(owner) ||
        layout.token_count >= (uint32_t)ds4_session_ctx(owner) ||
        layout.vocab != (uint32_t)ds4_engine_vocab_size(d->state.engine) ||
        !dist_kv_raw_live_valid(d->state.engine, &layout)) {
        if (errlen) snprintf(err, errlen, "DS4 KV payload does not match current distributed runtime");
        return 1;
    }

    int *tokens = layout.token_count ?
        malloc((size_t)layout.token_count * sizeof(tokens[0])) : NULL;
    float *logits = malloc((size_t)layout.vocab * sizeof(logits[0]));
    uint32_t *n_comp = calloc(layout.n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = calloc(layout.n_layers, sizeof(n_index_comp[0]));
    if ((layout.token_count && !tokens) || !logits || !n_comp || !n_index_comp) {
        free(tokens);
        free(logits);
        free(n_comp);
        free(n_index_comp);
        if (errlen) snprintf(err, errlen, "out of memory loading distributed KV payload");
        return 1;
    }
    for (uint32_t i = 0; i < layout.token_count; i++) {
        uint32_t tok = 0;
        if (dist_payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            free(tokens);
            free(logits);
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (tok > (uint32_t)INT_MAX ||
            tok >= (uint32_t)ds4_engine_vocab_size(d->state.engine)) {
            free(tokens);
            free(logits);
            free(n_comp);
            free(n_index_comp);
            if (errlen) snprintf(err, errlen, "distributed KV payload token is outside vocabulary");
            return 1;
        }
        tokens[i] = (int)tok;
    }
    int empty_token_sentinel = 0;
    int *tokens_arg = layout.token_count ? tokens : &empty_token_sentinel;
    const uint64_t token_hash = dist_token_hash_prefix(tokens_arg, layout.token_count);
    if (dist_payload_read_bytes(fp, logits,
                                (uint64_t)layout.vocab * sizeof(logits[0]),
                                &remaining, err, errlen) != 0) {
        free(tokens);
        free(logits);
        free(n_comp);
        free(n_index_comp);
        return 1;
    }
    int rc = 1;
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0)
            goto cleanup;
        if (n_comp[il] > layout.comp_cap) {
            if (errlen) snprintf(err, errlen, "DS4 KV payload has invalid compressed row count");
            goto cleanup;
        }
    }
    for (uint32_t il = 0; il < layout.n_layers; il++) {
        if (dist_payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0)
            goto cleanup;
        if (n_index_comp[il] > layout.comp_cap) {
            if (errlen) snprintf(err, errlen, "DS4 KV payload has invalid indexer row count");
            goto cleanup;
        }
    }

    const uint32_t shard_count = dist_kv_route_shard_count(d);
    for (uint32_t shard = 0; shard < shard_count; shard++) {
        uint32_t layer_start = 0, layer_end = 0;
        const ds4_dist_route_entry *entry = NULL;
        FILE *tmp = NULL;
        uint64_t shard_bytes = 0;
        dist_kv_route_shard(d, shard, &layer_start, &layer_end, &entry);
        if (dist_prepare_shard_from_session_payload(d,
                                                    fp,
                                                    &remaining,
                                                    &layout,
                                                    n_comp,
                                                    n_index_comp,
                                                    layer_start,
                                                    layer_end,
                                                    &tmp,
                                                    &shard_bytes,
                                                    err,
                                                    errlen) != 0)
            goto cleanup;
        if (shard == 0) {
            if (ds4_session_load_layer_payload(owner, tmp, shard_bytes,
                                               tokens_arg, layout.token_count,
                                               layer_start, layer_end,
                                               err, errlen) != 0) {
                fclose(tmp);
                goto cleanup;
            }
        } else {
            if (dist_load_remote_shard_from_payload(d, entry,
                                                    tokens_arg, layout.token_count,
                                                    token_hash,
                                                    tmp, shard_bytes,
                                                    err, errlen) != 0) {
                fclose(tmp);
                goto cleanup;
            }
        }
        fclose(tmp);
    }
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "DS4 KV payload has trailing bytes");
        goto cleanup;
    }
    if (ds4_session_set_logits(owner, logits, (int)layout.vocab) != 0) {
        if (errlen) snprintf(err, errlen, "failed to restore distributed logits");
        goto cleanup;
    }
    rc = 0;

cleanup:
    free(tokens);
    free(logits);
    free(n_comp);
    free(n_index_comp);
    return rc;
}

/* =========================================================================
 * Coordinator Session API
 * =========================================================================
 *
 * These functions are the distributed backend for the normal ds4_session API.
 * Program frontends should keep using ds4_session_sync/eval/save/load; ds4.c
 * selects these calls when the owning session has a coordinator attached.
 */

int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    (void)owner;
    if (!out || !engine || !opt) {
        if (errlen) snprintf(err, errlen, "missing distributed session parameters");
        return 1;
    }
    *out = NULL;
    if (opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "distributed session requires coordinator role");
        return 1;
    }
    if (dist_validate_options(opt, err, errlen) != 0) return 1;

    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, errlen);
    if (listen_fd < 0) return 1;

    ds4_dist_session *d = calloc(1, sizeof(*d));
    if (!d) {
        close(listen_fd);
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return 1;
    }

    d->listen_fd = listen_fd;
    d->state.engine = engine;
    d->state.model_id = (uint32_t)ds4_engine_model_id(engine);
    d->state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    d->state.local_start = opt->layers.start;
    d->state.local_end = dist_resolved_layer_end(opt, d->state.n_layers);
    d->state.ctx_size = ctx_size > 0 ? (uint32_t)ctx_size : 0u;
    d->state.local_has_output = opt->layers.has_output;
    d->state.local_can_output_head = ds4_engine_has_output_head(engine);
    d->state.replay_check = opt->replay_check;
    d->state.debug = opt->debug;
    d->state.use_control_for_work = true;
    d->state.prefill_chunk = opt->prefill_chunk;
    d->state.prefill_window = opt->prefill_window;
    d->state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    d->state.transport_policy = opt->transport;
    d->state.nhi_device = opt->nhi_device;
    pthread_mutex_init(&d->state.mu, NULL);
    d->session_id = dist_make_session_id(d);
    d->request_id = 1;
    /* KV snapshots use separate data connections and can run while pipelined
     * WORK results are outstanding.  Keep them out of the WORK request-id stream
     * so progress callbacks cannot perturb the reader's contiguous expectations. */
    d->snapshot_request_id = UINT64_C(1) << 63;

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&d->state,
                     "ds4: distributed coordinator API: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     d->state.model_id,
                     d->state.n_layers,
                     opt->layers.start,
                     local_end,
                     d->state.activation_bits);

    if (ds4_dist_spec_trust_all_enabled() &&
        ds4_engine_mtp_draft_tokens(engine) > 1) {
        fprintf(stderr,
                "ds4: WARNING: DS4_DIST_SPEC_TRUST_ALL=1 — target-vs-draft acceptance DISABLED (research mode); batched target-state advancement remains and emitted completion is not guaranteed to match greedy decode\n");
    }

    d->accept_ctx.state = &d->state;
    d->accept_ctx.listen_fd = listen_fd;
    if (pthread_create(&d->accept_tid, NULL, dist_coordinator_accept_main, &d->accept_ctx) != 0) {
        close(listen_fd);
        pthread_mutex_destroy(&d->state.mu);
        free(d);
        if (errlen) snprintf(err, errlen, "failed to start distributed coordinator accept loop");
        return 1;
    }
    pthread_detach(d->accept_tid);
    d->accept_started = true;
    *out = d;
    return 0;
}

void ds4_dist_session_free(ds4_dist_session *d) {
    if (!d) return;
    if (d->listen_fd >= 0) {
        shutdown(d->listen_fd, SHUT_RDWR);
        close(d->listen_fd);
        d->listen_fd = -1;
    }
    dist_route_plan_free(&d->plan);
    pthread_mutex_lock(&d->state.mu);
    d->state.shutting_down = true;
    for (ds4_dist_worker_entry *it = d->state.workers; it; it = it->next) {
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    pthread_mutex_unlock(&d->state.mu);
    /* Client threads are detached and remove their registry entries after the
     * socket closes. Keep this small coordinator object process-lifetime to
     * avoid racing those threads during application shutdown. */
}

int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return -1;
    }

    ds4_dist_route_plan probe = {0};
    if (!dist_coordinator_build_route_plan(&d->state, &probe, NULL, err, errlen)) {
        return 0;
    }
    dist_route_plan_free(&probe);
    if (errlen) err[0] = '\0';
    return 1;
}

int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !prompt || prompt->len <= 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed sync request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    if (checkpoint &&
        checkpoint->len >= 0 &&
        checkpoint->len <= prompt->len &&
        ds4_tokens_starts_with(prompt, checkpoint))
    {
        if (checkpoint->len == prompt->len) return 0;

        uint32_t chunk_cap = 0;
        if (dist_coordinator_prefill_chunk_cap(&d->state, owner, &chunk_cap, err, errlen) != 0) {
            return 1;
        }
        const uint32_t pos0 = (uint32_t)checkpoint->len;
        const uint32_t suffix = (uint32_t)prompt->len - pos0;
        if (dist_coordinator_can_pipeline_prefill(&d->state, &d->plan, owner, suffix, chunk_cap)) {
            int prefill_rc = dist_coordinator_prefill_prompt_pipelined(&d->state,
                                                                       owner,
                                                                       &d->plan,
                                                                       prompt,
                                                                       pos0,
                                                                       suffix,
                                                                       false,
                                                                       chunk_cap,
                                                                       d->session_id,
                                                                       &d->request_id,
                                                                       logits,
                                                                       err,
                                                                       errlen);
            if (prefill_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
            }
            return 0;
        }

        uint32_t pos = pos0;
        while (pos < (uint32_t)prompt->len) {
            const uint32_t remaining = (uint32_t)prompt->len - pos;
            const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
            int eval_rc = dist_coordinator_eval_span(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt->v + pos,
                                                     chunk,
                                                     pos,
                                                     d->session_id,
                                                     d->request_id++,
                                                     false,
                                                     0,
                                                     false,
                                                     NULL,
                                                     logits,
                                                     NULL,
                                                     NULL,
                                                     err,
                                                     errlen);
            if (eval_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             eval_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
                return 0;
            }
            pos += chunk;
            dist_report_prefill_progress(owner, pos, (uint32_t)prompt->len);
        }
        return 0;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     err,
                                                     errlen);
    if (prefill_rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            return 1;
        }
        d->plan_ready = true;
    }
    return 0;
}

int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !checkpoint || checkpoint->len < 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed decode request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    int rc = dist_coordinator_eval_span(&d->state,
                                        owner,
                                        &d->plan,
                                        &token,
                                        1,
                                        (uint32_t)checkpoint->len,
                                        d->session_id,
                                        d->request_id++,
                                        false,
                                        0,
                                        false,
                                        NULL,
                                        logits,
                                        NULL,
                                        NULL,
                                        err,
                                        errlen);
    if (rc != 0) {
        ds4_tokens transcript = {0};
        ds4_tokens_copy(&transcript, checkpoint);
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     &transcript,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            ds4_tokens_free(&transcript);
            return 1;
        }
        d->plan_ready = true;
        ds4_tokens_free(&transcript);
        rc = 0;
    }
    return rc;
}

/* =========================================================================
 * Distributed Legacy-MTP Speculative Cycle
 * ========================================================================= */

/* Re-decode an accepted speculative prefix one token at a time through the
 * ordinary distributed route (the same single-token eval the non-spec arm
 * uses), and make the accept/emit decision exactly from those single-token
 * logits: after decoding token i, the next token in `tokens` must equal
 * `dist_logits_argmax(last_logits)` re-computed on the single-token
 * output-head path.  The scan stops at the first divergence and returns the
 * number of verified (emitted) tokens, so no batched verify-row argmax ever
 * decides what is accepted.  On success *last_logits holds the exact logits
 * after the final replayed token.  It is also the recovery path: each span
 * here is the plain decode, so a caller can rewind to the committed prefix
 * and keep going instead of ending the request.  The worker's verify-span
 * frontier snapshot is consumed on the first token to rewind the remote
 * slice; the remaining tokens follow as ordinary single-token spans. */
static int dist_mtp_spec_replay_accept(
        ds4_dist_session *d,
        ds4_session *owner,
        const int *tokens,
        int count,
        uint32_t pos0,
        bool remote_rewind,
        float *last_logits,
        char *err,
        size_t errlen) {
    const int vocab = ds4_engine_vocab_size(d->state.engine);
    for (int i = 0; i < count; i++) {
        const bool rewind = remote_rewind && i == 0;
        const int rc = dist_coordinator_eval_span(
                &d->state,
                owner,
                &d->plan,
                &tokens[i],
                1,
                pos0 + (uint32_t)i,
                d->session_id,
                d->request_id++,
                false,
                rewind ? DS4_DIST_WORK_F_SPEC_ROLLBACK : 0u,
                false,
                NULL,
                last_logits,
                NULL,
                NULL,
                err,
                errlen);
        if (rc != 0) {
            if (getenv("DS4_DSPARK_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: dist spec exact replay token %d failed rc=%d: %s\n",
                        i, rc, err && err[0] ? err : "(no detail)");
            }
            return -1;
        }
        if (i + 1 < count) {
            const int next = dist_logits_argmax(last_logits, vocab);
            if (next != tokens[i + 1]) {
                if (getenv("DS4_DSPARK_SPEC_LOG")) {
                    fprintf(stderr,
                            "ds4: dist spec exact replay rejects token %d "
                            "(argmax=%d != %d)\n",
                            i + 1, next, tokens[i + 1]);
                }
                return i + 1;
            }
        }
    }
    return count;
}

/* The speculative WORK/RESULT extensions ride on negotiated v3 capability
 * bits and a specific topology: a coordinator must only emit them when the
 * route is a single worker that owns the output head (final logits) and that
 * worker selected DS4_DIST_V3_CAP_SPEC_DECODE_V1.  A multi-hop route cannot
 * carry spec flags (intermediate hops reject them), and an older peer
 * rejects unknown WORK flag bits, which would otherwise fail the request. */
static bool dist_route_plan_supports_spec(const ds4_dist_route_plan *plan) {
    if (!plan || plan->count != 1u) return false;
    const ds4_dist_route_entry *only = &plan->entry[0];
    if ((only->caps & DS4_DIST_V3_CAP_SPEC_DECODE_V1) == 0) return false;
    if ((only->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) return false;
    return true;
}

static bool dist_route_plan_supports_exact_required(
        const ds4_dist_route_plan *plan) {
    return dist_route_plan_supports_spec(plan) &&
        (plan->entry[0].caps & DS4_DIST_V3_CAP_SPEC_EXACT_V1) != 0;
}

void ds4_dist_session_print_spec_stats(const ds4_dist_session *d) {
    if (!d || d->spec_cycles == 0) return;
    /* Same gate as the single-node DSpark stats line: enabled only when
     * DS4_DSPARK_STATS is set to a non-empty value other than "0". */
    const char *env = getenv("DS4_DSPARK_STATS");
    if (!env || !env[0] || strcmp(env, "0") == 0) return;
    const double accept_rate =
        d->spec_proposed != 0
            ? (100.0 * (double)d->spec_accepted /
               (double)d->spec_proposed)
            : 0.0;
    fprintf(stderr,
            "ds4: dist spec stats cycles=%llu proposed=%llu "
            "accepted_draft=%llu accept_rate=%.2f%%\n",
            (unsigned long long)d->spec_cycles,
            (unsigned long long)d->spec_proposed,
            (unsigned long long)d->spec_accepted,
            accept_rate);
}

/* Recover from a failed speculative span without hard-ending the request.
 * Rebuild the committed prefix (all tokens before the cycle's incoming
 * first_token) so both machines are clean at `start`, then re-eval that
 * pending token through the ordinary per-token distributed decode.  The
 * plain eval owns its own rebuild fallback, so this path keeps the request
 * alive even when the speculative rebuild above it failed.  Logs a one-line
 * WHY at each stage so the next failing window pins the residual path. */
static int dist_mtp_spec_recover_after_failure(
        ds4_dist_session *d,
        ds4_session *owner,
        uint32_t start,
        int first_token,
        int span_rc,
        char *err,
        size_t errlen) {
    ds4_engine *e = d->state.engine;
    const int vocab = ds4_engine_vocab_size(e);
    fprintf(stderr,
            "ds4: dist spec span failed rc=%d why=%s; re-syncing from committed prefix\n",
            span_rc, err && err[0] ? err : "(no detail)");
    (void)ds4_session_dist_timeline_truncate(owner, start);
    float *prefix_logits = malloc((size_t)vocab * sizeof(float));
    if (prefix_logits) {
        if (dist_coordinator_rebuild_from_transcript(
                &d->state,
                owner,
                &d->plan,
                ds4_session_tokens(owner),
                d->session_id,
                &d->request_id,
                prefix_logits,
                &d->plan_generation,
                span_rc != DS4_DIST_RECV_REMOTE_ERROR,
                err,
                errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            fprintf(stderr,
                    "ds4: dist spec prefix rebuild failed why=%s; trying plain per-token decode\n",
                    err && err[0] ? err : "(no detail)");
        } else {
            d->plan_ready = true;
        }
        free(prefix_logits);
    } else {
        fprintf(stderr,
                "ds4: dist spec prefix rebuild oom; trying plain per-token decode\n");
    }
    if (ds4_session_eval(owner, first_token, err, errlen) == 0) {
        d->plan_ready = true;
        return 0;
    }
    fprintf(stderr,
            "ds4: dist spec plain per-token fallback failed why=%s\n",
            err && err[0] ? err : "(no detail)");
    return -1;
}

int ds4_dist_session_mtp_spec_cycle(
        ds4_dist_session *d,
        ds4_session *owner,
        int first_token,
        int max_tokens,
        int eos_token,
        int *accepted,
        int accepted_cap,
        char *err,
        size_t errlen) {
    if (!d || !owner || max_tokens <= 0 || accepted_cap <= 0 || !accepted) {
        return 0;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return -1;

    ds4_engine *e = d->state.engine;
    int draft_cap = e ? ds4_engine_mtp_draft_tokens(e) : 0;
    if (draft_cap >= 2 && !dist_route_plan_supports_spec(&d->plan)) {
        if (!d->spec_cap_warned) {
            d->spec_cap_warned = true;
            const char *why;
            if (d->plan.count != 1u) {
                why = "distributed speculation requires a single "
                      "output-owning route worker";
            } else if ((d->plan.entry[0].flags &
                        DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) {
                why = "the distributed route worker does not own the "
                      "output logits";
            } else {
                why = "a route worker did not negotiate "
                      "DS4_DIST_V3_CAP_SPEC_DECODE_V1";
            }
            fprintf(stderr,
                    "ds4: distributed speculative decode disabled: %s; "
                    "falling back to per-token decode\n",
                    why);
        }
        draft_cap = 0;
    }
    if (draft_cap < 2) {
        if (ds4_session_eval(owner, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (draft_cap > 16) draft_cap = 16;
    d->spec_cycles++;
    const int vocab = ds4_engine_vocab_size(e);
    const ds4_tokens *timeline = ds4_session_tokens(owner);
    if (!timeline || timeline->len < 0) {
        if (errlen) snprintf(err, errlen, "distributed session has no token timeline");
        return -1;
    }
    const uint32_t start = (uint32_t)timeline->len;

    /* Failure recovery is the dist_mtp_spec_recover_after_failure helper:
     * rewind to the committed prefix, rebuild both machines, then re-eval
     * the pending token through the plain per-token distributed decode. */
    static const char fail_msg[] = "distributed speculative decode failed";

    int min_verify = 3;
    {
        const char *env = getenv("DS4_DIST_SPEC_MIN_DRAFT");
        if (env && env[0]) {
            const int v = atoi(env);
            if (v >= 2 && v <= 16) min_verify = v;
        }
    }

    /* Trust-all research mode: emit every drafted token with no verify span
     * or acceptance check (measurement only, never the default).  The base
     * decode still runs so the target commits the pending token and feeds the
     * drafter; the returned drafts are then trusted and emitted as-is. */
    if (ds4_dist_spec_trust_all_enabled()) {
        d->spec_pending_count = 0;
        int n_accept = 0;
        accepted[n_accept++] = first_token;
        float *logits0 = malloc((size_t)vocab * sizeof(float));
        if (!logits0) {
            if (errlen) snprintf(err, errlen, "out of memory allocating speculative logits");
            return -1;
        }
        int drafts[16];
        int n_drafts = 0;
        const double cycle_t0 = dist_now_sec();
        const int rc = dist_coordinator_eval_span(&d->state,
                                                  owner,
                                                  &d->plan,
                                                  &first_token,
                                                  1,
                                                  start,
                                                  d->session_id,
                                                  d->request_id++,
                                                  false,
                                                  DS4_DIST_WORK_F_OUTPUT_DRAFTS,
                                                  false,
                                                  NULL,
                                                  logits0,
                                                  drafts,
                                                  &n_drafts,
                                                  err,
                                                  errlen);
        const double base_ms = (dist_now_sec() - cycle_t0) * 1000.0;
        if (rc != 0) {
            free(logits0);
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    rc, err, errlen) != 0) {
                fprintf(stderr,
                        "ds4: dist spec unrecoverable after trust-all base decode why=%s\n",
                        err && err[0] ? err : "(no detail)");
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        ds4_session_set_logits(owner, logits0, vocab);
        int draft_n = draft_cap;
        if (first_token != eos_token && max_tokens > 1 &&
            n_accept < accepted_cap && n_drafts > 0) {
            if (draft_n > max_tokens - n_accept) draft_n = max_tokens - n_accept;
            if (draft_n > accepted_cap - n_accept) draft_n = accepted_cap - n_accept;
            const int room = ds4_session_ctx(owner) - ds4_session_tokens(owner)->len;
            if (draft_n > room - 1) draft_n = room - 1;
            if (n_drafts < draft_n) draft_n = n_drafts;
        } else {
            draft_n = 0;
        }
        if (draft_n < 0) draft_n = 0;
        for (int i = 0; i < draft_n; i++) {
            if (drafts[i] == eos_token) {
                draft_n = i + 1;
                break;
            }
        }
        double trusted_ms = 0.0;
        if (draft_n > 0) {
            /* Trust means skipping target-vs-draft acceptance, not leaving
             * the target KV/timeline behind the emitted stream. Advance all
             * trusted drafts in one ordinary batch span, retain that target
             * state, and sample the next boundary from its final logits. */
            const uint64_t rows_bytes =
                (uint64_t)draft_n * (uint64_t)vocab * sizeof(float);
            float *rows = rows_bytes <= SIZE_MAX ? malloc((size_t)rows_bytes) : NULL;
            if (!rows) {
                free(logits0);
                if (errlen) snprintf(err, errlen,
                                     "out of memory allocating trust-all rows");
                return -1;
            }
            const double trusted_t0 = dist_now_sec();
            const int trc = dist_coordinator_eval_span(
                &d->state, owner, &d->plan, drafts, (uint32_t)draft_n,
                start + 1u, d->session_id, d->request_id++, false,
                DS4_DIST_WORK_F_SPEC_VERIFY |
                    DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS,
                false, NULL, rows, NULL, NULL, err, errlen);
            trusted_ms = (dist_now_sec() - trusted_t0) * 1000.0;
            if (trc != 0) {
                free(rows);
                free(logits0);
                if (dist_mtp_spec_recover_after_failure(
                        d, owner, start, first_token, trc,
                        err, errlen) != 0) {
                    if (errlen) snprintf(err, errlen, "%s", fail_msg);
                    return -1;
                }
                return n_accept;
            }
            ds4_session_set_logits(
                owner, rows + (uint64_t)(draft_n - 1) * vocab, vocab);
            free(rows);
            d->spec_proposed += (uint64_t)draft_n;
            d->spec_accepted += (uint64_t)draft_n;
            for (int i = 0; i < draft_n && n_accept < accepted_cap; i++)
                accepted[n_accept++] = drafts[i];
        }
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: dist mtp spec trust_all token=%d drafts=%d emitted=%d base=%.1fms trusted_span=%.1fms\n",
                    first_token,
                    draft_n,
                    n_accept - 1,
                    base_ms,
                    trusted_ms);
        }
        free(logits0);
        return n_accept;
    }

    /* 0. Fused span: when the previous cycle fully accepted its block, the
     * worker already proposed continuation drafts for exactly this token at
     * exactly this position.  Verify them together with the base token in a
     * single span (rows = [first_token, drafts...]), skipping the separate
     * base decode and its round trip.  Any invalidation (different token,
     * moved timeline, too few drafts) falls through to the classic cycle. */
    int fused_pending[16];
    int fused_k = 0;
    if (d->spec_pending_count > 0 &&
        d->spec_pending_token == first_token &&
        d->spec_pending_pos == start) {
        fused_k = d->spec_pending_count;
        memcpy(fused_pending, d->spec_pending_drafts,
               (size_t)fused_k * sizeof(fused_pending[0]));
    }
    d->spec_pending_count = 0;
    if (fused_k > draft_cap) fused_k = draft_cap;
    /* The verify span rows must fit the draft_tokens scratch and the DSpark
     * proposer's block size; clamp to the same bound as the classic cycle. */
    if (fused_k > 5) fused_k = 5;
    if (fused_k > max_tokens - 1) fused_k = max_tokens - 1;
    if (fused_k > accepted_cap - 1) fused_k = accepted_cap - 1;
    {
        const int room = ds4_session_ctx(owner) - (int)start;
        if (fused_k > room - 2) fused_k = room - 2;
    }
    if (first_token == eos_token) fused_k = 0;
    if (fused_k >= min_verify) {
        d->spec_proposed += (uint64_t)fused_k;
        ds4_dist_mtp_frontier fused_frontier;
        if (!ds4_session_dist_frontier_snapshot(owner, &fused_frontier)) {
            if (errlen) snprintf(err, errlen, "coordinator speculative frontier snapshot failed");
            return -1;
        }
        int span_tokens[17];
        span_tokens[0] = first_token;
        memcpy(span_tokens + 1, fused_pending,
               (size_t)fused_k * sizeof(span_tokens[0]));
        const uint32_t rows_n = (uint32_t)fused_k + 1u;
        const bool exact_span =
            dist_route_plan_supports_exact_required(&d->plan) &&
            ds4_session_dist_spec_exact_verify(owner, rows_n);
        float *rows = malloc((size_t)rows_n * (size_t)vocab * sizeof(float));
        if (!rows) {
            if (errlen) snprintf(err, errlen, "out of memory allocating speculative verify rows");
            return -1;
        }
        int cont[16];
        int n_cont = 0;
        const double fused_t0 = dist_now_sec();
        const int frc = dist_coordinator_eval_span(&d->state,
                                                   owner,
                                                   &d->plan,
                                                   span_tokens,
                                                   rows_n,
                                                   start,
                                                   d->session_id,
                                                   d->request_id++,
                                                   false,
                                                   DS4_DIST_WORK_F_SPEC_VERIFY |
                                                       DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS |
                                                       DS4_DIST_WORK_F_OUTPUT_DRAFTS |
                                                       (exact_span
                                                           ? DS4_DIST_WORK_F_SPEC_EXACT_REQUIRED
                                                           : 0u),
                                                   false,
                                                   NULL,
                                                   rows,
                                                   cont,
                                                   &n_cont,
                                                   err,
                                                   errlen);
        const double fused_ms = (dist_now_sec() - fused_t0) * 1000.0;
        if (frc != 0) {
            free(rows);
            int n_accept = 0;
            accepted[n_accept++] = first_token;
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    frc, err, errlen) != 0) {
                fprintf(stderr,
                        "ds4: dist spec unrecoverable after fused verify why=%s\n",
                        err && err[0] ? err : "(no detail)");
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        int accept_n = 0;
        for (int i = 0; i < fused_k; i++) {
            if (dist_logits_argmax(rows + (uint64_t)i * vocab, vocab) !=
                span_tokens[i + 1]) {
                break;
            }
            accept_n++;
        }
        /* The exact per-row span already decoded every token through the
         * ordinary single-token layer-slice path with the KV advanced row by
         * row, so rows[] equals what the replay arm would have produced.  On
         * a full accept we can keep the span's commits and use the last row
         * directly, skipping the serial replay round trips.  A partial accept
         * (or a non-exact backend) still rewinds the rejected suffix and
         * re-commits the accepted prefix through the exact replay arm. */
        if (exact_span && accept_n == fused_k) {
            const float *last_row = rows + (uint64_t)fused_k * vocab;
            d->spec_accepted += (uint64_t)fused_k;
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: dist mtp spec fused exact drafted=%d accepted=%d cont=%d span=%.1fms\n",
                        fused_k, fused_k, n_cont, fused_ms);
            }
            ds4_session_set_logits(owner, last_row, vocab);
            if (n_cont > 0 && span_tokens[fused_k] != eos_token) {
                int cnt = n_cont > 16 ? 16 : n_cont;
                memcpy(d->spec_pending_drafts, cont,
                       (size_t)cnt * sizeof(cont[0]));
                d->spec_pending_count = cnt;
                d->spec_pending_token = dist_logits_argmax(last_row, vocab);
                d->spec_pending_pos = start + rows_n;
            }
            free(rows);
            int n_accept = 0;
            accepted[n_accept++] = first_token;
            for (int i = 0; i < fused_k && n_accept < accepted_cap; i++) {
                accepted[n_accept++] = span_tokens[i + 1];
                if (span_tokens[i + 1] == eos_token) break;
            }
            return n_accept;
        }
        /* Exact commit: re-decode the base token plus the accepted drafts one
         * token at a time through the ordinary distributed route so the
         * committed tokens and the next sample match plain per-token decode. */
        if (!ds4_session_dist_frontier_restore(owner, &fused_frontier)) {
            free(rows);
            if (errlen) snprintf(err, errlen, "coordinator speculative frontier restore failed");
            int n_accept = 0;
            accepted[n_accept++] = first_token;
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    -1, err, errlen) != 0) {
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        if (!ds4_session_dist_timeline_truncate(owner, start)) {
            free(rows);
            if (errlen) snprintf(err, errlen, "coordinator speculative timeline rewind failed");
            return -1;
        }
        float *replay_logits = malloc((size_t)vocab * sizeof(float));
        if (!replay_logits) {
            free(rows);
            if (errlen) snprintf(err, errlen, "out of memory allocating speculative replay logits");
            return -1;
        }
        const int replay_count = 1 + accept_n;
        const int repl = dist_mtp_spec_replay_accept(d, owner, span_tokens,
                                                     replay_count, start, true,
                                                     replay_logits, err, errlen);
        free(rows);
        if (repl < 0) {
            free(replay_logits);
            int n_accept = 0;
            accepted[n_accept++] = first_token;
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    repl, err, errlen) != 0) {
                fprintf(stderr,
                        "ds4: dist spec unrecoverable after fused exact replay why=%s\n",
                        err && err[0] ? err : "(no detail)");
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        const int exact_drafts = repl - 1; /* span_tokens[0] is the base token. */
        d->spec_accepted += (uint64_t)exact_drafts;
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: dist mtp spec fused drafted=%d accepted=%d cont=%d span=%.1fms\n",
                    fused_k, exact_drafts, n_cont, fused_ms);
        }
        ds4_session_set_logits(owner, replay_logits, vocab);
        if (exact_drafts == fused_k && n_cont > 0 &&
            span_tokens[fused_k] != eos_token) {
            int cnt = n_cont > 16 ? 16 : n_cont;
            memcpy(d->spec_pending_drafts, cont,
                   (size_t)cnt * sizeof(cont[0]));
            d->spec_pending_count = cnt;
            d->spec_pending_token = dist_logits_argmax(replay_logits, vocab);
            d->spec_pending_pos = start + rows_n;
        }
        free(replay_logits);
        int n_accept = 0;
        accepted[n_accept++] = first_token;
        for (int i = 0; i < exact_drafts && n_accept < accepted_cap; i++) {
            accepted[n_accept++] = span_tokens[i + 1];
            if (span_tokens[i + 1] == eos_token) break;
        }
        return n_accept;
    }

    /* 1. Base decode of first_token; the worker returns its logits plus the
     * MTP head's draft tokens for the following positions. */
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    int drafts[16];
    int n_drafts = 0;
    float *logits0 = malloc((size_t)vocab * sizeof(float));
    if (!logits0) {
        if (errlen) snprintf(err, errlen, "out of memory allocating speculative logits");
        return -1;
    }
    double cycle_t0 = dist_now_sec();
    int rc = dist_coordinator_eval_span(&d->state,
                                        owner,
                                        &d->plan,
                                        &first_token,
                                        1,
                                        start,
                                        d->session_id,
                                        d->request_id++,
                                        false,
                                        DS4_DIST_WORK_F_OUTPUT_DRAFTS,
                                        false,
                                        NULL,
                                        logits0,
                                        drafts,
                                        &n_drafts,
                                        err,
                                        errlen);
    const double base_ms = (dist_now_sec() - cycle_t0) * 1000.0;
    if (rc != 0) {
        free(logits0);
        if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                rc, err, errlen) != 0) {
            fprintf(stderr,
                    "ds4: dist spec unrecoverable after base decode why=%s\n",
                    err && err[0] ? err : "(no detail)");
            if (errlen) snprintf(err, errlen, "%s", fail_msg);
            return -1;
        }
        return n_accept;
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr,
                "ds4: dist mtp spec base token=%d drafts=%d draft0=%d\n",
                first_token,
                n_drafts,
                n_drafts > 0 ? drafts[0] : -1);
    }
    if (first_token == eos_token || max_tokens == 1 ||
        n_accept >= accepted_cap || n_drafts < 1) {
        ds4_session_set_logits(owner, logits0, vocab);
        free(logits0);
        return n_accept;
    }
    /* The first draft must agree with the base model's own greedy choice,
     * otherwise the speculation is already wrong before any verify work. */
    if (drafts[0] != dist_logits_argmax(logits0, vocab)) {
        ds4_session_set_logits(owner, logits0, vocab);
        free(logits0);
        return n_accept;
    }
    int draft_n = draft_cap;
    if (draft_n > max_tokens - n_accept) draft_n = max_tokens - n_accept;
    if (draft_n > accepted_cap - n_accept) draft_n = accepted_cap - n_accept;
    int room = ds4_session_ctx(owner) - ds4_session_tokens(owner)->len;
    if (draft_n > room - 1) draft_n = room - 1;
    if (n_drafts < draft_n) draft_n = n_drafts;
    /* A verify span costs two serialized slice evals; shallow blocks cannot
     * return that even when fully accepted.  Below the threshold, emit the
     * base token and keep its logits live (the worker also rejects
     * single-token verify spans). */
    if (draft_n < min_verify || draft_n < 2) {
        ds4_session_set_logits(owner, logits0, vocab);
        free(logits0);
        return n_accept;
    }
    free(logits0);
    d->spec_proposed += (uint64_t)draft_n;

    /* 2. Speculative verify span: decode the draft rows in one multi-token
     * span per machine.  Both sides snapshot their frontier first so a
     * partial accept can rewind and re-eval the accepted prefix. */
    ds4_dist_mtp_frontier frontier;
    if (!ds4_session_dist_frontier_snapshot(owner, &frontier)) {
        if (errlen) snprintf(err, errlen, "coordinator speculative frontier snapshot failed");
        return -1;
    }
    const uint64_t rows_bytes = (uint64_t)draft_n * (uint64_t)vocab * sizeof(float);
    float *rows = malloc((size_t)rows_bytes);
    if (!rows) {
        if (errlen) snprintf(err, errlen, "out of memory allocating speculative verify rows");
        return -1;
    }
    int verify_top0 = -1;
    /* The exact two-row verifier skips the batch path but also skips the
     * DSpark capture, which would starve the next propose; keep it for the
     * legacy MTP head only. */
    const bool decode2_verify = draft_n == 2 && ds4_engine_has_mtp(e);
    /* DSpark verify spans also return continuation drafts proposed from the
     * last row, so a full accept can fuse the next cycle into one span. */
    const bool want_cont = !decode2_verify && !ds4_engine_has_mtp(e);
    const bool exact_span =
        !decode2_verify &&
        dist_route_plan_supports_exact_required(&d->plan) &&
        ds4_session_dist_spec_exact_verify(owner, (uint32_t)draft_n);
    int cont[16];
    int n_cont = 0;
    const double verify_t0 = dist_now_sec();
    rc = dist_coordinator_eval_span(&d->state,
                                    owner,
                                    &d->plan,
                                    drafts,
                                    (uint32_t)draft_n,
                                    start + 1u,
                                    d->session_id,
                                    d->request_id++,
                                    false,
                                    DS4_DIST_WORK_F_SPEC_VERIFY |
                                        DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS |
                                        (want_cont ? DS4_DIST_WORK_F_OUTPUT_DRAFTS : 0u) |
                                        (exact_span
                                            ? DS4_DIST_WORK_F_SPEC_EXACT_REQUIRED
                                            : 0u),
                                    decode2_verify,
                                    decode2_verify ? &verify_top0 : NULL,
                                    rows,
                                    want_cont ? cont : NULL,
                                    want_cont ? &n_cont : NULL,
                                    err,
                                    errlen);
    const double verify_ms = (dist_now_sec() - verify_t0) * 1000.0;
    if (rc != 0) {
        free(rows);
        if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                rc, err, errlen) != 0) {
            fprintf(stderr,
                    "ds4: dist spec unrecoverable after verify span why=%s\n",
                    err && err[0] ? err : "(no detail)");
            if (errlen) snprintf(err, errlen, "%s", fail_msg);
            return -1;
        }
        return n_accept;
    }

    /* 3. Greedy acceptance scan.  rows[i] holds the target logits after
     * decoding drafts[i]; drafts[i+1] must match its argmax to accept.  The
     * two-row verifier reports the exact row-0 argmax separately. */
    int accept_n = draft_n;
    if (decode2_verify) {
        if (verify_top0 != drafts[1]) accept_n = 1;
    } else {
        for (int i = 0; i < draft_n - 1; i++) {
            if (dist_logits_argmax(rows + (uint64_t)i * vocab, vocab) !=
                drafts[i + 1]) {
                accept_n = i + 1;
                break;
            }
        }
    }
    /* On the exact path accept_n above is authoritative (rows[] equals
     * sequential decode); on the batched path it is only a replay-length
     * bound and the final accept/emit decision is re-derived inside
     * dist_mtp_spec_replay_accept from single-token output-head logits, so
     * the batched verify rows never decide what is emitted. */
    if (accept_n > 0) {
        /* The exact per-row span already decoded every draft through the
         * ordinary single-token layer-slice path with the KV advanced row by
         * row, so rows[] equals what the replay arm would have produced.  A
         * full accept can therefore keep the span's commits and use the last
         * row directly, skipping the serial replay round trips.  Partial
         * accepts (or a non-exact backend) still rewind the rejected suffix
         * and re-commit the accepted prefix through the exact replay arm. */
        if (exact_span && accept_n == draft_n) {
            const float *last_row = rows + (uint64_t)(draft_n - 1) * vocab;
            d->spec_accepted += (uint64_t)draft_n;
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: dist mtp spec verify exact drafted=%d accepted=%d decode2=0 base=%.1fms verify=%.1fms\n",
                        draft_n, draft_n, base_ms, verify_ms);
            }
            ds4_session_set_logits(owner, last_row, vocab);
            if (want_cont && n_cont > 0 && drafts[draft_n - 1] != eos_token) {
                int cnt = n_cont > 16 ? 16 : n_cont;
                memcpy(d->spec_pending_drafts, cont,
                       (size_t)cnt * sizeof(cont[0]));
                d->spec_pending_count = cnt;
                d->spec_pending_token = dist_logits_argmax(last_row, vocab);
                d->spec_pending_pos = start + 1u + (uint32_t)draft_n;
            }
            free(rows);
            for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                accepted[n_accept++] = drafts[i];
                if (drafts[i] == eos_token) break;
            }
            return n_accept;
        }
        /* Exact commit: re-decode the accepted drafts one token at a time
         * from the committed prefix (first_token was already committed by
         * the base decode) so the emitted tokens and the next sample match
         * plain per-token decode.  accept_n is always >= 1 here because
         * drafts[0] was pre-verified against the base decode's argmax. */
        if (!ds4_session_dist_frontier_restore(owner, &frontier)) {
            free(rows);
            if (errlen) snprintf(err, errlen, "coordinator speculative frontier restore failed");
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    -1, err, errlen) != 0) {
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        if (!ds4_session_dist_timeline_truncate(owner, start + 1u)) {
            free(rows);
            if (errlen) snprintf(err, errlen, "coordinator speculative timeline rewind failed");
            return -1;
        }
        float *replay_logits = malloc((size_t)vocab * sizeof(float));
        if (!replay_logits) {
            free(rows);
            if (errlen) snprintf(err, errlen, "out of memory allocating speculative replay logits");
            return -1;
        }
        const int repl = dist_mtp_spec_replay_accept(d, owner, drafts, accept_n,
                                                     start + 1u, true,
                                                     replay_logits, err, errlen);
        free(rows);
        if (repl < 0) {
            free(replay_logits);
            if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                                    repl, err, errlen) != 0) {
                fprintf(stderr,
                        "ds4: dist spec unrecoverable after classic exact replay why=%s\n",
                        err && err[0] ? err : "(no detail)");
                if (errlen) snprintf(err, errlen, "%s", fail_msg);
                return -1;
            }
            return n_accept;
        }
        ds4_session_set_logits(owner, replay_logits, vocab);
        d->spec_accepted += (uint64_t)repl;
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: dist mtp spec verify drafted=%d accepted=%d decode2=%d base=%.1fms verify=%.1fms\n",
                    draft_n,
                    repl,
                    decode2_verify ? 1 : 0,
                    base_ms,
                    verify_ms);
        }
        if (repl == draft_n && want_cont && n_cont > 0 &&
            drafts[draft_n - 1] != eos_token) {
            /* Full accept: arm the next cycle's fused span from the exact
             * final logits. */
            int cnt = n_cont > 16 ? 16 : n_cont;
            memcpy(d->spec_pending_drafts, cont,
                   (size_t)cnt * sizeof(cont[0]));
            d->spec_pending_count = cnt;
            d->spec_pending_token = dist_logits_argmax(replay_logits, vocab);
            d->spec_pending_pos = start + 1u + (uint32_t)draft_n;
        }
        free(replay_logits);
        for (int i = 0; i < repl && n_accept < accepted_cap; i++) {
            accepted[n_accept++] = drafts[i];
            if (drafts[i] == eos_token) break;
        }
        return n_accept;
    }
    /* Defensive: accept_n should always be >= 1 because drafts[0] was
     * pre-verified against the base decode; if it is ever 0, recover the
     * base token exactly instead of returning with stale logits. */
    free(rows);
    if (dist_mtp_spec_recover_after_failure(d, owner, start, first_token,
                                            0, err, errlen) != 0) {
        fprintf(stderr,
                "ds4: dist spec unrecoverable after classic empty accept why=%s\n",
                err && err[0] ? err : "(no detail)");
        if (errlen) snprintf(err, errlen, "%s", fail_msg);
        return -1;
    }
    return n_accept;
}

/* =========================================================================
 * Standalone Coordinator Entrypoint
 * ========================================================================= */

static int dist_run_coordinator(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    char err[256];
    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    state.local_start = opt->layers.start;
    state.local_end = dist_resolved_layer_end(opt, state.n_layers);
    state.ctx_size = gen && gen->ctx_size > 0 ? (uint32_t)gen->ctx_size : 0u;
    state.local_has_output = opt->layers.has_output;
    state.local_can_output_head = ds4_engine_has_output_head(engine);
    state.replay_check = opt->replay_check;
    state.debug = opt->debug;
    state.use_control_for_work = gen && gen->prompt;
    state.prefill_chunk = opt->prefill_chunk;
    state.prefill_window = opt->prefill_window;
    state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    state.transport_policy = opt->transport;
    state.nhi_device = opt->nhi_device;
    pthread_mutex_init(&state.mu, NULL);

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&state,
                     "ds4: distributed coordinator: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     state.model_id,
                     state.n_layers,
                     opt->layers.start,
                     local_end,
                     state.activation_bits);

    ds4_dist_accept_ctx accept_ctx = {
        .state = &state,
        .listen_fd = listen_fd,
    };
    if (!gen || !gen->prompt) {
        dist_coordinator_accept_main(&accept_ctx);
        return 0;
    }

    pthread_t accept_tid;
    if (pthread_create(&accept_tid, NULL, dist_coordinator_accept_main, &accept_ctx) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: pthread_create failed for accept loop\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(accept_tid);

    return dist_run_coordinator_generation(&state, gen);
}

/* =========================================================================
 * Worker Control Loop And Result Frames
 * ========================================================================= */

static int dist_worker_read_loop(
        ds4_dist_worker_state *state,
        int fd,
        ds4_transport *transport,
        uint32_t selected_caps) {
    ds4_dist_worker_upstream upstream;
    if (dist_worker_upstream_init(&upstream, state, fd, transport,
                                  selected_caps) != 0)
        return 1;
    int loop_rc = 0;

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            rc = dist_worker_handle_work(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_SAVE_REQ) {
            rc = dist_worker_handle_snapshot_save(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN) {
            rc = dist_worker_handle_snapshot_load(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static int dist_send_work_result_prepared(
        int fd,
        ds4_transport *transport,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes,
        bool allow_oob,
        const ds4_transport_bulk_desc *prepared_desc,
        ds4_transport_lease *tx_lease) {
    if (payload_bytes != 0 && !payload) return -1;
    if (telemetry_count != 0 && !telemetry) return -1;
    uint32_t wire_payload_bytes = payload_bytes;
    uint64_t hidden_values = 0;
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        payload_bits = dist_activation_bits_or_default(payload_bits);
        if (!dist_activation_bits_valid(payload_bits) ||
            (payload_bytes % (uint32_t)sizeof(float)) != 0)
            return -1;
        hidden_values = payload_bytes / (uint32_t)sizeof(float);
        if (!dist_activation_wire_bytes(payload_bits, hidden_values, &wire_payload_bytes))
            return -1;
    } else if (status == 0) {
        /* ACK and every logits-family result use scalar 32-bit wire
         * semantics. Keep this aligned with
         * ds4_dist_v3_result_payload_validate(); zero is reserved for remote
         * error payloads. */
        payload_bits = 32u;
    } else {
        payload_bits = 0;
    }
    const uint64_t telemetry_bytes64 =
        (uint64_t)telemetry_count * sizeof(ds4_dist_telemetry_fixed);
    if (telemetry_bytes64 > UINT32_MAX) return -1;
    const uint32_t telemetry_bytes = (uint32_t)telemetry_bytes64;

    ds4_dist_result_fixed r;
    dist_u64_to_halves(request_id, &r.request_hi, &r.request_lo);
    dist_u64_to_halves(status == 0 ? result_hash : 0,
                       &r.result_hash_hi,
                       &r.result_hash_lo);
    r.status = status;
    r.result_kind = result_kind;
    r.telemetry_count = telemetry_count;
    r.telemetry_bytes = telemetry_bytes;
    r.payload_bytes = wire_payload_bytes;
    r.payload_bits = payload_bits;

    if (dist_transport_uses_v3(transport)) {
        const bool is_bulk = status == 0 && wire_payload_bytes != 0 &&
            (result_kind == DS4_DIST_RESULT_HIDDEN_STATE ||
             result_kind == DS4_DIST_RESULT_LOGITS);
        const void *bulk_wire = NULL;
        void *bulk_owned = NULL;
        uint32_t bulk_bytes = 0;
        ds4_transport_bulk_desc desc;
        bool local_desc_prepared = false;
        if (is_bulk) {
            if (result_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
                if (dist_pack_activation_payload(payload, hidden_values,
                                                 payload_bits, &bulk_wire,
                                                 &bulk_owned,
                                                 &bulk_bytes) != 0 ||
                    bulk_bytes != wire_payload_bytes) {
                    free(bulk_owned);
                    return -1;
                }
            } else {
                bulk_wire = payload;
                bulk_bytes = payload_bytes;
            }
            const uint32_t bulk_kind =
                result_kind == DS4_DIST_RESULT_HIDDEN_STATE
                    ? DS4_TRANSPORT_BULK_RESULT_HIDDEN
                    : DS4_TRANSPORT_BULK_RESULT_LOGITS;
            if (prepared_desc) {
                desc = *prepared_desc;
                const int desc_rc = desc.mode == DS4_TRANSPORT_BULK_NHI_OOB
                    ? ds4_transport_validate_oob_desc(
                          transport, &desc, bulk_kind, 0, request_id,
                          bulk_bytes, payload_bits, NULL, 0)
                    : ds4_transport_validate_inline_desc(
                          &desc, bulk_kind, 0, request_id,
                          bulk_bytes, payload_bits, NULL, 0);
                if (desc_rc != 0) {
                    free(bulk_owned);
                    return -1;
                }
            } else if (ds4_transport_prepare_bulk(
                           transport, bulk_kind, 0, request_id,
                           bulk_bytes, payload_bits, &desc,
                           NULL, 0) != 0) {
                free(bulk_owned);
                return -1;
            } else {
                local_desc_prepared = true;
            }
            if (!allow_oob && desc.mode == DS4_TRANSPORT_BULK_NHI_OOB) {
                desc.mode = DS4_TRANSPORT_BULK_TCP_INLINE;
                desc.frame_count = 0;
            }
        } else {
            if (prepared_desc || tx_lease) return -1;
            dist_bulk_none_desc(transport, 0, request_id, &desc);
        }

        if (tx_lease &&
            (!prepared_desc || desc.mode != DS4_TRANSPORT_BULK_NHI_OOB ||
             ds4_transport_lease_host_ptr(tx_lease) != bulk_wire ||
             ds4_transport_lease_bytes(tx_lease) != bulk_bytes)) {
            free(bulk_owned);
            errno = EINVAL;
            return -1;
        }

        const uint32_t control_payload_bytes = is_bulk ? 0u : wire_payload_bytes;
        uint32_t control_bytes = 0;
        uint32_t frame_bytes = 0;
        if (ds4_dist_v3_result_frame_sizes(telemetry_bytes,
                                           control_payload_bytes,
                                           bulk_bytes,
                                           desc.mode,
                                           &control_bytes,
                                           &frame_bytes) != 0) {
            if (local_desc_prepared) shutdown(fd, SHUT_RDWR);
            free(bulk_owned);
            return -1;
        }
        (void)control_bytes;
        ds4_transport_bulk_desc_wire desc_wire;
        if (ds4_transport_bulk_desc_encode(&desc, &desc_wire) != 0) {
            if (local_desc_prepared) shutdown(fd, SHUT_RDWR);
            free(bulk_owned);
            return -1;
        }
        ds4_dist_result_fixed result_wire = r;
        dist_result_to_wire(&result_wire);
        int rc = -1;
        bool lease_control_mark_attempted = false;
        if (dist_write_frame_header(fd, DS4_DIST_MSG_RESULT, frame_bytes) != 0 ||
            dist_write_full(fd, &result_wire, sizeof(result_wire)) != 0 ||
            dist_write_full(fd, &desc_wire, sizeof(desc_wire)) != 0)
            goto v3_result_done;
        for (uint32_t i = 0; i < telemetry_count; i++) {
            ds4_dist_telemetry_fixed tw = telemetry[i];
            dist_telemetry_to_wire(&tw);
            if (dist_write_full(fd, &tw, sizeof(tw)) != 0)
                goto v3_result_done;
        }
        if (control_payload_bytes != 0 &&
            dist_write_full(fd, payload, control_payload_bytes) != 0)
            goto v3_result_done;
        if (bulk_bytes != 0) {
            if (tx_lease) {
                lease_control_mark_attempted = true;
                if (ds4_transport_tx_lease_mark_control_sent(tx_lease) != 0 ||
                    dist_mapped_lease_commit_after_gpu_quiesce(tx_lease) != 0)
                    goto v3_result_done;
            } else if (ds4_transport_send_bulk_desc(
                           transport, &desc, bulk_wire, bulk_bytes) != 0) {
                goto v3_result_done;
            }
        }
        rc = 1;
v3_result_done:
        if (rc < 0) {
            const int saved = errno ? errno : EIO;
            /* A control write was attempted. Make a mapped lease irreversible
             * before caller cleanup so its prepared sequence cannot roll back,
             * even if the peer saw only a prefix of the frame. */
            if (tx_lease && !lease_control_mark_attempted)
                (void)ds4_transport_tx_lease_mark_control_sent(tx_lease);
            shutdown(fd, SHUT_RDWR);
            errno = saved;
        }
        free(bulk_owned);
        return rc;
    }

    const uint64_t frame_bytes = sizeof(ds4_dist_result_fixed) +
                                 telemetry_bytes64 +
                                 (uint64_t)wire_payload_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_result_fixed wire = r;
    dist_result_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_RESULT, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < telemetry_count; i++) {
        ds4_dist_telemetry_fixed tw = telemetry[i];
        dist_telemetry_to_wire(&tw);
        if (dist_write_full(fd, &tw, sizeof(tw)) != 0) return -1;
    }
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE && wire_payload_bytes != 0) {
        if (dist_write_activation_payload(transport, payload, hidden_values, payload_bits) != 0) return -1;
    } else if (payload_bytes && payload) {
        const bool payload_is_bulk = status == 0 &&
            result_kind == DS4_DIST_RESULT_LOGITS;
        if (payload_is_bulk) {
            if (ds4_transport_send_bulk(transport, payload, payload_bytes) != 0)
                return -1;
        } else if (dist_write_full(fd, payload, payload_bytes) != 0) {
            return -1;
        }
    }
    return 1;
}

static int dist_send_work_result(
        int fd,
        ds4_transport *transport,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes,
        bool allow_oob) {
    return dist_send_work_result_prepared(
        fd, transport, request_id, result_hash, status, result_kind,
        payload_bits, telemetry, telemetry_count, payload, payload_bytes,
        allow_oob, NULL, NULL);
}

static int dist_send_work_error(
        int fd,
        ds4_transport *transport,
        uint64_t request_id,
        const char *msg) {
    if (!msg) msg = "distributed work failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    return dist_send_work_result(fd, transport, request_id, 0, 1, 0, 0,
                                 NULL, 0, msg, (uint32_t)len, false);
}

static int dist_send_snapshot_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens,
        const char *msg) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX || begin->token_bytes != (uint32_t)token_bytes64) return -1;
    uint32_t msg_len = msg ? (uint32_t)strlen(msg) : 0u;
    if (msg_len != begin->message_bytes) return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64 + msg_len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_BEGIN, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (msg_len && dist_write_full(fd, msg, msg_len) != 0) return -1;
    return 1;
}

static int dist_send_snapshot_error(
        int fd,
        uint64_t request_id,
        uint64_t session_id,
        uint32_t model_id,
        uint32_t layer_start,
        uint32_t layer_end,
        const char *msg) {
    if (!msg) msg = "distributed snapshot failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_begin_fixed begin;
    memset(&begin, 0, sizeof(begin));
    begin.model_id = model_id;
    dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
    dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
    begin.layer_start = layer_start;
    begin.layer_end = layer_end;
    begin.status = 1;
    begin.message_bytes = (uint32_t)len;
    return dist_send_snapshot_begin(fd, &begin, NULL, msg);
}

static int dist_send_snapshot_done(int fd, uint64_t request_id, uint32_t status, const char *msg) {
    if (!msg) msg = "";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_done_fixed done;
    memset(&done, 0, sizeof(done));
    dist_u64_to_halves(request_id, &done.request_hi, &done.request_lo);
    done.status = status;
    done.message_bytes = (uint32_t)len;
    uint64_t frame_bytes64 = sizeof(done) + len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_done_fixed wire = done;
    dist_snapshot_done_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_DONE, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (len && dist_write_full(fd, msg, len) != 0) return -1;
    return 1;
}

static int dist_send_snapshot_file_chunks(int fd, uint64_t request_id, FILE *fp, uint64_t bytes) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) return -1;
    int rc = 1;
    while (bytes != 0) {
        const uint32_t n = bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (uint32_t)bytes;
        if (fread(buf, 1, n, fp) != n) {
            rc = -1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        dist_u64_to_halves(request_id, &chunk.request_hi, &chunk.request_lo);
        chunk.chunk_bytes = n;
        ds4_dist_snapshot_chunk_fixed wire = chunk;
        dist_snapshot_chunk_to_wire(&wire);
        const uint32_t frame_bytes = (uint32_t)sizeof(wire) + n;
        if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_CHUNK, frame_bytes) != 0 ||
            dist_write_full(fd, &wire, sizeof(wire)) != 0 ||
            dist_write_full(fd, buf, n) != 0) {
            rc = -1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

/* =========================================================================
 * Worker Route Parsing And Forwarding
 * ========================================================================= */

static bool dist_route_get_entry(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t target_index,
        ds4_dist_route_entry *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || target_index >= route_count) {
        if (errlen) snprintf(err, errlen, "invalid route entry index");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (i == target_index) {
            memcpy(out->host, p, fixed.host_len);
            out->host[fixed.host_len] = '\0';
            out->port = fixed.port;
            out->layer_start = fixed.layer_start;
            out->layer_end = fixed.layer_end;
            out->flags = fixed.flags;
            out->fd = -1;
            return true;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    if (errlen) snprintf(err, errlen, "route entry not found");
    return false;
}

static bool dist_route_get_return_target(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        ds4_dist_route_return *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || route_count == 0) {
        if (errlen) snprintf(err, errlen, "invalid route final destination");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed fixed;
    memcpy(&fixed, p, sizeof(fixed));
    dist_route_return_from_wire(&fixed);
    p += sizeof(fixed);
    remaining -= (uint32_t)sizeof(fixed);
    if (fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, fixed.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->kind = fixed.kind;
    out->port = fixed.port;
    if (fixed.host_len) {
        memcpy(out->host, p, fixed.host_len);
        out->host[fixed.host_len] = '\0';
    }
    p += fixed.host_len;
    remaining -= fixed.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

static bool dist_route_validate_blob(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t n_layers,
        char *err,
        size_t errlen) {
    if (route_count == 0) {
        if (route_bytes == 0) return true;
        if (errlen) snprintf(err, errlen, "route payload has entries without a route count");
        return false;
    }
    if (!route_blob) {
        if (errlen) snprintf(err, errlen, "route payload is missing");
        return false;
    }

    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    uint32_t prev_end = UINT32_MAX;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);

        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (fixed.port == 0 || fixed.port > 65535u) {
            if (errlen) snprintf(err, errlen, "invalid route port");
            return false;
        }
        if (fixed.layer_start >= n_layers || fixed.layer_end >= n_layers ||
            fixed.layer_end < fixed.layer_start) {
            if (errlen) snprintf(err, errlen, "invalid route layer range");
            return false;
        }
        if ((fixed.flags & ~DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
            if (errlen) snprintf(err, errlen, "invalid route flags");
            return false;
        }
        if ((fixed.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0 &&
            fixed.layer_end + 1u != n_layers) {
            if (errlen) snprintf(err, errlen, "route logits require final layer");
            return false;
        }
        if (i != 0 && fixed.layer_start != prev_end + 1u) {
            if (errlen) snprintf(err, errlen, "route layer ranges are not contiguous");
            return false;
        }

        p += fixed.host_len;
        remaining -= fixed.host_len;
        prev_end = fixed.layer_end;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed ret;
    memcpy(&ret, p, sizeof(ret));
    dist_route_return_from_wire(&ret);
    p += sizeof(ret);
    remaining -= (uint32_t)sizeof(ret);
    if (ret.host_len >= NI_MAXHOST || ret.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, ret.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
        if (errlen) snprintf(err, errlen, "unsupported route final destination");
        return false;
    }
    if (ret.host_len != 0 || ret.port != 0) {
        if (errlen) snprintf(err, errlen, "invalid upstream route final destination");
        return false;
    }
    p += ret.host_len;
    remaining -= ret.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

static int dist_send_work_frame(
        int fd,
        ds4_transport *transport,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob) {
    return dist_send_work_frame_prepared(fd, transport, work, tokens,
                                         input_hc, route_blob, NULL, NULL);
}

static int dist_send_work_frame_prepared(
        int fd,
        ds4_transport *transport,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob,
        const ds4_transport_bulk_desc *prepared_desc,
        ds4_transport_lease *tx_lease) {
    if (!work || !tokens || work->n_tokens == 0) return -1;
    const uint64_t token_bytes = (uint64_t)work->n_tokens * sizeof(uint32_t);
    if (token_bytes > UINT32_MAX || work->token_bytes != (uint32_t)token_bytes) return -1;
    if (work->input_hc_bytes != 0 && !input_hc) return -1;
    if (work->route_bytes != 0 && !route_blob) return -1;
    uint64_t input_hc_values = 0;
    if (work->input_hc_bytes != 0 &&
        !dist_activation_values_from_wire_bytes(work->input_hc_bits,
                                                work->input_hc_bytes,
                                                &input_hc_values))
        return -1;
    if (dist_transport_uses_v3(transport)) {
        const uint64_t session_id = dist_u64_from_halves(
            work->session_hi, work->session_lo);
        const uint64_t request_id = dist_u64_from_halves(
            work->request_hi, work->request_lo);
        ds4_transport_bulk_desc desc;
        const void *bulk_wire = NULL;
        void *bulk_owned = NULL;
        uint32_t bulk_bytes = 0;
        bool local_desc_prepared = false;
        if (work->input_hc_bytes != 0) {
            if (dist_pack_activation_payload(input_hc, input_hc_values,
                                             work->input_hc_bits,
                                             &bulk_wire, &bulk_owned,
                                             &bulk_bytes) != 0 ||
                bulk_bytes != work->input_hc_bytes) {
                free(bulk_owned);
                return -1;
            }
            if (prepared_desc) {
                desc = *prepared_desc;
                const int desc_rc = desc.mode == DS4_TRANSPORT_BULK_NHI_OOB
                    ? ds4_transport_validate_oob_desc(
                          transport, &desc,
                          DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                          session_id, request_id, bulk_bytes,
                          dist_activation_bits_or_default(work->input_hc_bits),
                          NULL, 0)
                    : ds4_transport_validate_inline_desc(
                          &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                          session_id, request_id, bulk_bytes,
                          dist_activation_bits_or_default(work->input_hc_bits),
                          NULL, 0);
                if (desc_rc != 0) {
                    free(bulk_owned);
                    return -1;
                }
            } else if (ds4_transport_prepare_bulk(
                           transport, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                           session_id, request_id, bulk_bytes,
                           dist_activation_bits_or_default(work->input_hc_bits),
                           &desc, NULL, 0) != 0) {
                free(bulk_owned);
                return -1;
            } else {
                local_desc_prepared = true;
            }
            /* One NHI endpoint represents one coordinator/worker route. A
             * multi-hop route remains descriptor-framed but inline. */
            if (work->route_count != 1u &&
                desc.mode == DS4_TRANSPORT_BULK_NHI_OOB) {
                desc.mode = DS4_TRANSPORT_BULK_TCP_INLINE;
                desc.frame_count = 0;
            }
        } else {
            if (prepared_desc || tx_lease) return -1;
            dist_bulk_none_desc(transport, session_id, request_id, &desc);
        }

        if (tx_lease &&
            (!prepared_desc || desc.mode != DS4_TRANSPORT_BULK_NHI_OOB ||
             ds4_transport_lease_host_ptr(tx_lease) != bulk_wire ||
             ds4_transport_lease_bytes(tx_lease) != bulk_bytes)) {
            free(bulk_owned);
            errno = EINVAL;
            return -1;
        }

        uint32_t control_bytes = 0;
        uint32_t frame_bytes = 0;
        if (ds4_dist_v3_work_frame_sizes(work->token_bytes,
                                         work->route_bytes,
                                         bulk_bytes,
                                         desc.mode,
                                         &control_bytes,
                                         &frame_bytes) != 0) {
            if (local_desc_prepared) shutdown(fd, SHUT_RDWR);
            free(bulk_owned);
            return -1;
        }
        (void)control_bytes;
        ds4_transport_bulk_desc_wire desc_wire;
        if (ds4_transport_bulk_desc_encode(&desc, &desc_wire) != 0) {
            if (local_desc_prepared) shutdown(fd, SHUT_RDWR);
            free(bulk_owned);
            return -1;
        }
        ds4_dist_work_fixed work_wire = *work;
        dist_work_to_wire(&work_wire);
        int rc = -1;
        bool lease_control_mark_attempted = false;
        if (dist_write_frame_header(fd, DS4_DIST_MSG_WORK, frame_bytes) != 0 ||
            dist_write_full(fd, &work_wire, sizeof(work_wire)) != 0 ||
            dist_write_full(fd, &desc_wire, sizeof(desc_wire)) != 0)
            goto v3_done;
        for (uint32_t i = 0; i < work->n_tokens; i++) {
            uint32_t token = htonl((uint32_t)tokens[i]);
            if (dist_write_full(fd, &token, sizeof(token)) != 0)
                goto v3_done;
        }
        if (work->route_bytes &&
            dist_write_full(fd, route_blob, work->route_bytes) != 0)
            goto v3_done;
        if (bulk_bytes != 0) {
            if (tx_lease) {
                lease_control_mark_attempted = true;
                if (ds4_transport_tx_lease_mark_control_sent(tx_lease) != 0 ||
                    dist_mapped_lease_commit_after_gpu_quiesce(tx_lease) != 0)
                    goto v3_done;
            } else if (ds4_transport_send_bulk_desc(
                           transport, &desc, bulk_wire, bulk_bytes) != 0) {
                goto v3_done;
            }
        }
        rc = 0;
v3_done:
        if (rc < 0) {
            const int saved = errno ? errno : EIO;
            if (tx_lease && !lease_control_mark_attempted)
                (void)ds4_transport_tx_lease_mark_control_sent(tx_lease);
            shutdown(fd, SHUT_RDWR);
            errno = saved;
        }
        free(bulk_owned);
        return rc;
    }
    const uint64_t frame_bytes = sizeof(ds4_dist_work_fixed) +
                                 (uint64_t)work->token_bytes +
                                 work->input_hc_bytes +
                                 work->route_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_work_fixed wire = *work;
    dist_work_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_WORK, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < work->n_tokens; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (work->input_hc_bytes &&
        dist_write_activation_payload(transport,
                                      input_hc,
                                      input_hc_values,
                                      work->input_hc_bits) != 0)
        return -1;
    if (work->route_bytes && dist_write_full(fd, route_blob, work->route_bytes) != 0) return -1;
    return 0;
}

static int dist_worker_upstream_send_work_result_prepared(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes,
        bool allow_oob,
        const ds4_dist_tx_bulk_plan *tx_plan) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_result_prepared(
        upstream->fd, upstream->transport, request_id, result_hash,
        status, result_kind, payload_bits, telemetry, telemetry_count,
        payload, payload_bytes, allow_oob,
        dist_tx_bulk_plan_desc(tx_plan), tx_plan ? tx_plan->lease : NULL);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}


static int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_error(upstream->fd, upstream->transport, request_id, msg);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}

static int dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd,
        ds4_transport *transport,
        uint32_t selected_caps) {
    memset(upstream, 0, sizeof(*upstream));
    upstream->state = state;
    upstream->fd = fd;
    upstream->selected_caps = selected_caps;
    upstream->transport = transport
        ? ds4_transport_retain(transport)
        : ds4_transport_tcp_create(fd, NULL, 0);
    if (!upstream->transport) return -1;
    if (pthread_mutex_init(&upstream->write_mu, NULL) != 0) {
        ds4_transport_release(upstream->transport);
        upstream->transport = NULL;
        return -1;
    }
    if (pthread_mutex_init(&upstream->forward_mu, NULL) != 0) {
        pthread_mutex_destroy(&upstream->write_mu);
        ds4_transport_release(upstream->transport);
        upstream->transport = NULL;
        return -1;
    }
    return 0;
}

static bool dist_worker_forwarder_enqueue_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        const ds4_dist_telemetry_fixed *telemetry,
        const ds4_dist_v3_result_limits *result_limits,
        uint32_t activation_bits,
        uint32_t telemetry_count_limit,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    while (!forwarder->closing &&
           forwarder->pending_depth != 0 &&
           forwarder->pending_count >= forwarder->pending_depth) {
        pthread_cond_wait(&forwarder->queue_not_full, &forwarder->queue_mu);
    }
    if (forwarder->closing) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    ds4_dist_pending_request *node = calloc(1, sizeof(*node));
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    node->request_id = request_id;
    node->downstream_t0 = downstream_t0;
    if (telemetry) node->telemetry = *telemetry;
    if (result_limits) node->result_limits = *result_limits;
    node->activation_bits = activation_bits;
    node->telemetry_count_limit = telemetry_count_limit;
    if (forwarder->pending_tail) forwarder->pending_tail->next = node;
    else forwarder->pending_head = node;
    forwarder->pending_tail = node;
    forwarder->pending_count++;
    pthread_mutex_unlock(&forwarder->queue_mu);
    return true;
}

static bool dist_worker_forwarder_pop_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t *request_id,
        ds4_dist_telemetry_fixed *telemetry,
        ds4_dist_v3_result_limits *result_limits,
        uint32_t *activation_bits,
        uint32_t *telemetry_count_limit,
        double *downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *node = forwarder->pending_head;
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    forwarder->pending_head = node->next;
    if (!forwarder->pending_head) forwarder->pending_tail = NULL;
    if (forwarder->pending_count != 0) forwarder->pending_count--;
    pthread_cond_signal(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    if (request_id) *request_id = node->request_id;
    if (telemetry) *telemetry = node->telemetry;
    if (result_limits) *result_limits = node->result_limits;
    if (activation_bits) *activation_bits = node->activation_bits;
    if (telemetry_count_limit)
        *telemetry_count_limit = node->telemetry_count_limit;
    if (downstream_t0) *downstream_t0 = node->downstream_t0;
    free(node);
    return true;
}

static bool dist_worker_forwarder_remove_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request **link = &forwarder->pending_head;
    ds4_dist_pending_request *prev = NULL;
    while (*link) {
        ds4_dist_pending_request *node = *link;
        if (node->request_id == request_id) {
            *link = node->next;
            if (forwarder->pending_tail == node) forwarder->pending_tail = prev;
            if (forwarder->pending_count != 0) forwarder->pending_count--;
            pthread_cond_signal(&forwarder->queue_not_full);
            pthread_mutex_unlock(&forwarder->queue_mu);
            free(node);
            return true;
        }
        prev = node;
        link = &node->next;
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
    return false;
}

static void dist_worker_forwarder_note_send_done(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        uint32_t forward_send_usec,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    for (ds4_dist_pending_request *node = forwarder->pending_head; node; node = node->next) {
        if (node->request_id == request_id) {
            node->telemetry.forward_send_usec = forward_send_usec;
            node->downstream_t0 = downstream_t0;
            break;
        }
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void dist_worker_forwarder_clear_requests(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *it = forwarder->pending_head;
    forwarder->pending_head = NULL;
    forwarder->pending_tail = NULL;
    forwarder->pending_count = 0;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    while (it) {
        ds4_dist_pending_request *next = it->next;
        free(it);
        it = next;
    }
}

static void dist_worker_forwarder_close_queue(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    forwarder->closing = true;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void *dist_worker_forwarder_relay_main(void *arg) {
    ds4_dist_worker_forwarder *forwarder = arg;
    ds4_dist_worker_upstream *upstream = forwarder->upstream;
    int fd = forwarder->fd;
    uint8_t *buf = malloc(1024 * 1024);
    if (!buf) {
        shutdown(upstream->fd, SHUT_RDWR);
        return NULL;
    }
    DIST_DEBUG("relay start downstream=%s:%u fd=%d upstream_fd=%d",
               forwarder->host,
               forwarder->port,
               fd,
               upstream->fd);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc <= 0) {
            uint64_t pending_request = 0;
            if (dist_worker_forwarder_pop_request(
                    forwarder, &pending_request, NULL, NULL,
                    NULL, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     pending_request,
                                                     "next worker closed connection");
            }
            DIST_DEBUG("relay read header end downstream=%s:%u rc=%d err=%s",
                       forwarder->host,
                       forwarder->port,
                       rc,
                       err);
            break;
        }
        DIST_DEBUG("relay got frame downstream=%s:%u type=%u bytes=%u",
                   forwarder->host,
                   forwarder->port,
                   type,
                   bytes);
        uint64_t expected_request = 0;
        if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
            if (dist_worker_forwarder_pop_request(
                    forwarder, &expected_request, NULL, NULL,
                    NULL, NULL, NULL)) {
                dist_worker_upstream_send_work_error(
                    upstream, expected_request,
                    "next worker did not return valid RESULT");
            }
            shutdown(fd, SHUT_RDWR);
            DIST_DEBUG("relay invalid frame downstream=%s:%u",
                       forwarder->host, forwarder->port);
            break;
        }

        ds4_dist_result_fixed wire_result;
        rc = dist_read_full(fd, &wire_result, sizeof(wire_result));
        if (rc <= 0) {
            if (dist_worker_forwarder_pop_request(
                    forwarder, &expected_request, NULL, NULL,
                    NULL, NULL, NULL)) {
                dist_worker_upstream_send_work_error(
                    upstream, expected_request,
                    "next worker closed while returning RESULT");
            }
            DIST_DEBUG("relay read result fixed failed downstream=%s:%u rc=%d",
                       forwarder->host,
                       forwarder->port,
                       rc);
            break;
        }

        ds4_dist_result_fixed result = wire_result;
        dist_result_from_wire(&result);
        const uint64_t got_request =
            dist_u64_from_halves(result.request_hi, result.request_lo);
        const uint32_t body_bytes = bytes - (uint32_t)sizeof(wire_result);
        ds4_dist_telemetry_fixed local_telemetry;
        ds4_dist_v3_result_limits result_limits;
        uint32_t activation_bits = 0;
        uint32_t telemetry_count_limit = 0;
        double downstream_t0 = 0.0;
        if (!dist_worker_forwarder_pop_request(
                forwarder, &expected_request, &local_telemetry,
                &result_limits, &activation_bits,
                &telemetry_count_limit, &downstream_t0)) {
            shutdown(fd, SHUT_RDWR);
            DIST_DEBUG("relay got unexpected result request=%llu with no pending request",
                       (unsigned long long)got_request);
            break;
        }
        if (got_request != expected_request) {
            dist_worker_upstream_send_work_error(
                upstream, expected_request,
                "next worker RESULT request mismatch");
            shutdown(fd, SHUT_RDWR);
            DIST_DEBUG("relay request mismatch expected=%llu got=%llu",
                       (unsigned long long)expected_request,
                       (unsigned long long)got_request);
            break;
        }
        const bool valid_result =
            ds4_dist_v3_result_payload_validate(
                result.status, result.result_kind, result.payload_bytes,
                result.payload_bits, &result_limits, activation_bits,
                err, sizeof(err)) == 0;
        if (!valid_result || telemetry_count_limit == 0u ||
            result.telemetry_bytes %
                    (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
            result.telemetry_count !=
                result.telemetry_bytes /
                    (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
            result.telemetry_count > telemetry_count_limit ||
            result.telemetry_bytes > body_bytes ||
            result.payload_bytes != body_bytes - result.telemetry_bytes) {
            dist_worker_upstream_send_work_error(
                upstream, expected_request,
                valid_result
                    ? "next worker RESULT metadata exceeds request bounds"
                    : (err[0] ? err
                              : "next worker RESULT payload exceeds request bounds"));
            shutdown(fd, SHUT_RDWR);
            DIST_DEBUG("relay result bounds mismatch downstream=%s:%u telemetry=%u payload=%u frame=%u limit=%u",
                       forwarder->host,
                       forwarder->port,
                       result.telemetry_bytes,
                       result.payload_bytes,
                       body_bytes,
                       telemetry_count_limit);
            break;
        }
        local_telemetry.downstream_wait_usec = dist_usec_since(downstream_t0, dist_now_sec());
        const uint64_t out_telemetry_bytes64 =
            (uint64_t)result.telemetry_bytes + sizeof(ds4_dist_telemetry_fixed);
        const uint32_t out_telemetry_count = result.telemetry_count + 1u;
        const uint32_t model_telemetry_limit =
            (uint32_t)ds4_engine_layer_count(upstream->state->engine) + 1u;
        if (out_telemetry_bytes64 > UINT32_MAX ||
            out_telemetry_count == 0 ||
            out_telemetry_count > model_telemetry_limit) {
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed telemetry chain is too large");
            shutdown(fd, SHUT_RDWR);
            break;
        }
        const uint32_t out_telemetry_bytes = (uint32_t)out_telemetry_bytes64;
        const uint64_t out_frame_bytes64 = sizeof(ds4_dist_result_fixed) +
                                           out_telemetry_bytes64 +
                                           (uint64_t)result.payload_bytes;
        if (out_frame_bytes64 > UINT32_MAX) {
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed RESULT frame is too large");
            shutdown(fd, SHUT_RDWR);
            break;
        }
        DIST_DEBUG("relay result request=%llu status=%u kind=%u telemetry=%u payload=%u",
                   (unsigned long long)got_request,
                   result.status,
                   result.result_kind,
                   out_telemetry_count,
                   result.payload_bytes);

        pthread_mutex_lock(&upstream->write_mu);
        result.telemetry_count = out_telemetry_count;
        result.telemetry_bytes = out_telemetry_bytes;
        ds4_dist_result_fixed out_wire_result = result;
        dist_result_to_wire(&out_wire_result);
        int write_rc = dist_write_frame_header(upstream->fd,
                                               DS4_DIST_MSG_RESULT,
                                               (uint32_t)out_frame_bytes64);
        if (write_rc == 0) write_rc = dist_write_full(upstream->fd, &out_wire_result, sizeof(out_wire_result));

        uint32_t remaining = result.telemetry_bytes - (uint32_t)sizeof(ds4_dist_telemetry_fixed);
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        if (write_rc == 0) {
            ds4_dist_telemetry_fixed local_wire = local_telemetry;
            dist_telemetry_to_wire(&local_wire);
            if (dist_write_full(upstream->fd, &local_wire, sizeof(local_wire)) != 0) {
                write_rc = -1;
            }
        }

        remaining = result.payload_bytes;
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        pthread_mutex_unlock(&upstream->write_mu);

        DIST_DEBUG("relay wrote result request=%llu write_rc=%d remaining=%u",
                   (unsigned long long)got_request,
                   write_rc,
                   remaining);
        if (write_rc != 0) break;
        if (remaining != 0) break;
    }

    DIST_DEBUG("relay closing upstream_fd=%d", upstream->fd);
    dist_worker_forwarder_close_queue(forwarder);
    shutdown(upstream->fd, SHUT_RDWR);
    free(buf);
    return NULL;
}

static ds4_dist_worker_forwarder *dist_worker_get_forwarder(
        ds4_dist_worker_upstream *upstream,
        const char *host,
        uint32_t port,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&upstream->forward_mu);
    for (ds4_dist_worker_forwarder *it = upstream->forwarders; it; it = it->next) {
        if (it->port == port && !strcmp(it->host, host)) {
            pthread_mutex_unlock(&upstream->forward_mu);
            return it;
        }
    }

    int fd = dist_connect_endpoint(host, (int)port, err, errlen);
    if (fd < 0) {
        pthread_mutex_unlock(&upstream->forward_mu);
        return NULL;
    }

    ds4_dist_worker_forwarder *forwarder = calloc(1, sizeof(*forwarder));
    if (!forwarder) {
        close(fd);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "out of memory creating worker-to-worker forwarder");
        return NULL;
    }
    forwarder->upstream = upstream;
    snprintf(forwarder->host, sizeof(forwarder->host), "%s", host);
    forwarder->port = port;
    forwarder->fd = fd;
    forwarder->transport = ds4_transport_tcp_create(fd, err, errlen);
    if (!forwarder->transport) {
        close(fd);
        free(forwarder);
        pthread_mutex_unlock(&upstream->forward_mu);
        return NULL;
    }
    forwarder->pending_depth = dist_worker_forward_window();
    pthread_mutex_init(&forwarder->send_mu, NULL);
    pthread_mutex_init(&forwarder->queue_mu, NULL);
    pthread_cond_init(&forwarder->queue_not_full, NULL);
    if (pthread_create(&forwarder->tid, NULL, dist_worker_forwarder_relay_main, forwarder) != 0) {
        pthread_cond_destroy(&forwarder->queue_not_full);
        pthread_mutex_destroy(&forwarder->queue_mu);
        pthread_mutex_destroy(&forwarder->send_mu);
        close(fd);
        free(forwarder);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "failed to start worker-to-worker relay thread");
        return NULL;
    }
    forwarder->thread_started = true;
    forwarder->next = upstream->forwarders;
    upstream->forwarders = forwarder;
    pthread_mutex_unlock(&upstream->forward_mu);

    fprintf(stderr,
            "ds4: distributed worker: opened pipelined worker-to-worker connection to %s:%u (window %u)\n",
            host,
            port,
            forwarder->pending_depth);
    return forwarder;
}

static void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream) {
    pthread_mutex_lock(&upstream->forward_mu);
    ds4_dist_worker_forwarder *forwarders = upstream->forwarders;
    upstream->forwarders = NULL;
    pthread_mutex_unlock(&upstream->forward_mu);

    for (ds4_dist_worker_forwarder *it = forwarders; it; it = it->next) {
        dist_worker_forwarder_close_queue(it);
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    while (forwarders) {
        ds4_dist_worker_forwarder *next = forwarders->next;
        if (forwarders->thread_started) pthread_join(forwarders->tid, NULL);
        if (forwarders->fd >= 0) close(forwarders->fd);
        ds4_transport_release(forwarders->transport);
        dist_worker_forwarder_clear_requests(forwarders);
        pthread_cond_destroy(&forwarders->queue_not_full);
        pthread_mutex_destroy(&forwarders->queue_mu);
        pthread_mutex_destroy(&forwarders->send_mu);
        free(forwarders);
        forwarders = next;
    }

    ds4_transport_release(upstream->transport);
    pthread_mutex_destroy(&upstream->forward_mu);
    pthread_mutex_destroy(&upstream->write_mu);
}

static int dist_forward_work_to_next(
        ds4_dist_worker_upstream *upstream,
        const ds4_dist_route_entry *next,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        const ds4_dist_telemetry_fixed *telemetry,
        const void *route_blob) {
    char err[256];
    ds4_dist_worker_forwarder *forwarder =
        dist_worker_get_forwarder(upstream, next->host, next->port, err, sizeof(err));
    const uint64_t request_id = dist_u64_from_halves(work->request_hi, work->request_lo);
    if (!forwarder) {
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    ds4_dist_work_fixed forwarded = *work;
    forwarded.layer_start = next->layer_start;
    forwarded.layer_end = next->layer_end;
    forwarded.route_index = work->route_index + 1u;
    forwarded.flags |= DS4_DIST_WORK_F_INPUT_HC;
    forwarded.input_hc_bits = dist_activation_bits_or_default(work->input_hc_bits);
    if (!dist_activation_wire_bytes_from_f32_bytes(forwarded.input_hc_bits,
                                                   hidden_hc_bytes,
                                                   &forwarded.input_hc_bytes)) {
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "invalid forwarded hidden-state size");
    }
    if ((next->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        forwarded.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    } else {
        forwarded.flags &= ~DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }

    if (work->route_count == 0u ||
        forwarded.route_index >= work->route_count) {
        return dist_worker_upstream_send_work_error(
            upstream, request_id, "invalid forwarded route index");
    }
    ds4_dist_route_entry final_route;
    if (!dist_route_get_entry(route_blob, work->route_bytes,
                              work->route_count, work->route_count - 1u,
                              &final_route, err, sizeof(err))) {
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    err);
    }
    const bool final_output_logits =
        (final_route.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    ds4_dist_v3_result_limits result_limits;
    if (dist_result_limits_for_work(
            upstream->state->engine, forwarded.input_hc_bits,
            hidden_hc_bytes, forwarded.n_tokens, forwarded.flags,
            final_output_logits, &result_limits,
            err, sizeof(err)) != 0) {
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    err);
    }
    uint32_t telemetry_count_limit =
        work->route_count - forwarded.route_index;
    const uint32_t model_telemetry_limit =
        (uint32_t)ds4_engine_layer_count(upstream->state->engine);
    if (telemetry_count_limit > model_telemetry_limit)
        telemetry_count_limit = model_telemetry_limit;

    pthread_mutex_lock(&forwarder->send_mu);
    const double send_t0 = dist_now_sec();
    if (!dist_worker_forwarder_enqueue_request(
            forwarder, request_id, telemetry, &result_limits,
            forwarded.input_hc_bits, telemetry_count_limit, send_t0)) {
        pthread_mutex_unlock(&forwarder->send_mu);
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "out of memory tracking forwarded request");
    }
    DIST_DEBUG("forward send request=%llu to %s:%u route_index=%u tokens=%u pos=%u bytes=%u",
               (unsigned long long)request_id,
               next->host,
               next->port,
               forwarded.route_index,
               forwarded.n_tokens,
               forwarded.pos0,
               forwarded.input_hc_bytes);
    int rc = dist_send_work_frame(forwarder->fd, forwarder->transport,
                                  &forwarded, tokens, hidden_hc, route_blob);
    const double send_t1 = dist_now_sec();
    dist_worker_forwarder_note_send_done(forwarder,
                                         request_id,
                                         dist_usec_since(send_t0, send_t1),
                                         send_t1);
    pthread_mutex_unlock(&forwarder->send_mu);
    if (rc != 0) {
        DIST_DEBUG("forward send failed request=%llu to %s:%u",
                   (unsigned long long)request_id,
                   next->host,
                   next->port);
        dist_worker_forwarder_remove_request(forwarder, request_id);
        shutdown(forwarder->fd, SHUT_RDWR);
        int err_rc = dist_worker_upstream_send_work_error(upstream,
                                                          request_id,
                                                          "failed to forward distributed work");
        shutdown(upstream->fd, SHUT_RDWR);
        return err_rc;
    }
    DIST_DEBUG("forward send ok request=%llu", (unsigned long long)request_id);
    return 1;
}

/* =========================================================================
 * Worker KV Sessions And Snapshot Handlers
 * ========================================================================= */

static ds4_dist_worker_session *dist_worker_get_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id,
        char *err,
        size_t errlen) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }

    ds4_dist_worker_session *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return NULL;
    }
    if (ds4_session_create(&entry->session, state->engine, state->ctx_size) != 0) {
        free(entry);
        if (errlen) snprintf(err, errlen, "failed to create distributed worker session");
        return NULL;
    }
    entry->session_id = session_id;
    entry->next = state->sessions;
    state->sessions = entry;
    return entry;
}

static ds4_dist_worker_session *dist_worker_find_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }
    return NULL;
}

static uint32_t dist_worker_clear_sessions(ds4_dist_worker_state *state) {
    uint32_t n = 0;
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *it = state->sessions;
    state->sessions = NULL;
    pthread_mutex_unlock(&state->mu);

    while (it) {
        ds4_dist_worker_session *next = it->next;
        ds4_session_free(it->session);
        free(it);
        it = next;
        n++;
    }
    return n;
}

typedef struct {
    const uint8_t *p;
    uint32_t remaining;
} ds4_dist_mem_reader;


static int dist_temp_file(const char *prefix, char *path, size_t path_len, FILE **fp_out) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s.XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    FILE *fp = fdopen(fd, "w+b");
    if (!fp) {
        close(fd);
        unlink(tmpl);
        return -1;
    }
    snprintf(path, path_len, "%s", tmpl);
    *fp_out = fp;
    return 0;
}

static int dist_worker_handle_snapshot_save(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_req_fixed req;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    if (bytes != sizeof(req)) {
        dist_discard_bytes(upstream->fd, bytes);
        pthread_mutex_lock(&upstream->write_mu);
        int rc = dist_send_snapshot_error(upstream->fd, 0, 0, state->model_id,
                                          state->layer_start, state->layer_end,
                                          "invalid distributed snapshot save request");
        pthread_mutex_unlock(&upstream->write_mu);
        return rc;
    }
    int rc = dist_read_full(upstream->fd, &req, sizeof(req));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_req_from_wire(&req);
    request_id = dist_u64_from_halves(req.request_hi, req.request_lo);
    session_id = dist_u64_from_halves(req.session_hi, req.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(req.token_hash_hi, req.token_hash_lo);

    char err[256] = {0};
    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    uint64_t payload_bytes = 0;

    if (req.model_id != state->model_id ||
        req.layer_start != state->layer_start ||
        req.layer_end != state->layer_end ||
        req.token_count > (uint32_t)state->ctx_size) {
        snprintf(err, sizeof(err), "snapshot save request does not match worker state");
    } else {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session = dist_worker_find_session_locked(state, session_id);
        if (!session) {
            snprintf(err, sizeof(err), "worker has no distributed session to snapshot");
        } else {
            const ds4_tokens *timeline = ds4_session_tokens(session->session);
            uint64_t live_hash = 0;
            if (!timeline || timeline->len < 0 || (uint32_t)timeline->len != req.token_count) {
                snprintf(err, sizeof(err), "worker snapshot token count mismatch");
            } else {
                live_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
                if (live_hash != token_hash) {
                    snprintf(err, sizeof(err), "worker snapshot token hash mismatch");
                }
            }
            if (!err[0] && dist_temp_file("ds4-dist-save", tmp_path, sizeof(tmp_path), &tmp) != 0) {
                snprintf(err, sizeof(err), "failed to create worker snapshot temp file");
            }
            if (!err[0] &&
                ds4_session_save_layer_payload(session->session,
                                               tmp,
                                               state->layer_start,
                                               state->layer_end,
                                               err,
                                               sizeof(err)) != 0) {
                if (!err[0]) snprintf(err, sizeof(err), "failed to save worker KV shard");
            }
            if (!err[0] && fflush(tmp) != 0) {
                snprintf(err, sizeof(err), "failed to flush worker KV shard");
            }
            if (!err[0]) {
                off_t pos = ftello(tmp);
                if (pos < 0) snprintf(err, sizeof(err), "failed to measure worker KV shard");
                else payload_bytes = (uint64_t)pos;
            }
            if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
                snprintf(err, sizeof(err), "failed to rewind worker KV shard");
            }
        }
        pthread_mutex_unlock(&state->mu);
    }

    pthread_mutex_lock(&upstream->write_mu);
    if (err[0]) {
        rc = dist_send_snapshot_error(upstream->fd,
                                      request_id,
                                      session_id,
                                      state->model_id,
                                      state->layer_start,
                                      state->layer_end,
                                      err);
    } else {
        ds4_dist_snapshot_begin_fixed begin;
        memset(&begin, 0, sizeof(begin));
        begin.model_id = state->model_id;
        dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
        dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
        dist_u64_to_halves(token_hash, &begin.token_hash_hi, &begin.token_hash_lo);
        begin.layer_start = state->layer_start;
        begin.layer_end = state->layer_end;
        dist_u64_to_halves(payload_bytes, &begin.payload_hi, &begin.payload_lo);
        rc = dist_send_snapshot_begin(upstream->fd, &begin, NULL, NULL);
        if (rc > 0) rc = dist_send_snapshot_file_chunks(upstream->fd, request_id, tmp, payload_bytes);
        if (rc > 0) rc = dist_send_snapshot_done(upstream->fd, request_id, 0, NULL);
    }
    pthread_mutex_unlock(&upstream->write_mu);

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    return rc;
}

static int dist_worker_handle_snapshot_load(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_begin_fixed begin;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    char err[256] = {0};
    if (bytes < sizeof(begin)) {
        dist_discard_bytes(upstream->fd, bytes);
        return -1;
    }
    int rc = dist_read_full(upstream->fd, &begin, sizeof(begin));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_begin_from_wire(&begin);
    request_id = dist_u64_from_halves(begin.request_hi, begin.request_lo);
    session_id = dist_u64_from_halves(begin.session_hi, begin.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(begin.token_hash_hi,
                                                     begin.token_hash_lo);
    const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi,
                                                        begin.payload_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(begin);
    uint64_t expected_token_bytes = (uint64_t)begin.token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin.token_bytes != (uint32_t)expected_token_bytes ||
        begin.message_bytes != 0 ||
        body_bytes != begin.token_bytes) {
        dist_discard_bytes(upstream->fd, body_bytes);
        snprintf(err, sizeof(err), "invalid distributed snapshot load header");
    }

    /* Reject a token_count larger than the worker context up front, before the
     * allocation and the read loop below. token_count is only bounded to
     * ~UINT32_MAX/4 by the header check above, so without this a peer could
     * drive a multi-GB malloc and a full-length socket read before the
     * matching-state check further down (which also tests token_count > ctx_size)
     * finally rejects the request. */
    if (!err[0] && begin.token_count > (uint32_t)state->ctx_size) {
        dist_discard_bytes(upstream->fd, body_bytes);
        snprintf(err, sizeof(err), "snapshot token count exceeds worker context size");
    }

    int *tokens = NULL;
    if (!err[0]) {
        tokens = malloc((size_t)begin.token_count * sizeof(tokens[0]));
        if (!tokens && begin.token_count != 0) {
            dist_discard_bytes(upstream->fd, begin.token_bytes);
            snprintf(err, sizeof(err), "out of memory reading snapshot tokens");
        }
    }
    for (uint32_t i = 0; !err[0] && i < begin.token_count; i++) {
        uint32_t wire_token = 0;
        rc = dist_read_full(upstream->fd, &wire_token, sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return rc == 0 ? 0 : -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX ||
            token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            snprintf(err, sizeof(err), "snapshot token id is outside the model vocabulary");
            tokens[i] = 0;
        } else {
            tokens[i] = (int)token;
        }
    }
    if (!err[0] &&
        dist_token_hash_prefix(tokens, begin.token_count) != token_hash) {
        snprintf(err, sizeof(err), "snapshot load token hash mismatch");
    }
    if (!err[0] &&
        (begin.model_id != state->model_id ||
         begin.layer_start != state->layer_start ||
         begin.layer_end != state->layer_end ||
         begin.token_count > (uint32_t)state->ctx_size)) {
        snprintf(err, sizeof(err), "snapshot load request does not match worker state");
    }

    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    if (!err[0] && dist_temp_file("ds4-dist-load", tmp_path, sizeof(tmp_path), &tmp) != 0) {
        snprintf(err, sizeof(err), "failed to create worker snapshot restore temp file");
    }

    uint8_t *buf = NULL;
    if (!err[0]) {
        buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
        if (!buf) snprintf(err, sizeof(err), "out of memory restoring worker KV shard");
    }
    uint64_t received = 0;
    while (!err[0] && received < payload_bytes) {
        uint32_t type = 0, chunk_frame_bytes = 0;
        rc = dist_read_frame_header(upstream->fd, &type, &chunk_frame_bytes, err, sizeof(err));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            chunk_frame_bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(upstream->fd, chunk_frame_bytes);
            snprintf(err, sizeof(err), "expected distributed snapshot chunk");
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(upstream->fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi, chunk.request_lo);
        uint32_t chunk_bytes = chunk_frame_bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(upstream->fd, chunk_bytes);
            snprintf(err, sizeof(err), "invalid distributed snapshot chunk");
            break;
        }
        rc = dist_read_full(upstream->fd, buf, chunk_bytes);
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (fwrite(buf, 1, chunk_bytes, tmp) != chunk_bytes) {
            snprintf(err, sizeof(err), "failed to write worker KV shard temp file");
            break;
        }
        received += chunk_bytes;
    }
    free(buf);

    if (!err[0] && fflush(tmp) != 0) {
        snprintf(err, sizeof(err), "failed to flush worker KV shard restore file");
    }
    if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
        snprintf(err, sizeof(err), "failed to rewind worker KV shard restore file");
    }
    if (!err[0]) {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session =
            dist_worker_get_session_locked(state, session_id, err, sizeof(err));
        if (session &&
            ds4_session_load_layer_payload(session->session,
                                           tmp,
                                           payload_bytes,
                                           tokens,
                                           begin.token_count,
                                           state->layer_start,
                                           state->layer_end,
                                           err,
                                           sizeof(err)) == 0) {
            session->token_hash = token_hash;
            session->token_hash_valid = true;
        } else {
            if (!err[0]) snprintf(err, sizeof(err), "failed to restore worker KV shard");
            if (session) session->token_hash_valid = false;
        }
        pthread_mutex_unlock(&state->mu);
    }

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    free(tokens);

    pthread_mutex_lock(&upstream->write_mu);
    rc = dist_send_snapshot_done(upstream->fd, request_id, err[0] ? 1u : 0u,
                                 err[0] ? err : NULL);
    pthread_mutex_unlock(&upstream->write_mu);
    if (err[0] && received < payload_bytes) return -1;
    return rc;
}

/* =========================================================================
 * Worker Layer Execution
 * ========================================================================= */

static void dist_work_message_free(ds4_dist_work_message *message) {
    if (!message) return;
    free(message->tokens);
    if (message->input_lease)
        ds4_transport_lease_release(message->input_lease);
    else
        free(message->input_hc_wire);
    free(message->route_blob);
    memset(message, 0, sizeof(*message));
}

static int dist_work_message_commit_input_lease(
        ds4_dist_work_message *message) {
    if (!message || !message->input_lease) return 0;
    int rc = dist_mapped_lease_commit_after_gpu_quiesce(
        message->input_lease);
    ds4_transport_lease_release(message->input_lease);
    message->input_lease = NULL;
    message->input_hc_wire = NULL;
    return rc;
}

/* Once a mapped RX lease has been acquired, the descriptor and OOB event are
 * structurally valid and belong to this generation. A semantic rejection must
 * therefore consume/repost that event before notifying the peer; releasing an
 * active RX lease would correctly treat it as an unsafe abort and poison the
 * otherwise healthy link. */
static int dist_worker_reject_received_work(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_work_message *message,
        uint64_t request_id,
        const char *error_message) {
    if (dist_work_message_commit_input_lease(message) != 0) return -1;
    return dist_worker_upstream_send_work_error(upstream, request_id,
                                                 error_message);
}

static int dist_worker_reject_work(
        ds4_dist_worker_upstream *upstream,
        uint32_t unread_bytes,
        ds4_dist_work_message *work,
        uint64_t request_id,
        const char *message) {
    if (unread_bytes != 0 &&
        dist_discard_bytes(upstream->fd, unread_bytes) <= 0)
        return -1;
    work->reject_request_id = request_id;
    snprintf(work->reject_message, sizeof(work->reject_message), "%s", message);
    return DS4_DIST_WORK_RECV_REJECTED;
}

static int dist_worker_reject_typed_work(
        ds4_dist_worker_upstream *upstream,
        uint32_t unread_bytes,
        ds4_dist_work_message *work,
        uint64_t request_id,
        const char *message) {
    if (work && work->v3) {
        /* A malformed v3 WORK cannot be made recoverable with a raw TCP drain:
         * a sequenced descriptor would leave rx_sequence behind, while an OOB
         * sender may already have produced an event. Abandon the generation. */
        shutdown(upstream->fd, SHUT_RDWR);
        errno = EPROTO;
        return -1;
    }
    return dist_worker_reject_work(upstream, unread_bytes, work,
                                   request_id, message);
}

/* Read one v2 WORK frame into typed ownership. The wire order remains exactly
 * fixed | tokens | hidden-state | route, but bulk bytes now flow through the
 * persistent transport rather than a temporary whole-frame allocation. */
static int dist_worker_recv_work_message(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes,
        ds4_dist_work_message *message) {
    memset(message, 0, sizeof(*message));
    message->frame_bytes = bytes;
    message->v3 = dist_transport_uses_v3(upstream->transport);
    const uint32_t fixed_bytes = (uint32_t)sizeof(ds4_dist_work_fixed);
    if (bytes < fixed_bytes) {
        if (message->v3) {
            shutdown(upstream->fd, SHUT_RDWR);
            errno = EPROTO;
            return -1;
        }
        return dist_worker_reject_work(upstream, bytes, message, 0,
                                       "truncated distributed WORK frame");
    }

    ds4_dist_work_fixed wire;
    int rc = dist_read_full(upstream->fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    message->work = wire;
    dist_work_from_wire(&message->work);
    const ds4_dist_work_fixed *work = &message->work;
    const uint64_t request_id =
        dist_u64_from_halves(work->request_hi, work->request_lo);
    const uint64_t session_id =
        dist_u64_from_halves(work->session_hi, work->session_lo);
    uint32_t unread = bytes - fixed_bytes;
    if (message->v3) {
        if (unread < sizeof(ds4_transport_bulk_desc_wire)) {
            shutdown(upstream->fd, SHUT_RDWR);
            errno = EPROTO;
            return -1;
        }
        ds4_transport_bulk_desc_wire desc_wire;
        rc = dist_read_full(upstream->fd, &desc_wire, sizeof(desc_wire));
        if (rc <= 0) return rc == 0 ? 0 : -1;
        unread -= (uint32_t)sizeof(desc_wire);
        if (ds4_transport_bulk_desc_decode(&desc_wire,
                                           &message->bulk_desc) != 0) {
            shutdown(upstream->fd, SHUT_RDWR);
            return -1;
        }
    }

    if (message->v3 && work->route_count != 1u) {
        return dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "distributed v3 currently supports one remote worker link");
    }

    const uint64_t token_bytes_expected =
        (uint64_t)work->n_tokens * sizeof(uint32_t);
    uint64_t body_bytes_expected = token_bytes_expected + work->route_bytes;
    if (!message->v3 ||
        message->bulk_desc.mode == DS4_TRANSPORT_BULK_TCP_INLINE)
        body_bytes_expected += work->input_hc_bytes;
    if (token_bytes_expected > UINT32_MAX ||
        work->token_bytes != (uint32_t)token_bytes_expected ||
        body_bytes_expected != unread) {
        return dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "invalid distributed WORK payload sizes");
    }
    if (work->n_tokens == 0 ||
        work->pos0 > (uint32_t)state->ctx_size ||
        work->n_tokens > (uint32_t)state->ctx_size - work->pos0) {
        return dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "WORK token span exceeds worker context");
    }
    if (work->route_bytes > DS4_DIST_MAX_ROUTE_BYTES) {
        return dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "distributed WORK route metadata is too large");
    }

    const bool input_hc_present =
        (work->flags & DS4_DIST_WORK_F_INPUT_HC) != 0;
    if (!input_hc_present && work->input_hc_bytes != 0) {
        return dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "WORK frame has hidden bytes without input flag");
    }
    if (input_hc_present) {
        const uint64_t hc_values = state->hidden_f32_values;
        const uint64_t expected_values = (uint64_t)work->n_tokens * hc_values;
        const uint32_t input_hc_bits =
            dist_activation_bits_or_default(work->input_hc_bits);
        uint32_t expected_wire_bytes = 0;
        if (!dist_activation_bits_valid(input_hc_bits) ||
            !dist_activation_wire_bytes(input_hc_bits,
                                        expected_values,
                                        &expected_wire_bytes) ||
            work->input_hc_bytes != expected_wire_bytes) {
            return dist_worker_reject_typed_work(
                upstream, unread, message, request_id,
                "input hidden-state size does not match token span");
        }
    }

    if (message->v3) {
        int desc_rc = 0;
        if (work->input_hc_bytes == 0) {
            desc_rc = dist_validate_bulk_none_desc(
                upstream->transport, &message->bulk_desc,
                session_id, request_id);
        } else if (message->bulk_desc.mode ==
                   DS4_TRANSPORT_BULK_TCP_INLINE) {
            desc_rc = ds4_transport_validate_inline_desc(
                &message->bulk_desc,
                DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                session_id, request_id, work->input_hc_bytes,
                dist_activation_bits_or_default(work->input_hc_bits),
                NULL, 0);
        } else if (message->bulk_desc.mode ==
                   DS4_TRANSPORT_BULK_NHI_OOB &&
                   work->route_count == 1u) {
            desc_rc = ds4_transport_validate_oob_desc(
                upstream->transport, &message->bulk_desc,
                DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                session_id, request_id, work->input_hc_bytes,
                dist_activation_bits_or_default(work->input_hc_bits),
                NULL, 0);
        } else {
            desc_rc = -1;
            errno = EPROTO;
        }
        if (desc_rc != 0) {
            shutdown(upstream->fd, SHUT_RDWR);
            return -1;
        }
    }

    message->tokens = malloc((size_t)work->n_tokens * sizeof(message->tokens[0]));
    if (!message->tokens) {
        shutdown(upstream->fd, SHUT_RDWR);
        return -1;
    }
    bool invalid_token = false;
    for (uint32_t i = 0; i < work->n_tokens; i++) {
        uint32_t token_wire = 0;
        rc = dist_read_full(upstream->fd, &token_wire, sizeof(token_wire));
        if (rc <= 0) {
            dist_work_message_free(message);
            return -1;
        }
        const uint32_t token = ntohl(token_wire);
        if (token > (uint32_t)INT_MAX) {
            invalid_token = true;
            message->tokens[i] = 0;
        } else {
            message->tokens[i] = (int)token;
        }
    }
    unread -= work->token_bytes;

    if (!message->v3 && work->input_hc_bytes != 0) {
        message->input_hc_wire = malloc(work->input_hc_bytes);
        if (!message->input_hc_wire) {
            dist_work_message_free(message);
            shutdown(upstream->fd, SHUT_RDWR);
            return -1;
        }
        rc = ds4_transport_recv_bulk(upstream->transport,
                                     message->input_hc_wire,
                                     work->input_hc_bytes);
        if (rc <= 0) {
            dist_work_message_free(message);
            return -1;
        }
        unread -= work->input_hc_bytes;
    }

    if (work->route_bytes != 0) {
        message->route_blob = malloc(work->route_bytes);
        if (!message->route_blob) {
            dist_work_message_free(message);
            shutdown(upstream->fd, SHUT_RDWR);
            return -1;
        }
        rc = dist_read_full(upstream->fd, message->route_blob,
                            work->route_bytes);
        if (rc <= 0) {
            dist_work_message_free(message);
            return -1;
        }
        unread -= work->route_bytes;
    }
    if (message->v3 && invalid_token) {
        int reject_rc = dist_worker_reject_typed_work(
            upstream, unread, message, request_id,
            "WORK token id is outside the model vocabulary");
        if (reject_rc < 0) dist_work_message_free(message);
        return reject_rc;
    }
    if (message->v3 && work->input_hc_bytes != 0) {
        if (message->bulk_desc.mode == DS4_TRANSPORT_BULK_NHI_OOB &&
            dist_activation_bits_or_default(work->input_hc_bits) == 32u &&
            ds4_transport_mapped_leases_supported(upstream->transport)) {
            rc = ds4_transport_rx_lease_acquire(
                upstream->transport, &message->bulk_desc,
                &message->input_lease, NULL, 0);
            if (rc == 0) {
                message->input_hc_wire =
                    ds4_transport_lease_host_ptr(message->input_lease);
            } else if (errno != ENOTSUP) {
                dist_work_message_free(message);
                return -1;
            }
        }
        if (!message->input_lease) {
            message->input_hc_wire = malloc(work->input_hc_bytes);
            if (!message->input_hc_wire) {
                dist_work_message_free(message);
                shutdown(upstream->fd, SHUT_RDWR);
                return -1;
            }
            rc = ds4_transport_recv_bulk_desc(upstream->transport,
                                              &message->bulk_desc,
                                              message->input_hc_wire,
                                              work->input_hc_bytes);
            if (rc <= 0) {
                dist_work_message_free(message);
                return -1;
            }
        }
        if (message->bulk_desc.mode == DS4_TRANSPORT_BULK_TCP_INLINE)
            unread -= work->input_hc_bytes;
    }
    if (unread != 0) {
        dist_work_message_free(message);
        shutdown(upstream->fd, SHUT_RDWR);
        return -1;
    }
    if (invalid_token) {
        dist_work_message_free(message);
        return dist_worker_reject_work(
            upstream, 0, message, request_id,
            "WORK token id is outside the model vocabulary");
    }
    return DS4_DIST_WORK_RECV_READY;
}

static int dist_worker_process_work_message(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        ds4_dist_work_message *message) {
    char err[256];
    ds4_dist_work_fixed work = message->work;
    const uint32_t bytes = message->frame_bytes;
    const uint64_t session_id = dist_u64_from_halves(work.session_hi, work.session_lo);
    const uint64_t request_id = dist_u64_from_halves(work.request_hi, work.request_lo);
    const uint64_t work_prefix_hash = dist_u64_from_halves(work.prefix_hash_hi,
                                                           work.prefix_hash_lo);
    const uint64_t work_result_hash = dist_u64_from_halves(work.result_hash_hi,
                                                           work.result_hash_lo);
    const bool profile = dist_decode_profile_enabled() && work.n_tokens == 1;
    const double total_t0 = profile ? dist_now_sec() : 0.0;
    DIST_DEBUG("worker work request=%llu layers=%u:%u tokens=%u pos=%u flags=0x%x token_bytes=%u input_hc=%u/%ub route_count=%u route_index=%u route_bytes=%u",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               work.flags,
               work.token_bytes,
               work.input_hc_bytes,
               work.input_hc_bits,
               work.route_count,
               work.route_index,
               work.route_bytes);

    const uint32_t descriptor_bytes = message->v3
        ? (uint32_t)sizeof(ds4_transport_bulk_desc_wire) : 0u;
    const uint32_t metadata_bytes = (uint32_t)sizeof(ds4_dist_work_fixed);
    if (bytes < metadata_bytes + descriptor_bytes) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "truncated distributed WORK metadata");
    }
    const uint32_t remaining = bytes - metadata_bytes - descriptor_bytes;
    const uint64_t token_bytes_expected = (uint64_t)work.n_tokens * sizeof(uint32_t);
    uint64_t payload_bytes_expected =
        (uint64_t)work.token_bytes + work.route_bytes;
    if (!message->v3 ||
        message->bulk_desc.mode == DS4_TRANSPORT_BULK_TCP_INLINE)
        payload_bytes_expected += work.input_hc_bytes;
    if ((uint64_t)work.token_bytes != token_bytes_expected ||
        payload_bytes_expected != remaining) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "invalid distributed WORK payload sizes");
    }
    if (work.route_count == 0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "WORK frame is missing distributed route");
    }
    if (work.route_index >= work.route_count) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "invalid distributed WORK route metadata");
    }
    if (work.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: work=%u worker=%u", work.model_id, state->model_id);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if (work.layer_start != state->layer_start || work.layer_end != state->layer_end) {
        snprintf(err, sizeof(err), "worker is assigned layers %u:%u but request asked for %u:%u",
                 state->layer_start, state->layer_end, work.layer_start, work.layer_end);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if ((work.flags & ~DS4_DIST_WORK_F_VALID_MASK) != 0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id, "invalid distributed WORK flags");
    }
    if (ds4_dist_v3_work_caps_validate(work.flags,
                                       upstream->selected_caps,
                                       err,
                                       sizeof(err)) != 0) {
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if (work.n_tokens == 0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id, "WORK frame has no tokens");
    }
    if (work.pos0 > (uint32_t)state->ctx_size ||
        work.n_tokens > (uint32_t)state->ctx_size - work.pos0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "WORK token span exceeds worker context");
    }

    const bool output_logits = (work.flags & DS4_DIST_WORK_F_OUTPUT_LOGITS) != 0;
    const bool input_hc_present = (work.flags & DS4_DIST_WORK_F_INPUT_HC) != 0;
    const bool ack_only = (work.flags & DS4_DIST_WORK_F_ACK_ONLY) != 0;
    if (input_hc_present && work.layer_start == 0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "layer 0 WORK must not provide input hidden-state");
    }
    if (!input_hc_present && work.layer_start != 0) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "nonzero layer WORK requires input hidden-state");
    }
    if (output_logits && !state->has_output) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "worker was not assigned the output head");
    }
    const uint32_t n_layers = (uint32_t)ds4_engine_layer_count(state->engine);
    if (output_logits && work.layer_end + 1u != n_layers) {
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "WORK logits require final transformer layer");
    }

    int *tokens = message->tokens;
    message->tokens = NULL;
    for (uint32_t i = 0; i < work.n_tokens; i++) {
        uint32_t token = (uint32_t)tokens[i];
        if (token > (uint32_t)INT_MAX || token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "WORK token id is outside the model vocabulary");
        }
        tokens[i] = (int)token;
    }
    if (dist_token_hash_update_span(work_prefix_hash, tokens, work.n_tokens) !=
        work_result_hash) {
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "WORK token prefix hash metadata mismatch");
    }

    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t expected_hc_values = (uint64_t)work.n_tokens * hc_values;
    const uint64_t expected_hc_bytes64 = expected_hc_values * sizeof(float);
    if (expected_hc_bytes64 > UINT32_MAX) {
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "distributed hidden-state payload is too large");
    }
    const uint32_t expected_hc_bytes = (uint32_t)expected_hc_bytes64;
    const uint32_t input_hc_bits = dist_activation_bits_or_default(work.input_hc_bits);

    float *input_hc = NULL;
    const void *input_hc_wire = message->input_hc_wire;
    /* Prefix commits carry no hidden payload: the flags still say INPUT_HC
     * (the sender sets it for every mid-stack hop) but there is nothing to
     * read or size-check. */
    const bool frame_spec_commit =
        (work.flags & DS4_DIST_WORK_F_SPEC_COMMIT) != 0;
    if (input_hc_present && !frame_spec_commit) {
        uint32_t expected_hc_wire_bytes = 0;
        if (!dist_activation_bits_valid(input_hc_bits) ||
            !dist_activation_wire_bytes(input_hc_bits,
                                        expected_hc_values,
                                        &expected_hc_wire_bytes)) {
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "invalid distributed activation width");
        }
        if (work.input_hc_bytes != expected_hc_wire_bytes) {
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "input hidden-state size does not match token span");
        }
        if (work.input_hc_bytes != 0 && !input_hc_wire) {
            DIST_DEBUG("worker input hidden read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       -1,
                       work.input_hc_bytes);
            free(tokens);
            return -1;
        }
        DIST_DEBUG("worker input hidden read ok request=%llu bytes=%u bits=%u",
                   (unsigned long long)request_id,
                   work.input_hc_bytes,
                   input_hc_bits);
    } else if (work.input_hc_bytes != 0) {
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "WORK frame has hidden bytes without input flag");
    }
    void *route_blob = message->route_blob;
    message->route_blob = NULL;
    if (work.route_bytes != 0) {
        DIST_DEBUG("worker route read ok request=%llu bytes=%u",
                   (unsigned long long)request_id,
                   work.route_bytes);
    }

    ds4_dist_route_entry current_route;
    ds4_dist_route_entry next_route;
    const bool has_route = work.route_count != 0;
    const bool has_next = has_route && work.route_index + 1u < work.route_count;
    if (has_route) {
        if (!dist_route_validate_blob(route_blob, work.route_bytes, work.route_count,
                                      n_layers,
                                      err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(upstream, message,
                                                    request_id, err);
        }
        if (!dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index, &current_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(upstream, message,
                                                    request_id, err);
        }
        if (current_route.layer_start != work.layer_start ||
            current_route.layer_end != work.layer_end) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "WORK layer range does not match route entry");
        }
        const bool route_output_logits = (current_route.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
        if (route_output_logits != output_logits) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "WORK logits flag does not match route entry");
        }
        if (has_next &&
            !dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index + 1u, &next_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(upstream, message,
                                                    request_id, err);
        }
    }
    if (has_next && output_logits) {
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "non-final route entry requested logits");
    }
    if (has_route && !has_next) {
        ds4_dist_route_return ret;
        if (!dist_route_get_return_target(route_blob, work.route_bytes, work.route_count,
                                          &ret, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(upstream, message,
                                                    request_id, err);
        }
        if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "unsupported final result destination");
        }
    }

    const bool final_ack_only = ack_only && !has_next;
    const bool local_output_logits = output_logits && !has_next && !final_ack_only;
    const bool produce_hidden = !local_output_logits && !final_ack_only;
    const bool spec_verify = (work.flags & DS4_DIST_WORK_F_SPEC_VERIFY) != 0;
    const bool spec_rollback = (work.flags & DS4_DIST_WORK_F_SPEC_ROLLBACK) != 0;
    const bool output_drafts = (work.flags & DS4_DIST_WORK_F_OUTPUT_DRAFTS) != 0;
    const bool output_all_logits = (work.flags & DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS) != 0;
    const bool spec_commit = (work.flags & DS4_DIST_WORK_F_SPEC_COMMIT) != 0;
    const bool spec_exact_required =
        (work.flags & DS4_DIST_WORK_F_SPEC_EXACT_REQUIRED) != 0;
    if (spec_verify && spec_rollback) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "conflicting speculative WORK flags");
    }
    if (spec_commit &&
        (spec_verify || spec_rollback || output_drafts || output_all_logits ||
         spec_exact_required || ack_only || has_next || work.n_tokens == 0u)) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "conflicting speculative WORK flags");
    }
    if ((spec_verify || spec_rollback || output_drafts || output_all_logits ||
         spec_exact_required) &&
        (has_next || !output_logits || ack_only)) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "speculative flags require a final logits span");
    }
    if (spec_exact_required && (!spec_verify || !output_all_logits)) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(
            upstream, request_id,
            "exact speculative WORK requires an all-logits verify span");
    }
    if (spec_verify && work.n_tokens < 2u) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "speculative verify span needs multiple tokens");
    }
    /* Drafts ride either a single-token base decode or, combined with
     * OUTPUT_ALL_LOGITS, a fused verify span (continuation drafts proposed
     * from the last verified row). */
    if (output_drafts && work.n_tokens != 1u && !output_all_logits) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "draft output needs a single-token span");
    }
    const uint32_t vocab = (uint32_t)ds4_engine_vocab_size(state->engine);
    int drafts_max = output_drafts ? ds4_engine_mtp_draft_tokens(state->engine) : 0;
    if (drafts_max < 0) drafts_max = 0;
    if (drafts_max > 16) drafts_max = 16;
    if (output_drafts && drafts_max == 0 && !state->spec_warned) {
        state->spec_warned = true;
        fprintf(stderr,
                "ds4: dist worker: speculative drafts requested but no "
                "enabled support model (--mtp/--dspark) is available; "
                "falling back to per-token decode\n");
    }
    const bool decode2_span =
        output_all_logits && work.n_tokens == 2u && !spec_rollback &&
        !output_drafts;
    const uint32_t result_kind = final_ack_only || spec_commit
        ? DS4_DIST_RESULT_ACK
        : output_drafts && output_all_logits
            ? DS4_DIST_RESULT_LOGITS_NROWS
        : output_drafts
            ? DS4_DIST_RESULT_LOGITS_DRAFTS
            : decode2_span
                ? DS4_DIST_RESULT_LOGITS_DECODE2
                : output_all_logits
                    ? DS4_DIST_RESULT_LOGITS_NROWS
                    : (local_output_logits ? DS4_DIST_RESULT_LOGITS : DS4_DIST_RESULT_HIDDEN_STATE);
    size_t result_bytes = 0;
    if (!(final_ack_only || spec_commit)) {
        if (output_drafts && output_all_logits) {
            result_bytes = (size_t)work.n_tokens * vocab * 4u +
                           4u + 4u * (size_t)drafts_max;
        } else if (output_drafts) {
            result_bytes = (size_t)vocab * 4u + 4u +
                           4u * (size_t)drafts_max;
        } else if (decode2_span) {
            result_bytes = 4u + (size_t)vocab * 4u;
        } else if (output_all_logits) {
            result_bytes = (size_t)work.n_tokens * vocab * 4u;
        } else {
            result_bytes = local_output_logits
                ? (size_t)vocab * 4u
                : (size_t)expected_hc_bytes;
        }
        if (result_bytes > UINT32_MAX) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(
                upstream, request_id,
                "distributed result payload exceeds 32-bit size");
        }
    }
    float *result = result_bytes ? malloc(result_bytes) : NULL;
    ds4_dist_tx_bulk_plan result_tx_plan;
    memset(&result_tx_plan, 0, sizeof(result_tx_plan));
    bool result_mapped = false;
    if (result_bytes && !result) {
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "out of memory allocating distributed result");
    }

    const double decode_t0 = profile ? dist_now_sec() : 0.0;
    bool input_hc_uses_wire = false;
    uint32_t input_hc_decoded_bytes = 0;
    if (input_hc_present && !spec_commit &&
        dist_decode_activation_payload(input_hc_wire,
                                       input_hc_bits,
                                       work.input_hc_bytes,
                                       &input_hc,
                                       &input_hc_decoded_bytes,
                                       &input_hc_uses_wire,
                                       err,
                                       sizeof(err)) != 0) {
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if (input_hc_present && !spec_commit &&
        input_hc_decoded_bytes != expected_hc_bytes) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "decoded input hidden-state size does not match token span");
    }
    const double decode_t1 = profile ? dist_now_sec() : 0.0;

    const uint32_t result_payload_bits =
        result_kind == DS4_DIST_RESULT_HIDDEN_STATE ? input_hc_bits : 32u;
    /* Only the plain hidden-state/logits results ride the bulk descriptor
     * path; the speculative result kinds (drafts, per-row logits, decode2)
     * are control payloads and must not reserve a bulk plan. */
    const bool result_is_bulk =
        result_kind == DS4_DIST_RESULT_HIDDEN_STATE ||
        result_kind == DS4_DIST_RESULT_LOGITS;
    if (!has_next && work.route_count == 1u && result_bytes != 0 &&
        result_is_bulk) {
        int tx_plan_rc = dist_tx_bulk_plan_prepare(
            upstream->transport,
            result_kind == DS4_DIST_RESULT_HIDDEN_STATE
                ? DS4_TRANSPORT_BULK_RESULT_HIDDEN
                : DS4_TRANSPORT_BULK_RESULT_LOGITS,
            0, request_id, (uint32_t)result_bytes, result_payload_bits,
            &result_tx_plan, err, sizeof(err));
        if (tx_plan_rc < 0) {
            if (!input_hc_uses_wire) free(input_hc);
            free(result);
            dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
            free(route_blob);
            free(tokens);
            return -1;
        }
        if (result_tx_plan.lease) {
            free(result);
            result = ds4_transport_lease_host_ptr(result_tx_plan.lease);
            result_mapped = true;
        }
    }

    const double lock_t0 = profile ? dist_now_sec() : 0.0;
    pthread_mutex_lock(&state->mu);
    const double lock_t1 = profile ? dist_now_sec() : 0.0;
    ds4_dist_worker_session *session = dist_worker_get_session_locked(state, session_id, err, sizeof(err));
    if (!session) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0 &&
        ds4_session_layer_slice_reset(session->session, err, sizeof(err)) != 0) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }
    if (spec_rollback) {
        /* Rewind to the frontier captured before the speculative verify
         * span, then let the token-hash block recompute over the truncated
         * timeline so this re-eval span matches the coordinator's prefix. */
        if (!ds4_session_dist_frontier_restore(session->session,
                                               &session->frontier) ||
            !ds4_session_dist_timeline_truncate(session->session, work.pos0)) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            if (!result_mapped) free(result);
            dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "worker speculative frontier restore failed");
        }
        session->token_hash_valid = false;
        session->frontier.valid = false;
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0) {
        session->token_hash = DS4_DIST_TOKEN_HASH_INIT;
        session->token_hash_valid = true;
    } else if (!session->token_hash_valid) {
        const ds4_tokens *timeline = ds4_session_tokens(session->session);
        if (!timeline || timeline->len < 0) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            if (!result_mapped) free(result);
            dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
            free(route_blob);
            free(tokens);
            return dist_worker_reject_received_work(
                upstream, message, request_id,
                "worker session has no token timeline");
        }
        session->token_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
        session->token_hash_valid = true;
    }
    /* A prefix commit intentionally rewinds the timeline, so its incoming
     * hash cannot match the post-verify state; the ack carries the new
     * hash and the next span's prefix check covers any divergence. */
    if (!spec_commit && session->token_hash != work_prefix_hash) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "worker KV prefix hash mismatch");
    }
    if (spec_verify) {
        if (!ds4_session_dist_frontier_snapshot(session->session,
                                                &session->frontier)) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            if (!result_mapped) free(result);
            dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "worker speculative frontier snapshot failed");
        }
    }
    const bool exact_verify_rows = spec_exact_required &&
        ds4_session_dist_spec_exact_verify(session->session, work.n_tokens);
    if (spec_exact_required && !exact_verify_rows) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(
            upstream, request_id,
            "worker exact speculative verify gate is unavailable");
    }
    const double eval_t0 = dist_now_sec();
    int eval_rc = spec_commit
        ? ds4_session_dist_spec_commit_prefix(session->session,
                                              tokens,
                                              work.n_tokens,
                                              work.pos0,
                                              err,
                                              sizeof(err))
        : decode2_span
        ? ds4_session_eval_layer_slice_decode2(session->session,
                                               tokens[0],
                                               tokens[1],
                                               work.pos0,
                                               work.layer_start,
                                               work.layer_end,
                                               input_hc,
                                               input_hc + hc_values,
                                               NULL,
                                               NULL,
                                               true,
                                               (int *)result,
                                               (float *)((uint8_t *)result + 4u),
                                               err,
                                               sizeof(err))
        : output_all_logits
            ? (exact_verify_rows
               ? ds4_session_eval_layer_slice_exact_rows(session->session,
                                                         tokens,
                                                         work.n_tokens,
                                                         work.pos0,
                                                         work.layer_start,
                                                         work.layer_end,
                                                         input_hc,
                                                         NULL,
                                                         result,
                                                         true,
                                                         err,
                                                         sizeof(err))
               : ds4_session_eval_layer_slice_logits_all(session->session,
                                                         tokens,
                                                         work.n_tokens,
                                                         work.pos0,
                                                         work.layer_start,
                                                         work.layer_end,
                                                         input_hc,
                                                         NULL,
                                                         result,
                                                         err,
                                                         sizeof(err)))
            : ds4_session_eval_layer_slice(session->session,
                                           tokens,
                                           work.n_tokens,
                                           work.pos0,
                                           work.layer_start,
                                           work.layer_end,
                                           input_hc,
                                           produce_hidden ? result : NULL,
                                           local_output_logits,
                                           local_output_logits ? result : NULL,
                                           err,
                                           sizeof(err));
    const double eval_t1 = dist_now_sec();
    if (getenv("DS4_DIST_SYNC_TIMING") != NULL && work.n_tokens > 1) {
        fprintf(stderr,
                "ds4: dist sync timing worker eval pos=%u n=%u layers=%u:%u eval=%.1f ms rc=%d\n",
                work.pos0,
                work.n_tokens,
                state->layer_start,
                state->layer_end,
                (eval_t1 - eval_t0) * 1000.0,
                eval_rc);
    }
    if (eval_rc == 0) {
        session->token_hash = work_result_hash;
        session->token_hash_valid = true;
        /* The pre-verify frontier snapshot is consumed by a prefix commit. */
        if (spec_commit) session->frontier.valid = false;
    } else {
        session->token_hash_valid = false;
    }
    pthread_mutex_unlock(&state->mu);
    DIST_DEBUG("worker eval request=%llu layers=%u:%u tokens=%u pos=%u has_next=%d output=%d rc=%d",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               has_next ? 1 : 0,
               local_output_logits ? 1 : 0,
               eval_rc);

    if (eval_rc != 0) {
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(upstream, message,
                                                request_id, err);
    }

    if (dist_work_message_commit_input_lease(message) != 0) {
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return -1;
    }

    int drafts[16];
    int n_drafts = 0;
    if (output_drafts) {
        /* Drafts are advisory: on any drafting failure the result degrades
         * to plain logits and the coordinator runs ordinary decode.  The
         * dispatcher picks the loaded support model (legacy MTP head or the
         * DSpark propose pipeline). */
        if (drafts_max > 0) {
            char draft_err[128];
            int draft_token = tokens[0];
            uint32_t draft_pos = work.pos0;
            if (output_all_logits) {
                /* Continuation drafts for a fused verify span: propose from
                 * the last verified row, whose greedy token becomes the next
                 * base token whenever the whole span is accepted. */
                const float *last_row =
                    result + (uint64_t)(work.n_tokens - 1u) * vocab;
                uint32_t am = 0;
                for (uint32_t vi = 1; vi < vocab; vi++) {
                    if (last_row[vi] > last_row[am]) am = vi;
                }
                draft_token = (int)am;
                draft_pos = work.pos0 + work.n_tokens;
            }
            if (ds4_session_dist_support_draft(session->session,
                                               draft_token,
                                               draft_pos,
                                               drafts,
                                               drafts_max,
                                               &n_drafts,
                                               draft_err,
                                               sizeof(draft_err)) != 0) {
                n_drafts = 0;
                if (getenv("DS4_MTP_SPEC_LOG")) {
                    fprintf(stderr,
                            "ds4: dist worker draft failed: %s\n",
                            draft_err);
                }
            }
        }
    }

    uint32_t payload_bytes = (uint32_t)result_bytes;
    if (output_drafts) {
        /* Always write the draft count: a zero-draft cycle (scheduler or
         * confidence skip, or a failed drafter) must ship count=0, not an
         * uninitialized count under a drafts_max-sized payload.  The drafts
         * trail the logits payload: one row for a base decode, n rows for a
         * fused verify span. */
        const uint64_t logits_payload =
            (uint64_t)(output_all_logits ? work.n_tokens : 1u) * vocab * 4u;
        uint8_t *p = (uint8_t *)result + logits_payload;
        const uint32_t nd = n_drafts > 0 ? (uint32_t)n_drafts : 0u;
        memcpy(p, &nd, 4u);
        p += 4u;
        for (uint32_t i = 0; i < nd; i++) {
            int32_t dv = (int32_t)drafts[i];
            memcpy(p, &dv, 4u);
            p += 4u;
        }
        payload_bytes = (uint32_t)logits_payload + 4u + 4u * nd;
    }

    uint32_t result_wire_bytes = payload_bytes;
    if (result_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        !dist_activation_wire_bytes_from_f32_bytes(input_hc_bits,
                                                   payload_bytes,
                                                   &result_wire_bytes)) {
        if (!input_hc_uses_wire) free(input_hc);
        if (!result_mapped) free(result);
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
        free(route_blob);
        free(tokens);
        return dist_worker_reject_received_work(
            upstream, message, request_id,
            "invalid output hidden-state size");
    }

    ds4_dist_telemetry_fixed telemetry = {
        .layer_start = work.layer_start,
        .layer_end = work.layer_end,
        .route_index = work.route_index,
        .pos0 = work.pos0,
        .n_tokens = work.n_tokens,
        .eval_usec = dist_usec_since(eval_t0, eval_t1),
        .downstream_wait_usec = 0,
        .forward_send_usec = 0,
        .input_bytes = work.token_bytes + work.input_hc_bytes,
        .output_bytes = result_wire_bytes,
    };

    const double send_t0 = profile ? dist_now_sec() : 0.0;
    int send_rc;
    if (has_next) {
        send_rc = dist_forward_work_to_next(upstream,
                                            &next_route,
                                            &work,
                                            tokens,
                                            result,
                                            (uint32_t)result_bytes,
                                            &telemetry,
                                            route_blob);
    } else {
        send_rc = dist_worker_upstream_send_work_result_prepared(
            upstream, request_id, work_result_hash, 0, result_kind,
            result_payload_bits, &telemetry, 1, result, payload_bytes,
            work.route_count == 1u, &result_tx_plan);
    }
    const double send_t1 = profile ? dist_now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: dist decode profile: worker request=%llu pos=%u layers=%u:%u input_decode=%.3fms lock_wait=%.3fms eval=%.3fms send=%.3fms total=%.3fms input=%.2fMiB output=%.2fMiB rc=%d\n",
                (unsigned long long)request_id,
                work.pos0,
                work.layer_start,
                work.layer_end,
                (decode_t1 - decode_t0) * 1000.0,
                (lock_t1 - lock_t0) * 1000.0,
                (eval_t1 - eval_t0) * 1000.0,
                (send_t1 - send_t0) * 1000.0,
                (send_t1 - total_t0) * 1000.0,
                (double)(work.token_bytes + work.input_hc_bytes) / (1024.0 * 1024.0),
                (double)result_wire_bytes / (1024.0 * 1024.0),
                send_rc);
    }
    DIST_DEBUG("worker send complete request=%llu has_next=%d send_rc=%d",
               (unsigned long long)request_id,
               has_next ? 1 : 0,
               send_rc);
    if (send_rc <= 0 && result_tx_plan.prepared)
        dist_tx_bulk_plan_abort(upstream->transport, &result_tx_plan);
    else
        dist_tx_bulk_plan_release(&result_tx_plan);
    if (!input_hc_uses_wire) free(input_hc);
    if (!result_mapped) free(result);
    free(route_blob);
    free(tokens);
    return send_rc;
}

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_work_message message;
    int rc = dist_worker_recv_work_message(state, upstream, bytes, &message);
    if (rc == DS4_DIST_WORK_RECV_REJECTED) {
        rc = dist_worker_upstream_send_work_error(upstream,
                                                   message.reject_request_id,
                                                   message.reject_message);
        dist_work_message_free(&message);
        return rc;
    }
    if (rc <= 0) return rc;
    rc = dist_worker_process_work_message(state, upstream, &message);
    dist_work_message_free(&message);
    return rc;
}

/* =========================================================================
 * Worker Prefetch Queue
 * ========================================================================= */

static void dist_worker_job_free(ds4_dist_worker_job *job) {
    if (!job) return;
    dist_work_message_free(&job->message);
    free(job);
}

static int dist_worker_job_queue_init(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream) {
    memset(q, 0, sizeof(*q));
    q->state = state;
    q->upstream = upstream;
    q->depth = dist_worker_prefetch_depth();
    if (pthread_mutex_init(&q->mu, NULL) != 0) return -1;
    if (pthread_cond_init(&q->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&q->mu);
        return -1;
    }
    if (pthread_cond_init(&q->not_full, NULL) != 0) {
        pthread_cond_destroy(&q->not_empty);
        pthread_mutex_destroy(&q->mu);
        return -1;
    }
    if (pthread_cond_init(&q->idle, NULL) != 0) {
        pthread_cond_destroy(&q->not_full);
        pthread_cond_destroy(&q->not_empty);
        pthread_mutex_destroy(&q->mu);
        return -1;
    }
    return 0;
}

static void dist_worker_job_queue_clear_locked(ds4_dist_worker_job_queue *q) {
    ds4_dist_worker_job *it = q->head;
    q->head = NULL;
    q->tail = NULL;
    q->queued = 0;
    while (it) {
        ds4_dist_worker_job *next = it->next;
        dist_worker_job_free(it);
        it = next;
    }
}

static void dist_worker_job_queue_destroy(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    dist_worker_job_queue_clear_locked(q);
    pthread_mutex_unlock(&q->mu);
    pthread_cond_destroy(&q->idle);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
    pthread_mutex_destroy(&q->mu);
}

static void dist_worker_job_queue_finish(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static void dist_worker_job_queue_cancel(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    q->canceled = true;
    dist_worker_job_queue_clear_locked(q);
    pthread_cond_broadcast(&q->idle);
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static bool dist_worker_job_queue_enqueue(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_job *job) {
    pthread_mutex_lock(&q->mu);
    while (!q->closed && !q->canceled && q->queued >= q->depth) {
        pthread_cond_wait(&q->not_full, &q->mu);
    }
    if (q->closed || q->canceled) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    if (q->tail) q->tail->next = job;
    else q->head = job;
    q->tail = job;
    q->queued++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return true;
}

static ds4_dist_worker_job *dist_worker_job_queue_pop(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    while (!q->head && !q->closed && !q->canceled) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    if (q->canceled || !q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }
    ds4_dist_worker_job *job = q->head;
    q->head = job->next;
    if (!q->head) q->tail = NULL;
    q->queued--;
    q->active++;
    job->next = NULL;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return job;
}

static bool dist_worker_job_queue_complete(
        ds4_dist_worker_job_queue *q,
        int rc) {
    pthread_mutex_lock(&q->mu);
    if (q->active != 0) q->active--;
    if (rc <= 0) {
        q->rc = rc == 0 ? 0 : 1;
        q->closed = true;
        q->canceled = true;
        dist_worker_job_queue_clear_locked(q);
        pthread_cond_broadcast(&q->not_empty);
        pthread_cond_broadcast(&q->not_full);
    }
    const bool keep_running = rc > 0 && !q->canceled;
    pthread_cond_broadcast(&q->idle);
    pthread_mutex_unlock(&q->mu);
    return keep_running;
}

/* A correctly ordered peer may send a control frame immediately after it has
 * received the preceding RESULT, while the eval thread is in the tiny window
 * between completing that write and publishing active=0. Wait out that
 * handoff, but reject a control frame that overtakes work still in the FIFO. */
static bool dist_worker_job_queue_prepare_control(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    if (q->queued != 0) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    int wait_rc = 0;
    while (q->active != 0 && !q->canceled && wait_rc == 0) {
#ifdef DS4_TEST_HOOKS
        q->control_waiters++;
        pthread_cond_broadcast(&q->idle);
#endif
        wait_rc = pthread_cond_wait(&q->idle, &q->mu);
#ifdef DS4_TEST_HOOKS
        q->control_waiters--;
#endif
    }
    const bool ready = wait_rc == 0 && !q->canceled && q->queued == 0;
    pthread_mutex_unlock(&q->mu);
    return ready;
}

static void *dist_worker_prefetch_eval_main(void *arg) {
    ds4_dist_worker_job_queue *q = arg;
    for (;;) {
        ds4_dist_worker_job *job = dist_worker_job_queue_pop(q);
        if (!job) break;
        int rc;
        if (job->message.reject_message[0]) {
            rc = dist_worker_upstream_send_work_error(
                q->upstream,
                job->message.reject_request_id,
                job->message.reject_message);
        } else {
            rc = dist_worker_process_work_message(q->state,
                                                  q->upstream,
                                                  &job->message);
        }
        const uint64_t request_id = dist_u64_from_halves(
            job->message.work.request_hi, job->message.work.request_lo);
        dist_worker_job_free(job);
        if (!dist_worker_job_queue_complete(q, rc)) {
            fprintf(stderr,
                    "ds4: distributed worker: queued WORK request=%llu failed rc=%d: %s\n",
                    (unsigned long long)request_id,
                    rc,
                    strerror(errno));
            shutdown(q->upstream->fd, SHUT_RDWR);
            break;
        }
    }
    return NULL;
}

static int dist_worker_read_loop_prefetch(
        ds4_dist_worker_state *state,
        int fd,
        ds4_transport *transport,
        uint32_t selected_caps) {
    ds4_dist_worker_upstream upstream;
    if (dist_worker_upstream_init(&upstream, state, fd, transport,
                                  selected_caps) != 0)
        return 1;

    ds4_dist_worker_job_queue queue;
    if (dist_worker_job_queue_init(&queue, state, &upstream) != 0) {
        dist_worker_upstream_destroy(&upstream);
        return 1;
    }

    pthread_t eval_tid;
    if (pthread_create(&eval_tid, NULL, dist_worker_prefetch_eval_main, &queue) != 0) {
        dist_worker_job_queue_destroy(&queue);
        dist_worker_upstream_destroy(&upstream);
        return 1;
    }

    int loop_rc = 0;
    fprintf(stderr,
            "ds4: distributed worker: receive prefetch depth %u enabled\n",
            queue.depth);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            ds4_dist_worker_job *job = calloc(1, sizeof(*job));
            if (!job) {
                dist_discard_bytes(fd, bytes);
                /* An immediate error could overtake results for queued WORK.
                 * Make allocation failure terminal instead of reordering the
                 * request/response stream. */
                shutdown(fd, SHUT_RDWR);
                loop_rc = 1;
                break;
            }
            rc = dist_worker_recv_work_message(state, &upstream, bytes,
                                               &job->message);
            if (rc <= 0) {
                fprintf(stderr,
                        "ds4: distributed worker: failed to receive queued WORK rc=%d: %s\n",
                        rc, strerror(errno));
                dist_worker_job_free(job);
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            if (!dist_worker_job_queue_enqueue(&queue, job)) {
                dist_worker_job_free(job);
                loop_rc = 1;
                break;
            }
            continue;
        }
        if (!dist_worker_job_queue_prepare_control(&queue)) {
            /* Processing this request now would overtake WORK that has not
             * completed, or the eval path has already failed. */
            fprintf(stderr,
                    "ds4: distributed worker: control frame %u arrived while WORK was pending\n",
                    type);
            shutdown(fd, SHUT_RDWR);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_SAVE_REQ) {
            rc = dist_worker_handle_snapshot_save(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN) {
            rc = dist_worker_handle_snapshot_load(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    if (loop_rc == 0) dist_worker_job_queue_finish(&queue);
    else dist_worker_job_queue_cancel(&queue);
    pthread_join(eval_tid, NULL);
    if (loop_rc == 0 && queue.rc != 0) loop_rc = 1;
    dist_worker_job_queue_destroy(&queue);
    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static void *dist_worker_data_client_main(void *arg) {
    ds4_dist_data_client_ctx *ctx = arg;
    ds4_dist_worker_state *state = ctx->state;
    int fd = ctx->fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
        ? dist_worker_read_loop(state, fd, NULL, 0)
        : dist_worker_read_loop_prefetch(state, fd, NULL, 0);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: distributed worker: data connection %s:%s closed after error\n",
                peer_host,
                peer_port);
    }

    close(fd);
    return NULL;
}

static void *dist_worker_data_listener_main(void *arg) {
    ds4_dist_worker_state *state = arg;
    int listen_fd = state->listen_fd;
    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "ds4: distributed worker: data accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_data_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            fprintf(stderr, "ds4: distributed worker: out of memory accepting data connection\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_worker_data_client_main, ctx) != 0) {
            fprintf(stderr, "ds4: distributed worker: pthread_create failed for data connection\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

/* =========================================================================
 * Worker Entrypoint
 * ========================================================================= */

static int dist_run_worker(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size) {
    char layer_end[32];
    if (opt->layers.has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", opt->layers.end);

    char err[256];
    const char *listen_host = opt->listen_host;
    int requested_port = opt->listen_port > 0 ? opt->listen_port : 0;
    int listen_fd = dist_open_listener(listen_host, requested_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed worker: %s\n", err);
        return 1;
    }
    int listen_port_i = dist_listener_port(listen_fd);
    if (listen_port_i <= 0) {
        fprintf(stderr, "ds4: distributed worker: could not determine data listener port\n");
        close(listen_fd);
        return 1;
    }
    const uint32_t listen_port = (uint32_t)listen_port_i;

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.layer_start = opt->layers.start;
    state.layer_end = dist_resolved_layer_end(opt, (uint32_t)ds4_engine_layer_count(engine));
    state.has_output = opt->layers.has_output;
    state.ctx_size = ctx_size;
    state.hidden_f32_values = ds4_engine_hidden_f32_values(engine);
    state.listen_fd = listen_fd;
    state.transport_policy = opt->transport;
    state.nhi_device = opt->nhi_device;
    pthread_mutex_init(&state.mu, NULL);

    pthread_t data_tid;
    if (pthread_create(&data_tid, NULL, dist_worker_data_listener_main, &state) != 0) {
        fprintf(stderr, "ds4: distributed worker: pthread_create failed for data listener\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(data_tid);

    fprintf(stderr,
            "ds4: distributed worker: layers %u:%s model_id=%d data_listen=%s:%u connecting to coordinator %s:%d\n",
            opt->layers.start,
            layer_end,
            ds4_engine_model_id(engine),
            listen_host ? listen_host : "*",
            listen_port,
            opt->coordinator_host,
            opt->coordinator_port);

    bool legacy_v2 = opt->transport == DS4_DIST_TRANSPORT_DEFAULT ||
                     opt->transport == DS4_DIST_TRANSPORT_TCP;
    bool force_v3_tcp = false;
    for (;;) {
        int fd = dist_connect_endpoint(opt->coordinator_host, opt->coordinator_port, err, sizeof(err));
        if (fd < 0) {
            fprintf(stderr, "ds4: distributed worker: %s; retrying\n", err);
            dist_sleep_reconnect();
            continue;
        }

        char peer_host[NI_MAXHOST], peer_port[NI_MAXSERV];
        dist_peer_name(fd, peer_host, sizeof(peer_host), peer_port, sizeof(peer_port));
        fprintf(stderr, "ds4: distributed worker: connected to coordinator %s:%s\n", peer_host, peer_port);

        bool retry_legacy = false;
        bool retry_v3_tcp = false;
        uint32_t selected_caps = 0;
        ds4_transport *transport = dist_worker_negotiate_link(
            engine, opt, ctx_size, listen_port, fd, !legacy_v2,
            force_v3_tcp, &retry_legacy, &retry_v3_tcp, &selected_caps,
            err, sizeof(err));
        if (!transport) {
            fprintf(stderr, "ds4: distributed worker: link negotiation failed: %s\n",
                    err[0] ? err : strerror(errno));
            close(fd);
            if (retry_legacy) {
                legacy_v2 = true;
                fprintf(stderr,
                        "ds4: distributed worker: peer rejected v3 HELLO; retrying with exact v2 TCP framing\n");
            } else if (retry_v3_tcp) {
                force_v3_tcp = true;
                fprintf(stderr,
                        "ds4: distributed worker: NHI failed before activation; retrying protocol v3 over TCP\n");
            }
            dist_sleep_reconnect();
            continue;
        }

        /* The first mapped-slot implementation deliberately holds at most
         * one RX lease. Keep parsing and evaluation on one thread until a
         * later multi-lease pipeline can preserve POST_RX order. */
        const bool mapped_slots =
            ds4_transport_mapped_leases_supported(transport) != 0;
        int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH") || mapped_slots
            ? dist_worker_read_loop(&state, fd, transport, selected_caps)
            : dist_worker_read_loop_prefetch(&state, fd, transport,
                                             selected_caps);
        ds4_transport_release(transport);
        close(fd);
        uint32_t dropped_sessions = dist_worker_clear_sessions(&state);
        if (dropped_sessions) {
            fprintf(stderr,
                    "ds4: distributed worker: cleared %u sessions after coordinator disconnect\n",
                    dropped_sessions);
        }
        fprintf(stderr, "ds4: distributed worker: coordinator disconnected%s; reconnecting\n",
                rc ? " after error" : "");
        dist_sleep_reconnect();
    }
}

/* =========================================================================
 * CLI Option Parsing And Public Entrypoint
 * ========================================================================= */

static bool dist_parse_role(const char *s, ds4_distributed_role *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "none")) {
        *out = DS4_DISTRIBUTED_NONE;
        return true;
    }
    if (!strcmp(s, "coordinator")) {
        *out = DS4_DISTRIBUTED_COORDINATOR;
        return true;
    }
    if (!strcmp(s, "worker")) {
        *out = DS4_DISTRIBUTED_WORKER;
        return true;
    }
    return false;
}

static bool dist_parse_u32_component(const char *p, size_t len, uint32_t *out) {
    if (!p || len == 0 || !out) return false;
    char buf[32];
    if (len >= sizeof(buf)) return false;
    memcpy(buf, p, len);
    buf[len] = '\0';

    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(buf, &end, 10);
    if (errno != 0 || end == buf || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool dist_parse_layers(const char *s, ds4_distributed_layers *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "missing layer range");
        return false;
    }

    const char *colon = strchr(s, ':');
    if (!colon || colon == s || colon[1] == '\0') {
        if (errlen) snprintf(err, errlen, "expected A:B or A:output");
        return false;
    }
    if (strchr(colon + 1, ':')) {
        if (errlen) snprintf(err, errlen, "layer range has too many ':' separators");
        return false;
    }

    ds4_distributed_layers parsed = {0};
    if (!dist_parse_u32_component(s, (size_t)(colon - s), &parsed.start)) {
        if (errlen) snprintf(err, errlen, "invalid start layer in %s", s);
        return false;
    }

    const char *end = colon + 1;
    if (!strcmp(end, "output")) {
        parsed.end = UINT32_MAX;
        parsed.has_output = true;
    } else {
        if (!dist_parse_u32_component(end, strlen(end), &parsed.end)) {
            if (errlen) snprintf(err, errlen, "invalid end layer in %s", s);
            return false;
        }
        if (parsed.end < parsed.start) {
            if (errlen) snprintf(err, errlen, "layer range end precedes start in %s", s);
            return false;
        }
    }

    parsed.set = true;
    *out = parsed;
    return true;
}

static const char *dist_cli_need_arg(
        int *index,
        int argc,
        char **argv,
        const char *arg,
        char *err,
        size_t errlen) {
    if (!index || !argv || *index + 1 >= argc) {
        if (errlen) snprintf(err, errlen, "%s requires an argument", arg);
        return NULL;
    }
    return argv[++*index];
}

static bool dist_cli_parse_port(const char *s, const char *arg, int *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a TCP port", arg);
        return false;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v <= 0 || v > 65535) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", arg, s);
        return false;
    }
    *out = (int)v;
    return true;
}

bool ds4_dist_enabled(const ds4_dist_options *opt) {
    return opt && opt->role != DS4_DISTRIBUTED_NONE;
}

ds4_dist_options *ds4_dist_options_create(void) {
    return calloc(1, sizeof(ds4_dist_options));
}

void ds4_dist_options_free(ds4_dist_options *opt) {
    free(opt);
}

void ds4_dist_usage(FILE *fp) {
    fprintf(fp,
        "  --role ROLE\n"
        "      Distributed role: coordinator or worker.\n"
        "  --layers A:B\n"
        "      Inclusive distributed layer slice, e.g. 10:20 or 21:output.\n"
        "  --listen HOST PORT\n"
        "      Coordinator TCP listen address. Workers may later use it to force their data listener.\n"
        "  --coordinator HOST PORT\n"
        "      Coordinator TCP address for --role worker.\n"
        "  --dist-prefill-chunk N\n"
        "      Coordinator prefill pipeline chunk size. Default: session cap, normally 4096.\n"
        "      Non-default values are experimental and can change logits unless validated.\n"
        "  --dist-prefill-window N\n"
        "      Coordinator max end-to-end prefill chunks in flight. Default: workers+2, capped at 8.\n"
        "  --dist-activation-bits N\n"
        "      Coordinator hidden-state transport width: 32, 16, or 8. Default: 32.\n"
        "  --dist-replay-check\n"
        "      Coordinator diagnostic: reset and replay the prompt, then compare logits.\n"
        "  --debug\n"
        "      Print coordinator route/debug logs. Workers keep their normal logs without this.\n"
    );
}

ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    if (!arg) return DS4_DIST_CLI_NOT_MATCHED;
    if (!strcmp(arg, "--role")) {
        const char *role = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!role) return DS4_DIST_CLI_ERROR;
        if (!opt || !dist_parse_role(role, &opt->role)) {
            if (errlen) snprintf(err, errlen,
                                 "invalid distributed role: %s (valid roles: none, coordinator, worker)",
                                 role);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--layers")) {
        const char *layers = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!layers) return DS4_DIST_CLI_ERROR;
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_parse_layers(layers, &opt->layers, err, errlen)) {
            char detail[160];
            if (errlen && err[0] != '\0') {
                snprintf(detail, sizeof(detail), "%s", err);
                snprintf(err, errlen, "invalid --layers %s: %s", layers, detail);
            }
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--listen")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->listen_host || opt->listen_port) {
            if (errlen) snprintf(err, errlen, "specify --listen only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->listen_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->listen_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--coordinator")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "specify --coordinator only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->coordinator_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->coordinator_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-chunk")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_chunk, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-window")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_window, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->prefill_window > 64u) {
            if (errlen) snprintf(err, errlen, "%s must be <= 64", arg);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-activation-bits")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t bits = 0;
        if (!dist_parse_positive_u32(value, arg, &bits, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_activation_bits_valid(bits)) {
            if (errlen) snprintf(err, errlen, "%s must be 32, 16, or 8", arg);
            return DS4_DIST_CLI_ERROR;
        }
        opt->activation_bits = bits;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-transport")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->transport != DS4_DIST_TRANSPORT_DEFAULT) {
            if (errlen) snprintf(err, errlen, "specify --dist-transport only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!strcmp(value, "auto")) opt->transport = DS4_DIST_TRANSPORT_AUTO;
        else if (!strcmp(value, "tcp")) opt->transport = DS4_DIST_TRANSPORT_TCP;
        else if (!strcmp(value, "nhi")) opt->transport = DS4_DIST_TRANSPORT_NHI;
        else {
            if (errlen) snprintf(err, errlen,
                                 "invalid distributed transport: %s (valid: tcp, auto, nhi)",
                                 value);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-nhi-device")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->nhi_device) {
            if (errlen) snprintf(err, errlen, "specify --dist-nhi-device only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!value[0]) {
            if (errlen) snprintf(err, errlen, "--dist-nhi-device requires a nonempty path");
            return DS4_DIST_CLI_ERROR;
        }
        opt->nhi_device = value;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-replay-check")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->replay_check = true;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--debug")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->debug = true;
        return DS4_DIST_CLI_MATCHED;
    }
    return DS4_DIST_CLI_NOT_MATCHED;
}

static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen) {
    if (!opt) {
        if (errlen) snprintf(err, errlen, "missing distributed options");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_NONE) {
        if (opt->layers.set || opt->listen_host || opt->listen_port ||
            opt->coordinator_host || opt->coordinator_port ||
            opt->prefill_chunk != 0 || opt->prefill_window != 0 ||
            opt->activation_bits != 0 ||
            opt->transport != DS4_DIST_TRANSPORT_DEFAULT ||
            opt->nhi_device) {
            if (errlen) snprintf(err, errlen, "distributed options require --role coordinator or --role worker");
            return 1;
        }
        return 0;
    }

    if (!opt->layers.set) {
        if (errlen) snprintf(err, errlen, "--role %s requires --layers", dist_role_name(opt->role));
        return 1;
    }
    if (opt->prefill_window > 64u) {
        if (errlen) snprintf(err, errlen, "--dist-prefill-window must be <= 64");
        return 1;
    }
    if (opt->activation_bits != 0 && !dist_activation_bits_valid(opt->activation_bits)) {
        if (errlen) snprintf(err, errlen, "--dist-activation-bits must be 32, 16, or 8");
        return 1;
    }
    if (opt->transport < DS4_DIST_TRANSPORT_DEFAULT ||
        opt->transport > DS4_DIST_TRANSPORT_NHI) {
        if (errlen) snprintf(err, errlen, "invalid distributed transport selection");
        return 1;
    }
    if (opt->transport == DS4_DIST_TRANSPORT_NHI && !opt->nhi_device) {
        if (errlen) snprintf(err, errlen, "--dist-transport nhi requires --dist-nhi-device PATH");
        return 1;
    }
    if (opt->transport == DS4_DIST_TRANSPORT_TCP && opt->nhi_device) {
        if (errlen) snprintf(err, errlen, "--dist-nhi-device cannot be used with --dist-transport tcp");
        return 1;
    }
#ifndef __linux__
    if (opt->transport == DS4_DIST_TRANSPORT_NHI) {
        if (errlen) snprintf(err, errlen, "--dist-transport nhi is available only on Linux");
        return 1;
    }
#endif

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!opt->listen_host || opt->listen_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role coordinator requires --listen HOST PORT");
            return 1;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "--role coordinator must not use --coordinator");
            return 1;
        }
        return 0;
    }

    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        if (!opt->coordinator_host || opt->coordinator_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role worker requires --coordinator HOST PORT");
            return 1;
        }
        if (opt->prefill_chunk != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-chunk requires --role coordinator");
            return 1;
        }
        if (opt->prefill_window != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-window requires --role coordinator");
            return 1;
        }
        if (opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "--dist-activation-bits requires --role coordinator");
            return 1;
        }
        return 0;
    }

    if (errlen) snprintf(err, errlen, "invalid distributed role");
    return 1;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    if (dist_validate_options(opt, err, errlen) != 0) return 1;
    if (opt && opt->replay_check && opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "--dist-replay-check requires --role coordinator");
        return 1;
    }
    if (engine && opt) {
        engine->distributed = *opt;
        if (ds4_dist_enabled(opt)) {
            engine->load_slice = true;
            engine->load_layer_start = opt->layers.start;
            engine->load_layer_end = opt->layers.has_output ? UINT32_MAX : opt->layers.end;
            engine->load_output = opt->layers.has_output;
        }
    }
    return 0;
}

static int dist_validate_layers_for_model(const ds4_dist_options *opt, uint32_t n_layers, char *err, size_t errlen) {
    if (!opt || opt->role == DS4_DISTRIBUTED_NONE || !opt->layers.set) return 0;
    if (n_layers == 0) {
        if (errlen) snprintf(err, errlen, "model reports no layers");
        return 1;
    }

    const uint32_t last = n_layers - 1u;
    if (opt->layers.start > last) {
        if (errlen) snprintf(err, errlen, "layer range starts past final model layer %u", last);
        return 1;
    }
    if (!opt->layers.has_output && opt->layers.end > last) {
        if (errlen) snprintf(err, errlen, "layer range ends past final model layer %u", last);
        return 1;
    }
    if (opt->role == DS4_DISTRIBUTED_COORDINATOR && opt->layers.start != 0) {
        if (errlen) snprintf(err, errlen, "coordinator layer range must start at layer 0");
        return 1;
    }
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    if (!engine || !opt) {
        fprintf(stderr, "ds4: distributed runtime requires an open engine and options\n");
        return 1;
    }
    char err[256];
    if (dist_validate_options(opt, err, sizeof(err)) != 0 ||
        dist_validate_layers_for_model(opt, (uint32_t)ds4_engine_layer_count(engine), err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        return dist_run_coordinator(engine, opt, gen);
    }
    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        return dist_run_worker(engine, opt, gen ? gen->ctx_size : 0);
    }

    fprintf(stderr, "ds4: distributed runtime requested without a distributed role\n");
    return 1;
}
