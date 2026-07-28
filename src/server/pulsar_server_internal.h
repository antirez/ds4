/* pulsar_server_internal.h — internal shared declarations for the server sources.
 * Produced by the multi-TU split of pulsar_server.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_SERVER_INTERNAL_H
#define PULSAR_SERVER_INTERNAL_H

#include "pulsar.h"
#include "pulsar_help.h"
#include "pulsar_kvstore.h"
#include "rax.h"

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * GPU worker.  The worker owns the pulsar_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>


/* Build-time version string (Makefile passes -DDS4_VERSION_STR from
 * `git describe`); fall back to "unknown" for non-Makefile/ad-hoc builds. */
#ifndef PULSAR_VERSION_STR
#define PULSAR_VERSION_STR "unknown"
#endif

/* ---- shared macros ---- */



#define PULSAR_SERVER_IO_TIMEOUT_SEC 10
#define PULSAR_SERVER_SEND_STALL_TIMEOUT_MS 2000
/* Trusted-LAN posture, but a single stuck or hostile peer must not exhaust
 * threads (one thread per connection) or hold a socket open forever while
 * trickling bytes (slowloris). The cap bounds concurrent client threads;
 * the read deadline bounds the total time a request may take to ARRIVE
 * (generation time is not counted - streaming responses run as long as the
 * model needs). */
#define PULSAR_SERVER_MAX_CLIENTS 64
#define PULSAR_SERVER_REQUEST_READ_DEADLINE_SEC 30

/* Multi-session serving increment 2: the worker steps each job as a resumable
 * state machine in bounded quanta instead of running it to completion. A
 * decode quantum yields back to the worker loop once it has emitted at least
 * this many tokens — the bound is checked between sampling iterations, so one
 * speculative burst (up to 17 accepted tokens) can overshoot it (a prefill
 * quantum is one engine chunk, bounded by the engine's chunked-prefill
 * machinery). With one slot the quanta run back-to-back and behavior is
 * identical to run-to-completion; increment 3 interleaves slots at these
 * boundaries. */
#define PULSAR_SERVER_DECODE_QUANTUM_TOKENS 16

/* Ceiling for bytes queued in a slot_writer for a client that stops reading.
 * The stall timeout is the real slow-client guard; this cap only bounds worst
 * case memory if a client keeps draining just enough to defeat it. */
#define PULSAR_SERVER_WRITER_MAX_PENDING_BYTES (64u * 1024u * 1024u)


/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256


#define PULSAR_DSML "｜DSML｜"
#define PULSAR_DSML_SHORT "DSML｜"
#define PULSAR_TOOL_CALLS_START "<" PULSAR_DSML "tool_calls>"
#define PULSAR_TOOL_CALLS_END "</" PULSAR_DSML "tool_calls>"
#define PULSAR_INVOKE_START "<" PULSAR_DSML "invoke"
#define PULSAR_INVOKE_END "</" PULSAR_DSML "invoke>"
#define PULSAR_PARAM_START "<" PULSAR_DSML "parameter"
#define PULSAR_PARAM_END "</" PULSAR_DSML "parameter>"
#define PULSAR_TOOL_CALLS_START_SHORT "<" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_TOOL_CALLS_END_SHORT "</" PULSAR_DSML_SHORT "tool_calls>"
#define PULSAR_INVOKE_START_SHORT "<" PULSAR_DSML_SHORT "invoke"
#define PULSAR_INVOKE_END_SHORT "</" PULSAR_DSML_SHORT "invoke>"
#define PULSAR_PARAM_START_SHORT "<" PULSAR_DSML_SHORT "parameter"
#define PULSAR_PARAM_END_SHORT "</" PULSAR_DSML_SHORT "parameter>"


/* =========================================================================
 * Tool Call Text Memory.
 * =========================================================================
 *
 * The model speaks DSML, while OpenAI and Anthropic clients round-trip tool
 * calls as JSON.  Re-rendering that JSON is not always the same byte sequence:
 * clients may preserve, sort, or rebuild object keys differently.  Tool call
 * ids are the bridge between both worlds.  For every generated tool call we
 * remember the exact DSML block sampled by the model under a random id.  When
 * the client later sends the same id back in conversation history, we replay
 * the sampled DSML verbatim and keep the KV cache aligned with the live model
 * state.
 */

#define PULSAR_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define PULSAR_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)


/* =========================================================================
 * KV Cache.
 * =========================================================================
 *
 * The server has one live GPU session.  We persist reusable DS4 session
 * snapshots when a cold prompt reaches a useful prefix, when a long continued
 * conversation has grown far enough, and when a request evicts the live session.
 * The cache key is the SHA1 of the rendered byte prefix.  The payload still
 * stores exact token IDs and graph state; the filename only selects a checkpoint
 * whose decoded transcript bytes are a prefix of the next rendered request.
 *
 * Files are loaded with plain read/write I/O into the existing graph tensors;
 * mmap is deliberately avoided here so cache restore cannot add more VM
 * mappings to a process that already maps a very large GGUF.
 *
 * Stores are created only when the live graph is already at the checkpoint we
 * want to persist.  For long cold prompts this means prefill reaches the stable
 * boundary first, writes that prefix, and then continues with the suffix.  We
 * never roll the session backward just to build a disk cache entry: that would
 * turn cache population into a second hidden prefill.
 *
 * File layout:
 *
 *   "KVC" version
 *   quant bits, save reason, token count, hit count, context size
 *   creation time, last-used time, payload byte count
 *   rendered text byte count + rendered text for human inspection
 *   DS4 engine payload written by pulsar_session_save_payload()
 *   optional tool-id map section
 *
 * The filename is SHA1(cache text bytes), not SHA1(token ids).  For ordinary
 * checkpoints the cache text is the rendered token prefix.  For live hidden
 * state it can instead be the client-visible transcript: the payload still
 * contains sampled reasoning KV, but the lookup key must be what the client can
 * replay after a process restart or session switch.
 *
 * The optional tool-id map is not part of model state, but it is needed to
 * render future client JSON back to the exact DSML sampled by the model.  We
 * persist only mappings whose DSML block appears in the saved cache text.
 */

#define KV_CACHE_FIXED_HEADER PULSAR_KVSTORE_FIXED_HEADER
#define KV_CACHE_HIT_HALF_LIFE_SECONDS PULSAR_KVSTORE_HIT_HALF_LIFE_SECONDS
#define KV_EXT_TOOL_MAP PULSAR_KVSTORE_EXT_TOOL_MAP
#define KV_EXT_RESPONSES_VISIBLE PULSAR_KVSTORE_EXT_RESPONSES_VISIBLE
#define KV_EXT_THINKING_VISIBLE PULSAR_KVSTORE_EXT_THINKING_VISIBLE
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u


/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical DSML block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

/* ---- shared types ---- */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

typedef struct server server;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_dsml;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *tool_namespace;
    /* Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    char **prop;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

/* Per-response timing metrics, surfaced in the additive "timings" object that
 * sits next to "usage" on the OpenAI chat/completions response (and the final
 * include_usage SSE chunk). Filled once by gen_step_finish from counters the
 * worker already keeps (g->t0/decode_t0/first_token_t, prompt/completion
 * counts, per-session DSpark deltas) — there is NO hot-path work here, and
 * these fields never influence sampling/rng/logits. All rates are derived in
 * the JSON emitter (guarded divisions), so a zero denominator omits a rate
 * rather than emitting NaN/inf. */
typedef struct {
    bool valid;
    double ttft_s;       /* wall time from request start to first emitted token */
    double prefill_s;    /* wall time spent in prefill (request start -> decode start) */
    double decode_s;     /* wall time spent decoding (decode start -> finish) */
    int prompt_n;        /* total prompt tokens (prefill target) */
    int cached_n;        /* prompt tokens served from a cache (<= prompt_n) */
    int decode_n;        /* completion tokens emitted */
    bool spec_active;    /* DSpark speculative decode ran this request */
    uint64_t spec_accepted; /* accepted draft tokens (this request) */
    uint64_t spec_draft;    /* proposed/verified draft tokens (this request) */
    uint64_t spec_drafts;   /* draft rounds (this request) */
    uint64_t spec_gen;      /* tokens emitted by the spec loop (this request) */
} req_timings;

typedef struct {
    req_kind kind;
    api_style api;
    pulsar_tokens prompt;
    char *model;
    bool model_from_request;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    tool_schema_orders tool_orders;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    /* Presence flags: true iff the CLIENT sent the parameter in the request
     * body. request_init() fills the value fields with engine defaults at
     * parse time, so the values alone cannot distinguish "explicitly 1.0"
     * from "absent" — downstream policy (e.g. think-mode defaults in
     * generate.c) must consult these and default only what is absent.
     * Zeroed by request_init's memset; set only in api_parse.c. */
    bool has_temperature;
    bool has_top_k;
    bool has_top_p;
    bool has_min_p;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    int cache_read_tokens;
    int cache_write_tokens;
    req_timings timings;
    pulsar_think_mode think_mode;
    bool has_tools;
    /* tool_choice="required" (OpenAI) / {"type":"any"|"tool"} (Anthropic):
     * force a tool call. The prompt is prefilled into an open DSML tool_calls
     * block (thinking skipped) and generate_job seeds the output with the
     * SAME opener so the model must complete an invoke. forced_tool_name is
     * set for Anthropic {"type":"tool","name":X}: the opener then includes
     * the named invoke so the model can only fill in the parameters. */
    bool force_tool_call;
    char *forced_tool_name;
    bool prompt_preserves_reasoning;
    /* For /v1/responses: emit reasoning_summary_* events / fields only when the
     * client opted in via reasoning.summary. Other APIs leave this false; the
     * field is ignored on those code paths. */
    bool reasoning_summary_emit;
    /* Responses continuation contract:
     *
     * A live Responses tool loop is not a normal "new prompt with a long
     * prefix" request.  The protocol gives tool outputs a call_id that binds
     * them to a prior assistant tool call.  If that call_id is still known in
     * memory, the live KV is the authoritative prefix, including any hidden
     * thinking that the client did not replay.  These fields carry the parsed
     * evidence needed by generate_job() to append only the new suffix.
     *
     * A tool-output-only request has no stateless prefix to match.  If the live
     * call_id binding is gone by the time the worker executes it, DS4 must ask
     * for a full replay rather than cold-prefilling a prompt that starts with a
     * naked tool result.  Similarly, if live state is gone, a reasoning-mode
     * tool replay must contain the prior reasoning item (or an equivalent
     * opaque reasoning state from a future implementation). */
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    tool_replay_stats tool_replay;
} request;

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for protocol-specific DSML stream projections.  The model
 * still samples DSML; these states only translate already-sampled bytes into
 * OpenAI / Anthropic wire events while final parsing remains authoritative. */
typedef struct {
    dsml_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_reasoning;
    bool sent_content;
    openai_tool_stream tool;
} openai_stream;

typedef enum {
    DSML_DECODE_OUTSIDE,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} dsml_syntax;

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string;
    bool json_escaped;
} dsml_decode_tracker;

typedef enum {
    RESP_STREAM_THINKING,
    RESP_STREAM_TEXT,
    RESP_STREAM_SUPPRESS,
} responses_stream_mode;

typedef struct {
    responses_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool reasoning_item_opened;
    bool reasoning_item_closed;
    bool reasoning_summary_started;
    bool reasoning_closed_naturally;
    bool message_item_opened;
    bool message_text_part_open;
    bool message_item_closed;
    bool reasoning_emitted_any;
    bool message_emitted_any;
    buf reasoning_text;
    buf message_text;
    char response_id[40];
    char reasoning_id[40];
    char message_id[40];
    int reasoning_index;   /* output_index of the reasoning item (0 if present) */
    int message_index;     /* output_index of the assistant message item */
    int next_output_index; /* monotonic counter for upcoming output items */
    int sequence;          /* monotonic per-event sequence_number Codex consumes */
} responses_stream;

/* Item identity per tool call must be stable across added/done/completed. */
typedef struct {
    char fc_id[40];
    char call_id[64];
    bool is_custom;
    int output_index;
} responses_tool_item;

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_TOOL,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
    ANTH_BLOCK_TOOL,
} anthropic_block_type;

typedef struct {
    dsml_tool_stream_state state;
    const dsml_syntax *syn;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} anthropic_tool_stream;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_thinking;
    bool sent_text;
    anthropic_tool_stream tool;
} anthropic_stream;

typedef struct job job;

/* ---- deferred slot writer (multi-session increment 2) ----
 *
 * The GPU worker must never sleep in poll() waiting for a slow client while it
 * could be advancing the model. While a job is bound to a slot, the worker
 * installs a slot_writer for the job's (non-blocking) socket: send_all() on
 * that fd becomes best-effort non-blocking — bytes that do not fit in the
 * socket buffer are queued in order and flushed at every quantum boundary,
 * then drained fully before the job is signalled done. Client threads never
 * install a writer, so their send_all() keeps the bounded-blocking behavior.
 * Failure semantics match the blocking path: a hard socket error fails
 * immediately, and a peer that accepts no bytes for
 * PULSAR_SERVER_SEND_STALL_TIMEOUT_MS (or overflows the pending cap) fails the
 * stream, which the generation loop reports exactly as before. */
typedef struct {
    int fd;
    buf pending;                 /* accepted-but-unsent bytes, in wire order */
    size_t off;                  /* consumed prefix of pending */
    long long stall_deadline_ms; /* 0 = disarmed (nothing pending) */
    bool failed;
} slot_writer;

void slot_writer_init(slot_writer *w, int fd);
void slot_writer_install(slot_writer *w);   /* thread-local; NULL uninstalls */
bool slot_writer_flush(slot_writer *w);     /* non-blocking best effort */
bool slot_writer_drain(slot_writer *w);     /* blocking, stall-timeout bounded */
void slot_writer_free(slot_writer *w);

typedef pulsar_kvstore_entry kv_entry;

typedef pulsar_kvstore_options kv_cache_options;

typedef pulsar_kvstore kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    /* Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /* Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    /* Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /* Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

/* ---- Session pool (multi-session serving, increment 1) ----
 *
 * PROCESS-GLOBAL CUDA STATE AUDIT (Tier 1 §1.5 — must-do, read before growing
 * the pool or adding GPU threads).
 *
 * All GPU work in this server runs on the SINGLE worker thread (worker_main in
 * generate.c). Client threads only parse HTTP and block on a per-job condvar
 * until the worker finishes (http_server.c handle path); they never touch a
 * pulsar_session_* or pulsar_gpu_* entry point. Verified by grepping every
 * pulsar_session_/pulsar_gpu_ call site under src/server: all of them are reached
 * only from the generation state machine (gen_* in generate.c) driven by
 * worker_main (or from cli_main startup/shutdown, before the worker starts and
 * after it joins) — with two deliberate exceptions, both plain loads of data
 * immutable after startup (no CUDA behind them): http_server.c reads
 * pulsar_session_ctx(server.sess) on client threads, and client paths read the
 * model id via server_model_id_from_engine (pulsar_engine_model_id, a static
 * shape constant). Nothing else: /metrics in particular makes NO engine
 * calls — the worker publishes per-slot KV positions and the spec-decode
 * counters into plain server fields under mu (m_slot_pos/m_slot_ctx/m_spec,
 * server_publish_metrics_snapshot in generate.c, refreshed at bind time and
 * once per quantum) and send_metrics reads only those snapshots. No CUDA
 * call is made off the worker thread. This is a correctness invariant, not
 * an accident.
 *
 * It matters because the CUDA layer keeps process-global, NON-thread-safe state
 * that all sessions share:
 *   - g_cublas / g_cublaslt handles (pulsar_cuda_runtime.cu, pulsar_cuda_matmul.cu),
 *     bound to cudaStreamPerThread;
 *   - the MXFP8 weight map g_fp8_mx_by_offset (std::unordered_map) plus its
 *     direct-mapped front cache fc_off/fc_ptr (pulsar_cuda_matmul.cu);
 *   - the function-local static lt_shape_cache (pulsar_cuda_matmul.cu);
 *   - the determinism setting CUBLASLT_REDUCTION_SCHEME_NONE (pulsar_cuda_matmul.cu).
 *
 * Because these are shared and unlocked, Tier 1 keeps ONE GPU worker thread and
 * multiplexes sessions by time-slicing on that thread. Tier 2 must NOT naively
 * spawn a second GPU thread: cudaStreamPerThread would give it a distinct
 * stream but it would still race on g_cublas/g_cublaslt and the weight/shape
 * caches. Any future concurrency stays on the single GPU lane (batched kernels),
 * not multiple GPU threads, until these globals are made per-context.
 *
 * Increment 1 was pure structural plumbing (pool of capacity 1). Increment 2
 * made the generation path re-entrant: each job runs as a per-slot resumable
 * state machine (gen_state in generate.c) that the worker steps in bounded
 * quanta — one prefill chunk, or up to PULSAR_SERVER_DECODE_QUANTUM_TOKENS decode
 * tokens, per step. All GPU work still happens on the single worker thread.
 * Increment 3 adds the scheduler: the worker binds queued jobs to free slots
 * (FIFO, warmest-prefix slot choice, lazily provisioning extra slots under the
 * KV admission budget) and round-robins one quantum at a time over the bound
 * slots. Increment 4 adds LRU eviction: when the queue head cannot be placed
 * cleanly (no fitting free slot, or only a warm slot it would clobber) and
 * provisioning was refused by a constraint eviction can relieve (full pool /
 * full admission ledger — deliberately NOT the MemAvailable floor, which
 * freed CUDA memory does not promptly move), the worker evicts the
 * least-recently-serviced IDLE slot — snapshot to the disk kv cache, free
 * the session, release its ACTUAL bytes from the ledger — and provisions in
 * its place (see the increment-4 block in generate.c; slot 0 is pinned).
 * Batched decode is a later increment (Tier 2). */

/* Pool capacity (increment 3). Slot 0 is provisioned at startup with the
 * configured --ctx-size; the rest are provisioned lazily, only when a job
 * arrives while every provisioned slot is busy (or would clobber another
 * conversation's warm KV) AND the packed-KV admission budget still has room.
 * A single client therefore always runs on slot 0, byte-identical to the
 * increment-2 single-session server. */
#define PULSAR_SESSION_POOL_CAP 5

/* Default context for lazily provisioned secondary slots (plan Tier 1 §1.4:
 * keep the default per-session context far below the lone-session maximum;
 * compressed-KV cost scales with ctx, so concurrency is bounded by the sum of
 * context sizes). A request that needs more than this gets a slot sized to
 * its need (capped at slot 0's ctx), admission permitting. */
#define PULSAR_SERVER_EXTRA_SLOT_CTX_TOKENS 65536

/* The rendered-prompt BOS marker. One definition for every render site AND
 * the startup trivial-match-threshold derivation (cli_main.c), so the
 * derived threshold can never silently drift from what rendered prompts
 * actually begin with. (Unit tests keep independent string literals on
 * purpose — they pin the wire bytes, not this macro.) */
#define PULSAR_SERVER_RENDER_BOS "<｜begin▁of▁sentence｜>"

/* Slot-routing trivial-match allowance (task #30, 2026-07-16). The router's
 * choose-vs-provision gate treats a candidate slot's common token prefix as
 * TRIVIAL — "just the shared rendered-template header, not a warm
 * continuation" — below a threshold of
 *     tokens(BOS + think-max preamble) + this allowance,
 * measured once per model at startup (cli_main.c). The derived part is the
 * largest template-injected text two UNRELATED conversations can share; the
 * allowance covers incidental natural-language prologue overlap between
 * distinct conversations (measured 3–8 tokens beyond the header across real
 * conversation pairs in the task-#24 bounce repro — 64 gives ~8x margin)
 * while staying an order of magnitude below any warm state worth a slot:
 * the 512-token disk-snapshot floor (--kv-cache-min-tokens) and the
 * multi-thousand-token preambles the session pool exists for. Used only by
 * the routing decision (server_slot_match_is_trivial); prefill reuse of a
 * chosen slot still honors arbitrarily short common prefixes. */
#define PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS 64

/* Admission-control budget (Tier 1 §1.4). GB10 unified memory is ~121 GiB
 * usable; weights are queried at runtime (pulsar_engine_weights_resident_bytes).
 * The overhead reserve is the fixed process footprint measured on the GB10,
 * independent of session count. Re-measured 2026-07-15 over three clean
 * server restarts (production v5mx gguf, ctx=98304, prefill_chunk=2048,
 * drop_caches before each load; MemAvailable deltas net of weights_resident
 * 85.04 GiB and slot 0's 2.68 GiB ledgered cost):
 *   - startup component (CUDA context, GPU page tables, pinned staging
 *     buffers): 8.54 / 9.87 / 9.41 GiB;
 *   - lazy first-request component (cuBLASLt workspaces at ~32 MiB/GEMM,
 *     FP8 workspaces, MXFP4 expert staging, GEMV activation buffers):
 *     8.71 / 8.71 / 8.72 GiB — stable to 10 MiB across restarts;
 *   - total steady state: 17.25 / 18.59 / 18.13 GiB, mean 17.99.
 * No further erosion after the first generation (MemAvailable flat within
 * ±0.01 GiB over subsequent generations and minutes of idle), so 18 GiB IS
 * the steady-state reserve; MemAvailable-based checks made after warm-up
 * must not re-reserve any part of it (see PULSAR_SERVER_MEM_FLOOR_BYTES).
 *
 * F1 addendum (task #32, 2026-07-17): the lazy first-request component had
 * grown to ~9.2 GiB (instrumented 3-client greedy burst: MemAvailable fell
 * 13.8 GiB in ~1.5 s, of which 4.59 GiB was a ledgered slot-1 create and
 * the rest this working set + measured total overhead ~18.7-19.0 GiB), and
 * it materialized mid-burst, AFTER every admission check had already read a
 * stale-high MemAvailable.  Two changes de-fang it:
 *   - cli_main.c runs a warmup generation at startup, so the working set
 *     materializes before the listener opens and before any admission math;
 *   - the admission budget is then re-derived from MEASURED post-warmup
 *     MemAvailable, min()'d with the static formula below, so these
 *     constants are an upper bound rather than the load-bearing estimate.
 * Re-measure the constants when the measured/static gap logged at startup
 * ("session admission: measured budget") exceeds ~2 GiB. */
#define PULSAR_SERVER_USABLE_BYTES          (121ull * 1024ull * 1024ull * 1024ull)
#define PULSAR_SERVER_PROCESS_OVERHEAD_BYTES (18ull * 1024ull * 1024ull * 1024ull)

/* Free-memory floor (2026-07-13 lockup postmortem; re-sized 2026-07-15):
 * kernel/OS breathing room ONLY — the last-resort backstop for when other
 * accounting is wrong. It deliberately does NOT cover any process overhead:
 * that is PULSAR_SERVER_PROCESS_OVERHEAD_BYTES' job. The original 6 GiB was
 * sized before the 18 GiB overhead constant existed and double-counted
 * caution on a warmed box: the ~8.7 GiB lazy first-request allocations
 * erode MemAvailable inside the reserve the ledger already subtracted, so
 * the live floor check vetoed sessions the ledger legally admitted
 * (measured 2026-07-14: third 2.5 GiB session refused at 8.39 GiB avail vs
 * the 8.50 the 6 GiB floor demanded, with ~5.9 GiB genuinely free at full
 * commit). 4 GiB backstop sizing (2026-07-15): the incident kernel died
 * near 0 and watchdogs fire at ~5; the measured warmed-box steady state
 * with three live sessions is 5.96 GiB avail, so 4 GiB re-admits the
 * incident shape (needs 2.5 + 4 = 6.5 <= 8.39) while still refusing a
 * further session at that 5.96 steady state (would leave ~3.4). Measured
 * per-session fault-in overshoot beyond the ledgered estimate is ~0.13 GiB
 * (2.63 actual vs 2.50 committed), so the realized post-admission floor
 * stays >= ~3.8 GiB.
 * Two guards use it:
 *   - server_kv_budget_bytes subtracts it from the admission budget, so the
 *     ledger can never legally commit the machine to zero free;
 *   - server_mem_floor_admits: provision_slot (and the eviction precheck)
 *     additionally refuse to create a session unless
 *     MemAvailable >= estimated cost + this floor.
 * The MemAvailable read is a coarse belt-and-suspenders guard: driver 610's
 * UVM accounting lags MemAvailable (and under UVM pressure MemTotal itself
 * SHRINKS — the incident box reported MemTotal 866 MiB — so percentage-based
 * monitors like earlyoom are useless here). Sessions also fault in AFTER
 * the check (measured: two sessions provisioned within 1 s both passed at
 * 11.23 GiB avail before either faulted), so the floor bounds intent, not
 * the instantaneous worst case. One /proc/meminfo read per provisioning
 * attempt, never on a hot path. */
#define PULSAR_SERVER_MEM_FLOOR_BYTES        (4ull * 1024ull * 1024ull * 1024ull)

typedef enum {
    SLOT_IDLE = 0,     /* no live job; the slot's KV may be warm and reusable */
    SLOT_PREFILLING,   /* ingesting a prompt (chunked prefill) */
    SLOT_DECODING,     /* generating tokens */
    SLOT_EVICTED,      /* session freed, ledger released; KV state spilled to
                          the disk kv cache (snapshot-on-evict) so a returning
                          client restores via the normal disk-text path instead
                          of a cold prefill. The slot entry (provisioned ==
                          false) is reusable by the next provisioning. */
} slot_state;

/* Resumable per-job generation state (defined in generate.c). Owns everything
 * that used to be a local of the run-to-completion generate_job: prompt/cache
 * resolution results, prefill progress, stream writers for all four API
 * surfaces, decode-loop trackers, and the deferred socket writer. */
typedef struct gen_state gen_state;

/* A pool slot is a pure bank descriptor over the server's ONE session
 * (server.sess). Slot 0 is provisioned at startup and pinned; slots 1..cap-1
 * are provisioned lazily and evicted LRU-first by the scheduler (worker
 * thread only) — an evicted slot is a reusable hole (provisioned == false,
 * state SLOT_EVICTED) below the n_slots high-water mark. */
typedef struct {
    bool         provisioned;             /* false until admitted; cleared on
                                             eviction (the slot is a reusable
                                             hole). Every reader that used to
                                             skip sess == NULL skips this. */
    uint32_t     bank;                    /* Tier-2: this slot's bank id in the
                                             shared pool (slot i -> bank i). 0 in
                                             classic (non-pooled) mode. */
    int          committed_pos;           /* Tier-2: this bank's committed KV
                                             frontier length (== pulsar_session_pos
                                             when this bank is the live one).
                                             Kept current at every op boundary so
                                             routing/metrics can read a non-live
                                             bank's position without a bank swap. */
    struct job  *active_job;              /* request bound to this slot, or NULL */
    gen_state   *gen;                     /* resumable state for active_job */
    slot_state   state;
    int          ctx_size;                /* context this slot was admitted for */
    uint64_t     est_cost_bytes;          /* ledger-committed session cost (ACTUAL
                                             resident bytes once the session exists;
                                             the true-cost estimate only gates
                                             admission before the create) */
    uint64_t     tokens_emitted;          /* decode bookkeeping for the scheduler */
    uint64_t     last_serviced_us;        /* last quantum wall-clock (scheduler) */
    /* Per-conversation continued-store frontier (see kv_cache_tracker_bind):
     * the shared pulsar_kvstore keeps one continued_last_store_tokens field, but
     * the schedule it tracks belongs to this slot's conversation. */
    int          continued_last_store_tokens;
    /* Tier-2 task #55 increment 2b — proactive-eviction guard. `spilled` means this
     * bank's comp/index PHYSICAL was cudaFree'd (raw KV bit-identical on disk at
     * <spill_dir>/spill-bank-<bank>.kv) while its conversation stays bound here; it
     * is restored (alloc_physical + kv_load) before this slot next decodes. Distinct
     * from SLOT_EVICTED (which frees the bank for a DIFFERENT conversation). */
    bool         spilled;
    /* Protocol live bindings for THIS slot's sampled KV frontier (guarded by
     * server.tool_mu — client threads read them at parse time). They bind
     * tool-call ids / visible transcripts to the session they were sampled on,
     * so a continuation can never match another slot's frontier. */
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
} session_slot;

struct server {
    pulsar_engine *engine;
    /* The ONE session (created at startup, freed once at shutdown): classic
     * mode == 1 bank-less slot (slot 0), pool mode == pool_banks banks over
     * it. Slots are pure bank descriptors; all engine work goes through this
     * pointer. */
    pulsar_session *sess;
    /* Advertised id override (server_config.served_model_id); NULL => built-in
     * server_model_id_from_engine. Read via server_served_model_id. */
    const char *served_model_id;
    /* Advertised display-name override (server_config.served_model_name); NULL
     * => pulsar_engine_model_name. Read via server_served_model_name. */
    const char *served_model_name;
    /* Session pool. slots[0..n_slots) are provisioned; the worker thread is
     * the only mutator of slot fields and n_slots (n_slots additionally
     * published under mu for readers on client threads). */
    session_slot slots[PULSAR_SESSION_POOL_CAP];
    int          n_slots;            /* provisioned slots (worker-owned; published under mu) */
    /* Tier-2 bank-pool state (worker thread only). `pool_banks` > 0 means the
     * shared-pool flip is active: all live slots share server.sess and each
     * owns one bank; `live_bank` is the bank whose device views + host carry are
     * currently installed on that session (server_bank_switch lazily saves the
     * old and restores the new). `spec_max_live` is the three-way scheduler knob
     * (generate.c worker_main): n_active decode banks <= spec_max_live take the
     * per-bank spec/plain time-slice lane, n_active > spec_max_live take the
     * batched multiseq lane. 0 in classic mode. */
    int          pool_banks;
    int          live_bank;
    int          spec_max_live;
    /* plan-34 phase-2 inc 5: fused mixed-batch lane (PULSAR_MIXED_BATCH, default OFF,
     * read once at startup). When ON, the worker folds ONE prefilling slot's next
     * chunk (a K-row prefill run) into the decode quantum's first mixed step
     * (pulsar_session_decode_mixed) instead of advancing it as a separate classic
     * sweep — true continuous batching (P=1). OFF => today's exact decode-quantum +
     * separate one-prefill-chunk time-slice (byte-identical). Only meaningful in
     * pool mode (pool_banks>0). */
    bool         mixed_batch_enabled;
    /* Prefill rows folded into EACH decode step of a fused quantum (PULSAR_MIXED_CHUNK,
     * read once; default 32). Spreading the prefill uniformly across the quantum's
     * steps (vs one big chunk on one step) is what trades the time-slice's per-
     * interval decode STALL for a small uniform per-token cost — the p99 lever. */
    int          mixed_chunk_tokens;
    /* plan-33 inc B: warm full-prefix FORK routing (PULSAR_WARM_FORK, default on;
     * read once at startup). When a request's prompt token-extends an idle warm
     * bank's committed history, the router forks that trunk into a FREE bank and
     * continues there, leaving the trunk intact for siblings. */
    bool         warm_fork_enabled;
    /* plan-33 inc D: minimum shared-prefix TOKEN count for a PARTIAL fork-cut to
     * be worth it (below this, reusing so few tokens loses to a plain cold
     * prefill; a full-prefix match still forks regardless). PULSAR_WARM_PARTIAL_MIN,
     * read once at startup; floored to the ratio-4 align (partial cuts below R
     * reuse nothing). */
    int          warm_partial_min;
    uint64_t     bank_marginal_bytes; /* Tier-2: per-bank ledger charge in pooled
                                         mode (even split of the admitted pool
                                         cost; conservative, demand-paged reality
                                         is smaller). 0 in classic mode. */
    uint64_t     kv_budget_bytes;    /* admission ceiling computed at startup */
    uint64_t     kv_committed_bytes; /* sum of est_cost_bytes over live slots (under mu) */
    /* Tier-2 task #55 increment 2b — proactive-eviction guard. `guard_enabled`
     * gates the whole mechanism (on iff overcommit sized N>1 banks and a spill dir
     * exists). `guard_touched_budget` is the resident-KV ceiling the guard keeps
     * touched_kv under = kv_budget − eager_reserved (banks may grow to 1M but total
     * physical is bounded); `guard_eager_bytes` the eager floor already resident.
     * `guard_evictions` counts spills for metrics. `spill_dir` is a LOCAL fast-disk
     * scratch (NOT the NAS; NOT tmpfs — either would defeat physical reclaim). */
    bool         guard_enabled;
    uint64_t     guard_touched_budget;
    uint64_t     guard_eager_bytes;
    uint64_t     guard_evictions;
    char         spill_dir[512];
    /* Trivial-match threshold for the choose-vs-provision routing decision:
     * template-header tokens measured at startup +
     * PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS (cli_main.c; immutable after
     * startup, worker thread reads only). */
    int slot_trivial_common_tokens;
    int default_tokens;
    kv_disk_cache kv;
    tool_memory tool_mem;
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    time_t started;                  /* wall-clock when the listener came up (uptime for /health) */
    int clients;
    /* /metrics scheduler + prefill gauges (all under mu). n_queued = jobs
     * enqueued not yet bound to a slot; n_generating = jobs bound to slots
     * (0..n_slots, time-sliced by the single worker). m_* are cumulative
     * prefill counters feeding the Prometheus prompt-throughput and
     * prefix-cache-hit metrics. */
    int n_queued;
    int n_generating;
    uint64_t m_prompt_tokens;     /* cumulative prompt tokens prefilled */
    uint64_t m_prefix_queries;    /* cumulative prompt tokens seen (hit-rate denom) */
    uint64_t m_prefix_hits;       /* cumulative prompt tokens served from prefix cache */
    /* Worker-published /metrics snapshots (under mu). The CUDA-state audit
     * above forbids engine calls on client threads, so the worker exports
     * per-slot KV positions/contexts and the engine spec-decode counters here
     * (server_publish_metrics_snapshot, at bind time and once per quantum);
     * send_metrics reads only these. */
    int m_slot_pos[PULSAR_SESSION_POOL_CAP];  /* pulsar_session_pos per provisioned slot */
    int m_slot_ctx[PULSAR_SESSION_POOL_CAP];  /* ctx_size per provisioned slot */
    pulsar_spec_metrics m_spec;               /* engine spec-decode counters */
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
};

/* Jobs are stack-owned by the client thread.  The worker signals completion
 * after the response has been written, so request data and the socket remain
 * valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

typedef enum {
    KV_REASON_UNKNOWN   = PULSAR_KVSTORE_REASON_UNKNOWN,
    KV_REASON_COLD      = PULSAR_KVSTORE_REASON_COLD,
    KV_REASON_CONTINUED = PULSAR_KVSTORE_REASON_CONTINUED,
    KV_REASON_EVICT     = PULSAR_KVSTORE_REASON_EVICT,
    KV_REASON_SHUTDOWN  = PULSAR_KVSTORE_REASON_SHUTDOWN,
} kv_cache_reason;

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

typedef struct {
    server *srv;
    session_slot *slot; /* slot whose session is prefilling (worker thread) */
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
    /* SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;
    bool stream;
    bool enable_cors;
    bool headers_sent;
    bool stream_failed;
    double last_keepalive;
} server_prefill_progress;

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

/* Resumable per-job generation state machine (moved verbatim from
 * generate.c when the scheduler split into its own TU): server_jobs.c owns
 * the lifecycle, but the scheduler/worker (server_sched.c) steps jobs by
 * phase and drives the batched/fused decode lanes through the batch_*
 * fields, so the definition is shared here. */
typedef enum {
    GEN_PREFILL_COLD = 0, /* syncing the cold-store prefix, one chunk/quantum */
    GEN_PREFILL_MAIN,     /* syncing the effective prompt, one chunk/quantum */
    GEN_DECODE_INIT,      /* (re)initialize a decode attempt (old decode_again) */
    GEN_DECODE,           /* sampling loop, K tokens per quantum */
    GEN_FINISH,           /* parse, checkpoints, final response */
    GEN_DONE,
} gen_phase;

struct gen_state {
    job *j;
    gen_phase phase;

    /* prompt/cache resolution (owned by gen_begin, read by later phases) */
    char err[160];
    pulsar_tokens effective_prompt;
    const pulsar_tokens *prompt_for_sync; /* &j->req.prompt or &effective_prompt */
    bool responses_protocol;
    bool responses_live_continuation;
    bool anthropic_live_continuation;
    bool thinking_live_continuation;
    char *disk_cache_path;
    int prompt_tokens;
    double t0;
    double first_token_t;  /* wall time the first output token was produced (TTFT);
                            * 0 until set. Request-lifetime: survives decode_again
                            * so it reflects the genuinely first token emitted. */
    uint64_t trace_id;
    char ctx_span[48];
    char req_flags[64];
    server_prefill_progress progress; /* stable address: callback userdata */
    int cold_store_len;
    int suppressed_continued_last;
    pulsar_tokens cold_prefix;

    /* prefill quantum policy (see gen_prefill_cancel_cb) */
    uint32_t prefill_min_suffix; /* 0 = interrupting is never exact */
    int prefill_chunks_done;     /* chunks completed in the current sync call */
    int prefill_last_current;
    int prefill_total;

    /* response identity + per-protocol stream projections; these live across
     * quanta AND across decode_again recovery attempts */
    char id[96];
    bool structured_stream;
    anthropic_stream anthropic_live;
    openai_stream openai_live;
    responses_stream responses_live;
    bool openai_live_chat;
    bool responses_live_chat;
    long responses_created_at;
    bool dsml_recovery_attempted;
    uint64_t rng;

    /* decode attempt state (reset by GEN_DECODE_INIT) */
    buf text;
    size_t plain_stream_pos;
    size_t stop_scan_from;
    const char *finish;
    int completion;
    int max_tokens;
    bool saw_tool_start;
    bool saw_tool_end;
    bool saw_orphan_tool_end;
    size_t tool_scan_from;
    int next_tool_progress;
    int next_decode_log;
    double decode_t0;
    pulsar_spec_metrics spec_start; /* per-session DSpark counters snapshotted at
                                  * decode start; diffed at finish for this
                                  * request's accept-rate/tokens-per-step */
    double last_decode_log_t;
    int last_decode_log_completion;
    thinking_state thinking;
    bool thinking_gates_tool_markers;
    bool tool_scan_waiting_for_think_close;
    size_t think_recovery_scan_from;
    bool think_tool_recovery_enabled;
    bool dspark_spec_enabled;
    dsml_decode_tracker dsml_tracker;

    /* Tier-2 batched-decode lane state (worker_batched_decode_quantum). A slot
     * becomes batch_active when it joins the shared multiseq lane; it stays
     * there until it finishes (no mid-conversation batched->classic switch, so
     * no stale-logits hazard). batch_feed_token is the next token to commit at
     * batch_feed_pos (the bank's KV frontier); batch_pending holds the tokens
     * committed via multiseq since the bank's last host-checkpoint save, to be
     * reconciled onto the checkpoint when the slot returns to a classic op
     * (finish/store). */
    bool batch_active;
    bool batch_feed_valid;
    int  batch_feed_token;
    int  batch_feed_pos;
    pulsar_tokens batch_pending;
    /* plan-34 inc 5: this prefill slot is not fusable (a fused step rejected its
     * run as not-position-true, e.g. a cache-warm resume); route it CLASSIC. Set
     * once by the fused quantum on giveup; the classic path handles it correctly. */
    bool no_fuse;

    /* deferred, non-blocking client writes (installed for send_all) */
    slot_writer writer;
};

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

typedef struct {
    server *srv;
    int fd;
} client_arg;

typedef struct {
    pulsar_engine_options engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    const char *chdir_path;
    const char *trace_path;
    const char *kv_disk_dir;
    bool kv_disk_disable;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_dsml_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
    /* Advertised model id (/v1/models "id"+"root", /version "model", /metrics
     * label). NULL keeps the built-in "deepseek-v4-flash"/"deepseek-v4-pro".
     * Set to an HF repo path so HF-convention tooling (llama-benchy: model id
     * == tokenizer repo id) resolves the tokenizer. Does not affect KV cache
     * keying (gguf path + numeric id) nor request acceptance (incoming model
     * field is not validated). */
    const char *served_model_id;
    /* Advertised human display name (/v1/models "name", /version "model_name").
     * NULL keeps the built-in shape name. Free-form; may contain spaces. */
    const char *served_model_name;
} server_config;

/* ---- shared globals ---- */

extern volatile sig_atomic_t g_stop_requested;
extern volatile sig_atomic_t g_listen_fd;
extern const dsml_syntax dsml_syntaxes[3];

/* ---- shared functions ---- */

void stop_signal_handler(int sig);
void die(const char *msg);
void *server_xmalloc(size_t n);
void *server_xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
bool random_bytes(void *dst, size_t len);
void pulsar_die(const char *msg); /* engine util.c; aborts the process */
char *xstrndup(const char *s, size_t n);
void buf_append(buf *b, const void *p, size_t n);
void buf_putc(buf *b, char c);
void buf_puts(buf *b, const char *s);
void buf_printf(buf *b, const char *fmt, ...);
char *buf_take(buf *b);
void buf_free(buf *b);
void json_ws(const char **p);
bool json_lit(const char **p, const char *lit);
bool json_string(const char **p, char **out);
bool json_number(const char **p, double *out);
bool json_int(const char **p, int *out);
bool json_bool(const char **p, bool *out);
bool json_skip_value(const char **p);
bool json_raw_value(const char **p, char **out);
char *json_minify_raw_value(const char *json);
bool json_content(const char **p, char **out);
void random_tool_id(char *dst, size_t dstlen, api_style api);
void tool_calls_free(tool_calls *calls);
void tool_calls_push(tool_calls *calls, tool_call tc);
void chat_msg_add_tool_call_id(chat_msg *m, const char *id);
void chat_msgs_free(chat_msgs *msgs);
void chat_msgs_push(chat_msgs *msgs, chat_msg msg);
void tool_schema_orders_free(tool_schema_orders *orders);
const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name);
void request_init(request *r, req_kind kind, int max_tokens);
void request_free(request *r);
pulsar_think_mode think_mode_from_enabled(bool enabled, pulsar_think_mode effort);
bool parse_reasoning_effort_name(const char *s, pulsar_think_mode *out);
bool parse_reasoning_effort_value(const char **p, pulsar_think_mode *out);
bool parse_thinking_control_value(const char **p, bool *thinking_enabled);
bool parse_output_config_effort(const char **p, pulsar_think_mode *effort);
bool model_alias_disables_thinking(const char *model);
bool model_alias_enables_thinking(const char *model);
const char *server_model_id_from_engine(pulsar_engine *engine);
/* Advertised model id ("id"/"root"/metrics): --served-model-id override if set,
 * else the built-in id. Must be HF-resolvable for HF-convention tokenizer tools. */
const char *server_served_model_id(const server *s);
/* Advertised display name ("name"): --served-model-name override if set, else
 * the built-in shape name. Free-form; never used as a tokenizer/repo id. */
const char *server_served_model_name(const server *s);
void stop_list_clear(stop_list *stops);
void stop_list_push(stop_list *stops, char *s);
bool parse_stop(const char **p, stop_list *out);
bool stop_list_find_from(const stop_list *stops, const char *text,
                                size_t from, size_t *pos, size_t *len);
size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len);
size_t utf8_stream_safe_len(const char *s, size_t start,
                                   size_t limit, bool final);
bool parse_stream_options(const char **p, bool *include_usage);
void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json);
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders);
bool parse_messages(const char **p, chat_msgs *msgs);
bool parse_anthropic_messages(const char **p, chat_msgs *msgs);
bool parse_anthropic_system(const char **p, char **out);
void append_tool_result_text(buf *b, const char *s);
bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order);
void append_json_object_or_empty(buf *b, const char *json);
void append_dsml_tool_calls_text(buf *b, const tool_calls *calls);
bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas);
bool chat_history_preserves_reasoning(const chat_msgs *msgs,
                                             const char *tool_schemas);
char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     pulsar_think_mode think_mode);
bool responses_validate_tool_outputs(server *s, const chat_msgs *msgs,
                                            pulsar_think_mode think_mode,
                                            bool *requires_live_tool_state,
                                            bool *requires_live_reasoning,
                                            char *err, size_t errlen);
void responses_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
bool anthropic_validate_tool_results(server *s, const chat_msgs *msgs,
                                            bool *requires_live_tool_state,
                                            char *err, size_t errlen);
void anthropic_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs);
bool parse_chat_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                               int ctx_size, request *r, char *err, size_t errlen);
bool parse_anthropic_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen);
bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders);
bool parse_responses_request(pulsar_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen);
bool parse_completion_request(pulsar_engine *e, const char *body, int def_tokens,
                                     int ctx_size, request *r, char *err, size_t errlen);
bool send_all(int fd, const void *p, size_t n);
void json_escape(buf *b, const char *s);
void json_escape_n(buf *b, const char *s, size_t n);
void json_escape_fragment_n(buf *b, const char *s, size_t n);
const char *find_any_tool_start(const char *s);
void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end);
size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit);
const char *find_last_substr(const char *s, const char *needle);
char *dsml_unescape_text(const char *s);
char *dsml_attr(const char *tag, const char *name);
bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls);
bool try_repair_dsml(const char *s, size_t len, buf *out);
bool parse_generated_message_for_response(const char *text,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 bool require_thinking_closed,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out);
void append_json_object_string(buf *b, const char *json);
void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders);
void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders);
bool http_response(int fd, bool enable_cors, int code, const char *type, const char *body);
bool http_error(int fd, bool enable_cors, int code, const char *msg);
void request_forced_tool_seed(const request *r, buf *out);
void request_apply_forced_tool_prefill(request *r);
bool request_exceeds_context(const request *r, int ctx_size);
bool http_error_context_length_exceeded(int fd, bool enable_cors,
                                               const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size);
bool sse_headers(int fd, bool enable_cors);
bool sse_error_event(int fd, const request *r, const char *msg);
bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish);
int clamp_usage_tokens(int value, int max);
void append_openai_usage_json(buf *b, const request *r,
                                     int prompt_tokens, int completion_tokens);
/* Emit the additive ",\"timings\":{...}" fragment (leading comma included) from
 * r->timings, or nothing when r->timings.valid is false. Rates are derived here
 * with guarded divisions; a zero denominator omits that rate. */
void append_openai_timings_json(buf *b, const request *r);
bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens);
bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens);
void openai_stream_start(const request *r, openai_stream *st);
void openai_stream_free(openai_stream *st);
bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit);
bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b);
const char *find_lit_bounded(const char *s, size_t n, const char *lit);
dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len);
bool dsml_decode_state_is_tool(dsml_decode_state state);
bool dsml_decode_state_uses_payload_sampling(dsml_decode_state state);
void dsml_decode_tracker_init(dsml_decode_tracker *dt);
void dsml_decode_tracker_update(dsml_decode_tracker *dt,
                                       const char *raw, size_t raw_len);
size_t tool_param_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const char *param_end,
                                               bool is_string);
bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final);
bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens);
bool request_uses_openai_live_stream(const request *r);
bool request_uses_responses_live_stream(const request *r);
bool request_uses_structured_stream(const request *r);
void responses_stream_init(const request *r, responses_stream *st);
void responses_stream_free(responses_stream *st);
bool responses_sse_created(int fd, const request *r, responses_stream *st,
                                  long created_at);
void responses_append_function_call_item(buf *b, const tool_call *tc,
                                                const responses_tool_item *item,
                                                const char *item_status,
                                                bool with_args,
                                                const tool_schema_orders *orders);
bool responses_sse_completed(int fd, const request *r,
                                    responses_stream *st,
                                    const tool_calls *calls,
                                    const responses_tool_item *tool_items,
                                    const char *finish,
                                    int prompt_tokens, int completion_tokens,
                                    long created_at);
bool responses_sse_stream_update(int fd, const request *r,
                                        responses_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final);
bool responses_sse_finish_live(int fd, const request *r,
                                      responses_stream *st,
                                      const char *raw, size_t raw_len,
                                      const char *recovered_content,
                                      const tool_calls *calls,
                                      const char *finish,
                                      int prompt_tokens, int completion_tokens,
                                      long created_at);
bool responses_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens);
bool final_response(int fd, bool enable_cors,
                           const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens);
void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix,
                                     const tool_schema_orders *orders);
bool anthropic_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens);
bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st);
void anthropic_stream_free(anthropic_stream *st);
size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);
bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final);
bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const tool_calls *calls,
                                      const char *finish, int completion_tokens);
double server_now_sec(void);
void server_log(pulsar_log_type type, const char *fmt, ...);
int tool_memory_max_entries(const tool_memory *m);
tool_memory_block *tool_memory_find_block_locked(tool_memory *m,
                                                        const char *dsml,
                                                        size_t len);
void tool_memory_free(tool_memory *m);
void live_tool_state_free(live_tool_state *st);
void visible_live_free(visible_live_state *st);
/* Live protocol bindings are per-slot (they describe one session's sampled
 * frontier); has_call_id scans every provisioned slot because request parsing
 * runs before the job is bound to a slot. */
void thinking_live_clear(server *s, session_slot *sl);
void thinking_live_remember(server *s, session_slot *sl, const char *visible_text);
void responses_live_remember(server *s, session_slot *sl, const char *visible_text,
                                    const tool_calls *calls);
void anthropic_live_remember(server *s, session_slot *sl, const tool_calls *calls);
void responses_live_clear(server *s, session_slot *sl);
void anthropic_live_clear(server *s, session_slot *sl);
bool responses_live_has_call_id(server *s, const char *id);
bool anthropic_live_has_call_id(server *s, const char *id);
bool responses_live_matches_request(server *s, const session_slot *sl,
                                           const stop_list *ids,
                                           int live_tokens);
bool anthropic_live_matches_request(server *s, const session_slot *sl,
                                           const stop_list *ids,
                                           int live_tokens);
/* Slots whose live binding contains all of the request's continuation ids
 * (worker thread; used to route a continuation to the session that owns it). */
session_slot *responses_live_slot_for_ids(server *s, const stop_list *ids);
session_slot *anthropic_live_slot_for_ids(server *s, const stop_list *ids);
bool tool_memory_has_id(server *s, const char *id);
void tool_memory_remember(server *s, const tool_calls *calls);
void tool_memory_put_source(server *s, const char *id, const char *dsml,
                                   tool_memory_source source);
void tool_memory_put(server *s, const char *id, const char *dsml);
void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats);
void assign_tool_call_ids(server *s, tool_calls *calls, api_style api);
void apply_openai_stream_tool_ids(tool_calls *calls,
                                         const openai_stream *st);
void apply_anthropic_stream_tool_ids(tool_calls *calls,
                                            const anthropic_stream *st);
kv_cache_options kv_cache_default_options(void);
void le_put32(uint8_t *p, uint32_t v);
void sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
bool id_list_contains(const stop_list *ids, const char *id);
void id_list_push_unique(stop_list *ids, const char *id);
void id_list_free(stop_list *ids);
void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids);
char *path_join(const char *dir, const char *name);
bool kv_tool_map_serialized_size(server *s, const char *text,
                                        uint64_t *bytes_out);
bool kv_tool_map_write(server *s, FILE *fp, const char *text,
                              uint64_t *written_bytes);
int kv_tool_map_load_from_pos(server *s, FILE *fp, const stop_list *wanted);
void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes);
void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);
double kv_entry_eviction_score(const kv_entry *e, const pulsar_tokens *live,
                                      uint64_t now,
                                      const pulsar_kvstore_eviction_context *incoming);
void kv_cache_evict(kv_disk_cache *kc, const pulsar_tokens *live,
                           uint64_t extra_bytes,
                           const pulsar_kvstore_eviction_context *incoming);
bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt);
void kv_cache_close(kv_disk_cache *kc);
char *render_tokens_text(pulsar_engine *engine, const pulsar_tokens *tokens, size_t *out_len);
void tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
void build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        pulsar_tokens *out);
int kv_cache_store_len(const kv_disk_cache *kc, int tokens);
int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const pulsar_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id);
int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens);
bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out);
bool kv_cache_store_live_prefix(server *s, session_slot *sl,
                                       const pulsar_tokens *tokens,
                                       int store_len, const char *reason);
/* Returns whether a checkpoint file was actually written — eviction uses this
 * for failure honesty (evict-without-snapshot falls back to client re-prefill;
 * older callers ignore the result as before). */
bool kv_cache_store_current(server *s, session_slot *sl, const char *reason);
/* The continued-store frontier (lib field kc->continued_last_store_tokens) is
 * per-conversation state on a kvstore shared by every slot. Every
 * tracker-touching operation brackets itself with these on the single worker
 * thread: bind loads the acting slot's frontier into the shared struct, flush
 * writes it back (2026-07-14 review: without this, slot A's high-water mark
 * suppressed slot B's continued checkpoints, and a cold request on B reset
 * A's schedule). */
void kv_cache_tracker_bind(server *s, session_slot *sl);
void kv_cache_tracker_flush(server *s, session_slot *sl);
void kv_cache_note_store(kv_disk_cache *kc, int tokens);
int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens);
void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens);
void kv_cache_discard_failed_disk_entry(server *s, session_slot *sl,
                                               const char *path);
void kv_cache_maybe_store_continued(server *s, session_slot *sl);
int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size);
int kv_cache_try_load_text(server *s, session_slot *sl, const char *prompt_text,
                                  pulsar_tokens *effective_prompt,
                                  char **loaded_path_out,
                                  uint8_t *loaded_ext_flags_out,
                                  bool responses_protocol);
int kv_cache_try_load(server *s, session_slot *sl, const request *req,
                             pulsar_tokens *effective_prompt,
                             char **loaded_path_out,
                             uint8_t *loaded_ext_flags_out);
int live_text_prefix_prompt(server *s, session_slot *sl, const request *req,
                                   pulsar_tokens *effective_prompt);
int responses_live_continuation_prompt(server *s, session_slot *sl,
                                              const request *req,
                                              int live_pos,
                                              pulsar_tokens *effective_prompt,
                                              int *matched_ids);
int anthropic_live_continuation_prompt(server *s, session_slot *sl,
                                              const request *req,
                                              int live_pos,
                                              pulsar_tokens *effective_prompt,
                                              int *matched_ids);
int responses_live_visible_prefix_prompt(server *s, session_slot *sl,
                                                const request *req,
                                                int live_pos,
                                                pulsar_tokens *effective_prompt);
int thinking_live_visible_prefix_prompt(server *s, session_slot *sl,
                                               const request *req,
                                               int live_pos,
                                               pulsar_tokens *effective_prompt);
/* Routing probe: does this slot's live thinking binding mark it as the warm
 * continuation of req's visible transcript? Same guards as
 * thinking_live_visible_prefix_prompt but byte-prefix check only — no
 * tokenization, no effective-prompt build. Returns the matched visible-key
 * length (>0), or 0 for no match (defined in kv_cache.c; unit-tested in
 * cli_main.c). */
size_t thinking_live_binds_prompt(server *s, session_slot *sl,
                                         const request *req, int live_pos);
/* Trivial-match classifier for the router's choose-vs-provision decision
 * (defined in generate.c; unit-tested in cli_main.c). */
bool server_slot_match_is_trivial(int common, int slot_pos,
                                         int trivial_tokens);
/* Admission predicate (defined in cli_main.c; unit-tested there). */
bool server_kv_admits(uint64_t kv_budget_bytes,
                             uint64_t committed_bytes,
                             uint64_t incoming_bytes);
/* Live MemAvailable floor predicate: kernel-breathing-room backstop applied
 * at provisioning time on top of the ledger (defined in cli_main.c;
 * unit-tested there). avail == 0 (unreadable /proc/meminfo) fails closed. */
bool server_mem_floor_admits(uint64_t avail_bytes, uint64_t est_bytes);

/* MemAvailable from /proc/meminfo, in bytes (0 on parse failure — callers
 * fail closed).  Never called on a token/layer hot path (defined in
 * generate.c; also used by startup warmup/budget derivation in cli_main.c). */
uint64_t server_mem_available_bytes(void);
/* Log estimate-vs-actual for a freshly created session, warn loudly on >10%
 * drift (sizing code out of sync with the allocator), and return the value
 * the ledger must commit — the actual (defined in generate.c). */
uint64_t server_reconciled_session_cost(int slot_idx, int ctx,
                                               uint64_t est_bytes,
                                               uint64_t actual_bytes);
/* Eviction ledger release: subtract an evicted slot's committed bytes from
 * the admission ledger total, warning loudly (and clamping to 0, which fails
 * toward over-admission being caught by the MemAvailable floor rather than
 * leaking budget forever) if the pairing ever underflows (defined in
 * generate.c; unit-tested in cli_main.c). */
uint64_t server_ledger_release(uint64_t committed_total, uint64_t slot_cost);
/* Tier-2 bank-aware frontier position of `sl`, correct whether or not sl->bank
 * is the currently-installed bank of the shared pool session (a non-live bank
 * reads its saved host carry via pulsar_session_bank_pos; the live bank reads the
 * live checkpoint). In classic (non-pooled) mode == pulsar_session_pos(s->sess).
 * Worker-thread scheduling reads AND the client/worker tool-id lookups use this
 * instead of pulsar_session_pos so a non-live bank's frontier is never misread as
 * the pool's live cursor (defined in generate.c). */
/* Install `bank` on the shared pool session: lazily saves the outgoing bank's
 * carry, reloads a guard-spilled target from disk, and repoints the graph's
 * views.  Returns false WITHOUT installing on any failure — callers must fail
 * the request rather than run against a half-repointed view set. */
bool server_bank_switch(server *s, int bank);
int server_slot_frontier_pos(const server *s, const session_slot *sl);
/* LRU eviction victim: least-recently-serviced idle provisioned slot,
 * tie-broken by smallest committed bytes; slot 0 pinned; protect[i] (may be
 * NULL) marks slots a queued live continuation still needs. Returns a slot
 * index or -1. Pure selection over host fields — never touches the session
 * (defined in generate.c; unit-tested in cli_main.c). */
int server_evict_pick_victim(const session_slot *slots, int n_slots,
                                    const bool *protect);
void trace_cache_capture(
        trace_cache_diag *d,
        const pulsar_tokens *live,
        const pulsar_tokens *prompt,
        int old_pos,
        int common);
const char *trace_cache_miss_reason(const trace_cache_diag *d);
uint64_t trace_begin(
        server *s,
        const job *j,
        int cached,
        int effective_prompt_tokens,
        const trace_cache_diag *cache_diag,
        const char *cache_source,
        int disk_cached,
        const char *disk_path);
void trace_piece(server *s, uint64_t id, const char *piece, size_t len);
void trace_event(server *s, uint64_t id, const char *fmt, ...);
void trace_finish(
        server *s,
        uint64_t id,
        const request *r,
        const char *final_finish,
        int completion,
        bool saw_tool_start,
        bool saw_tool_end,
        const char *parsed_content,
        const char *parsed_reasoning,
        const tool_calls *parsed_calls,
        double elapsed);
void request_ctx_span(char *buf, size_t len, int cached, int prompt);
void log_flags(char *buf, size_t len, bool responses_protocol,
                      bool tools, bool thinking,
                      bool dsml_start, bool dsml_end);
void log_decode_progress(req_kind kind, int prompt_tokens, int completion,
                                bool responses_protocol,
                                bool tools, bool thinking,
                                bool dsml_start, bool dsml_end,
                                double decode_t0,
                                double *last_t, int *last_completion);
void thinking_state_feed(thinking_state *st, const char *p, size_t len);
thinking_state thinking_state_from_prompt(const request *r);
char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail);
bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish);
char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls);
char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls);
char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content);
bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls);
bool enqueue(server *s, job *j);
void *worker_main(void *arg);
void server_publish_metrics_snapshot(server *s);
/* Job-lifecycle entry points (server_jobs.c), driven by the scheduler/
 * worker (server_sched.c): bind/step/unbind plus the three per-token
 * helpers the batched and fused mixed-batch quanta share with the classic
 * decode loop. */
void gen_resolve_sampling(const request *req, float *temperature,
                          int *top_k, float *top_p, float *min_p);
bool gen_emit_token(server *s, session_slot *sl, int token);
void gen_stream_begin(server *s, session_slot *sl);
void generate_job_begin(server *s, session_slot *sl, job *j);
void generate_job_step(server *s, session_slot *sl);
void generate_job_end(server *s, session_slot *sl);
void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens);
void *client_main(void *arg);
int listen_on(const char *host, int port);
void configure_client_socket(int fd);
void set_client_socket_nonblocking(int fd);
void usage(FILE *fp, const char *topic);

/* ---- shared inline helpers ---- */


#endif /* PULSAR_SERVER_INTERNAL_H */
