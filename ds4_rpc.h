#ifndef DS4_RPC_H
#define DS4_RPC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

/* Pipeline-parallel RPC transport.
 *
 * The head process owns layers [0, L_mid), drives session state, samples
 * tokens, and talks to one tail worker that owns [L_mid, DS4_N_LAYER) on a
 * second machine.  At each decode step the head ships (token, pos, residual)
 * to the tail and reads back the full logits vector.  Prefill, MTP, and the
 * disk KV cache are not handled here yet; this layer is intentionally narrow
 * so a first working multi-machine path can land before we expand surface.
 *
 * Wire format: little-endian, length-prefixed frames carrying one opcode and
 * one fixed payload shape per opcode.  The handshake validates that both
 * sides agree on model shape, routed-quant bits, and the split point; a
 * mismatch fails the connection rather than silently producing wrong output.
 *
 * This module is Metal-only by transitive dependency on the engine, but the
 * transport itself is plain POSIX sockets and would work on Linux/FreeBSD if
 * a CUDA/Vulkan engine ever needed it. */

#define DS4_RPC_MAGIC     0x43505244u  /* "DRPC" little-endian */
#define DS4_RPC_VERSION   1u

typedef enum {
    DS4_RPC_OP_HELLO_CLIENT  = 1, /* head -> tail: config exchange */
    DS4_RPC_OP_HELLO_SERVER  = 2, /* tail -> head: ack or error */
    DS4_RPC_OP_DECODE_REQ    = 3, /* head -> tail: token + pos + residual */
    DS4_RPC_OP_DECODE_REPLY  = 4, /* tail -> head: status + logits */
    DS4_RPC_OP_RESET         = 5, /* head -> tail: drop session, fresh KV */
    DS4_RPC_OP_RESET_REPLY   = 6, /* tail -> head: ack reset */
    DS4_RPC_OP_SHUTDOWN      = 7, /* either side: clean disconnect */
    DS4_RPC_OP_PREFILL_REQ   = 8, /* head -> tail: batch residual for a chunk */
    DS4_RPC_OP_PREFILL_REPLY = 9, /* tail -> head: status + maybe logits */
    DS4_RPC_OP_REWIND        = 10, /* head -> tail: rewind to target_pos */
    DS4_RPC_OP_REWIND_REPLY  = 11, /* tail -> head: ack */
    DS4_RPC_OP_MTP_TRIM      = 12, /* head -> tail: trim MTP drafts to accepted */
    DS4_RPC_OP_MTP_TRIM_REPLY = 13,
    DS4_RPC_OP_VERIFY_BATCH   = 14, /* head -> tail: batched all-or-nothing verify */
    DS4_RPC_OP_VERIFY_BATCH_REPLY = 15,
} ds4_rpc_op;

/* Configuration carried in the handshake.  Both sides must agree on every
 * field except role.  The split point is the tail's range; the head's range
 * is [0, tail_layer_start). */
typedef struct {
    uint32_t version;             /* DS4_RPC_VERSION */
    uint32_t n_layer_total;       /* DS4_N_LAYER, model-fixed */
    uint32_t n_embd;              /* DS4_N_EMBD */
    uint32_t n_hc;                /* DS4_N_HC */
    uint32_t n_vocab;             /* DS4_N_VOCAB */
    uint32_t routed_quant_bits;   /* 2 or 4 */
    uint32_t tail_layer_start;    /* L_mid, the split point */
    uint32_t tail_layer_end;      /* DS4_N_LAYER for a 2-machine setup */
    uint32_t ctx_size;            /* must match between head and tail; tail's
                                     KV cache is sized for this and overflows
                                     if the head runs longer prompts */
    uint32_t tail_has_mtp;        /* 1 if the worker has --mtp loaded, 0 else */
    uint32_t tail_mtp_draft_tokens; /* worker's --mtp-draft (1..16); ignored
                                       when tail_has_mtp == 0 */
    uint32_t reserved0;
    uint64_t model_file_bytes;    /* size of GGUF on disk, fingerprint proxy */
    uint8_t  model_sample[32];    /* first 32 bytes of GGUF (cheap fingerprint) */
} ds4_rpc_config;

#define DS4_RPC_MAX_DRAFTS 16

typedef struct ds4_rpc_handle ds4_rpc_handle;

/* Lifecycle. */
int  ds4_rpc_dial(const char *host, uint16_t port,
                  ds4_rpc_handle **out, char *err, size_t errlen);
int  ds4_rpc_listen_one(const char *bind_host, uint16_t port,
                        ds4_rpc_handle **out, char *err, size_t errlen);
void ds4_rpc_close(ds4_rpc_handle *h);
int  ds4_rpc_fd(const ds4_rpc_handle *h);

/* Handshake.  The client sends its config; the server reads it, decides
 * whether to accept, and replies with either ok or an error message.  The
 * server's accepted config is whatever it offered; the client should copy it
 * back into the engine if proceeding. */
int ds4_rpc_handshake_client(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             char *err, size_t errlen);
/* Variant that also returns the server's accepted config in out_peer so the
 * head can learn tail-side capabilities (MTP, etc.). */
int ds4_rpc_handshake_client_peer(ds4_rpc_handle *h,
                                  const ds4_rpc_config *cfg,
                                  ds4_rpc_config *out_peer,
                                  char *err, size_t errlen);
int ds4_rpc_handshake_server(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             ds4_rpc_config *peer, char *err, size_t errlen);

/* Head-side: ship one decode request and wait for the logits reply.
 * out_drafts (capacity DS4_RPC_MAX_DRAFTS) is filled with the MTP draft
 * tokens the tail produced, if any; *out_n_drafts is the count (may be 0).
 * Pass NULL for out_drafts and 0 for max_drafts if the head doesn't want
 * MTP support. */
int ds4_rpc_decode_request(ds4_rpc_handle *h,
                           uint32_t token, uint32_t pos,
                           bool want_drafts,
                           const float *residual_hc, uint64_t n_residual_floats,
                           float *out_logits, uint64_t n_logit_floats,
                           uint32_t *out_drafts, uint32_t max_drafts,
                           uint32_t *out_n_drafts,
                           char *err, size_t errlen);

/* Tail-side: receive one decode request (residual + metadata) and emit one
 * decode reply (logits + optional MTP drafts).  The two halves run in the
 * worker's serve loop. */
int ds4_rpc_decode_recv(ds4_rpc_handle *h,
                        uint32_t *token, uint32_t *pos,
                        bool *want_drafts,
                        float *residual_hc, uint64_t n_residual_floats,
                        char *err, size_t errlen);
int ds4_rpc_decode_reply(ds4_rpc_handle *h,
                         const float *logits, uint64_t n_logit_floats,
                         const uint32_t *drafts, uint32_t n_drafts,
                         char *err, size_t errlen);

/* After speculative verification on the head, tell the tail how many of its
 * MTP drafts were accepted so it can roll back the unused MTP cache rows. */
int ds4_rpc_mtp_trim(ds4_rpc_handle *h, uint32_t accepted_drafts,
                     char *err, size_t errlen);
int ds4_rpc_mtp_trim_recv(ds4_rpc_handle *h, uint32_t *accepted_drafts,
                          char *err, size_t errlen);
int ds4_rpc_mtp_trim_reply(ds4_rpc_handle *h, char *err, size_t errlen);

/* Batched all-or-nothing speculative verification.  Head ships n_tokens
 * residuals (its slice's hidden state for n_tokens consecutive draft tokens)
 * plus the expected-next tokens.  Tail runs batched prefill, computes per-
 * row argmax via the output head, compares to expected, and either commits
 * all n_tokens (returning logits at the final row for sampling the next-
 * after-accepted) or reverts its KV via spec_frontier_snapshot/restore and
 * returns accepted=0.  The head does the same KV snapshot on its side. */
int ds4_rpc_verify_batch_request(ds4_rpc_handle *h,
                                 uint32_t n_tokens, uint32_t pos_start,
                                 const float *batch_residual,
                                 uint64_t n_residual_floats,
                                 const uint32_t *expected_next,
                                 uint32_t n_expected,
                                 uint32_t *out_n_accepted,
                                 float *out_logits, uint64_t n_logit_floats,
                                 char *err, size_t errlen);
int ds4_rpc_verify_batch_recv(ds4_rpc_handle *h,
                              uint32_t *n_tokens, uint32_t *pos_start,
                              float *batch_residual, uint64_t max_residual_floats,
                              uint64_t *out_n_residual_floats,
                              uint32_t *expected_next, uint32_t max_expected,
                              uint32_t *out_n_expected,
                              char *err, size_t errlen);
int ds4_rpc_verify_batch_reply(ds4_rpc_handle *h,
                               uint32_t n_accepted,
                               const float *logits, uint64_t n_logit_floats,
                               char *err, size_t errlen);

/* Head-side: ship one prefill chunk's batch residual and (optionally) wait
 * for logits.  n_tokens is the chunk size; pos_start is where this chunk
 * begins in the absolute session position; want_logits is true only on the
 * final chunk of the prompt.  When want_logits is false, out_logits may be
 * NULL and n_logit_floats may be 0. */
int ds4_rpc_prefill_request(ds4_rpc_handle *h,
                            uint32_t n_tokens, uint32_t pos_start,
                            bool want_logits,
                            const float *batch_residual_hc,
                            uint64_t n_residual_floats,
                            float *out_logits, uint64_t n_logit_floats,
                            char *err, size_t errlen);

/* Tail-side counterparts.  ds4_rpc_prefill_recv writes the batch residual
 * into the caller's buffer; ds4_rpc_prefill_reply ships back the requested
 * logits (or empty reply if !want_logits). */
int ds4_rpc_prefill_recv(ds4_rpc_handle *h,
                         uint32_t *n_tokens, uint32_t *pos_start,
                         bool *want_logits,
                         float *batch_residual_hc, uint64_t max_residual_floats,
                         uint64_t *out_n_residual_floats,
                         char *err, size_t errlen);
int ds4_rpc_prefill_reply(ds4_rpc_handle *h,
                          bool has_logits,
                          const float *logits, uint64_t n_logit_floats,
                          char *err, size_t errlen);

/* Session control.  ds4_rpc_recv_op peeks at the next frame without
 * consuming it; the tail's serve loop uses it to dispatch DECODE_REQ
 * (handled by ds4_rpc_decode_recv) vs RESET (ds4_rpc_reset_recv) vs
 * SHUTDOWN (ds4_rpc_shutdown_recv). */
int ds4_rpc_reset(ds4_rpc_handle *h, char *err, size_t errlen);
int ds4_rpc_recv_op(ds4_rpc_handle *h, ds4_rpc_op *op,
                    char *err, size_t errlen);
int ds4_rpc_reset_recv(ds4_rpc_handle *h, char *err, size_t errlen);
int ds4_rpc_reset_reply(ds4_rpc_handle *h, char *err, size_t errlen);
int ds4_rpc_shutdown_send(ds4_rpc_handle *h);
int ds4_rpc_shutdown_recv(ds4_rpc_handle *h, char *err, size_t errlen);

/* Partial rewind: tell the tail to truncate its checkpoint to target_pos
 * without dropping the whole session.  Used by server-side tool-call
 * canonicalization which rewinds a few tokens after sampling a tool block. */
int ds4_rpc_rewind(ds4_rpc_handle *h, uint32_t target_pos,
                   char *err, size_t errlen);
int ds4_rpc_rewind_recv(ds4_rpc_handle *h, uint32_t *target_pos,
                        char *err, size_t errlen);
int ds4_rpc_rewind_reply(ds4_rpc_handle *h, char *err, size_t errlen);

#endif
