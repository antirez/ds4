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
    uint64_t model_file_bytes;    /* size of GGUF on disk, fingerprint proxy */
    uint8_t  model_sample[32];    /* first 32 bytes of GGUF (cheap fingerprint) */
} ds4_rpc_config;

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
int ds4_rpc_handshake_server(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             ds4_rpc_config *peer, char *err, size_t errlen);

/* Head-side: ship one decode request and wait for the logits reply. */
int ds4_rpc_decode_request(ds4_rpc_handle *h,
                           uint32_t token, uint32_t pos,
                           const float *residual_hc, uint64_t n_residual_floats,
                           float *out_logits, uint64_t n_logit_floats,
                           char *err, size_t errlen);

/* Tail-side: receive one decode request (residual + metadata) and emit one
 * decode reply (logits).  The two halves run in the worker's serve loop. */
int ds4_rpc_decode_recv(ds4_rpc_handle *h,
                        uint32_t *token, uint32_t *pos,
                        float *residual_hc, uint64_t n_residual_floats,
                        char *err, size_t errlen);
int ds4_rpc_decode_reply(ds4_rpc_handle *h,
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

#endif
