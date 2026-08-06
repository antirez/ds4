#ifndef DS4_DISTRIBUTED_RDMA_H
#define DS4_DISTRIBUTED_RDMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_dist_dchan.h"

/* RDMA-over-Thunderbolt transport for distributed inference (Apple only).
 *
 * All libibverbs use is confined to ds4_distributed_rdma.c; the rest of the
 * engine sees only this interface. librdma is loaded at runtime
 * (ds4_rdma_verbs.h), and off Apple the whole file compiles to stubs that report
 * "unavailable", so callers need no platform guard - the engine just stays on
 * TCP, exactly as ds4_tp.c does for tensor parallelism.
 *
 * A connection is a PAIR of UC QPs, one per direction, so the channel is full
 * duplex and thread-safe like a TCP socket: writer and reader touch disjoint
 * verbs objects and share no mutable state. That is what lets pipelined prefill
 * and worker prefetch run over RDMA (see the channel comment in the .c, and
 * Apple TN3205).
 *
 * Bring-up is two-phase: a connected UC pair needs each side's endpoint
 * (GID/QPN/PSN) before RTR. conn_create() builds the local QPs (RESET->INIT) and
 * reports both endpoints; ds4_distributed.c relays them over the TCP bootstrap;
 * conn_connect() drives INIT->RTR->RTS. Only the steady-state data plane runs
 * over the QPs.
 */

/* Wire-relayed connection endpoint. Fixed-size, byte-array GID so it serializes
 * verbatim (GID bytes are already network order). Exchanged during bootstrap. */
typedef struct {
    uint8_t  gid[16];
    uint32_t qp_num;
    uint32_t psn;
    uint32_t dev_port;
    uint32_t mtu;      /* enum ibv_mtu value actually negotiated for this QP */
    uint16_t lid;      /* 0 on RoCE links; carried for IB-mode links          */
    char     dev_name[32];
} ds4_dist_rdma_endpoint;

/* Both endpoints of a connection, exchanged as one bootstrap message. The pair
 * is crossed on connect: our tx QP is wired to the peer's rx QP and vice versa,
 * so both peers run the identical connect logic. */
typedef struct {
    ds4_dist_rdma_endpoint tx;   /* carries our data out, peer credits in */
    ds4_dist_rdma_endpoint rx;   /* carries peer data in, our credits out */
} ds4_dist_rdma_endpoints;

typedef struct ds4_dist_rdma_conn ds4_dist_rdma_conn;

/* Returns 1 if librdma loaded and at least one RDMA device is present, 0
 * otherwise (no provider, no device, or a non-Apple build). Never aborts. */
int ds4_dist_rdma_available(void);

/* Writes a short human-readable summary of RDMA availability into buf.
 * Always NUL-terminates when buflen > 0. Returns the number of devices found. */
int ds4_dist_rdma_describe(char *buf, size_t buflen);

/* Phase 1: open dev_name (NULL = first available) once per direction, each half
 * getting its own PD/CQ/UC QP taken to INIT with its ring registered. Registering
 * before RTS is deliberate: Apple's provider maps a QP's MR set once on the RTR
 * transition (TN3205), so a later MR is never mapped; the per-half PD keeps a QP
 * off the other direction's rings. On success *out owns all resources and
 * local_out holds the endpoints to relay. Returns 0, or -1 (err filled). */
int ds4_dist_rdma_conn_create(ds4_dist_rdma_conn **out,
                              const char *dev_name,
                              ds4_dist_rdma_endpoints *local_out,
                              char *err,
                              size_t errlen);

void ds4_dist_rdma_conn_destroy(ds4_dist_rdma_conn *conn);

/* Phase 2: drive both QPs INIT->RTR->RTS using the peer endpoints relayed over
 * the TCP bootstrap. Returns 0 on success, -1 on error (err filled). */
int ds4_dist_rdma_conn_connect(ds4_dist_rdma_conn *conn,
                               const ds4_dist_rdma_endpoints *remote,
                               char *err,
                               size_t errlen);

/* Wrap a connected (RTS) QP pair as a data-plane channel. Takes ownership of the
 * connection unconditionally: ds4_dist_dchan_close destroys it on success, and a
 * NULL return means it has already been destroyed, so callers must not destroy
 * it themselves. The channel provides a reliable in-order byte stream over UC,
 * safe for one concurrent reader and one concurrent writer. */
ds4_dist_dchan *ds4_dist_dchan_from_rdma_conn(ds4_dist_rdma_conn *conn, int liveness_fd);

#endif /* DS4_DISTRIBUTED_RDMA_H */
