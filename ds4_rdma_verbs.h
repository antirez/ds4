#ifndef DS4_RDMA_VERBS_H
#define DS4_RDMA_VERBS_H

/* Shared runtime loader for Apple's RDMA-over-Thunderbolt provider (librdma).
 *
 * Both RDMA transports use it: tensor-parallel gate exchange (ds4_tp.c) and the
 * pipeline-parallel data plane (ds4_distributed_rdma.c). librdma is loaded with
 * dlopen at runtime, not linked, so a normal build runs everywhere and simply
 * falls back to TCP on a machine without the provider (see Apple TN3205). Only
 * the setup/teardown verbs go through this table; the data-plane calls
 * (ibv_post_send/ibv_post_recv/ibv_poll_cq) are header inlines over
 * context->ops and are called directly.
 *
 * Compiled only where <infiniband/verbs.h> is present (Apple SDK); elsewhere
 * DS4_RDMA_HAVE_VERBS is undefined and the RDMA paths compile out.
 */

#if defined(__APPLE__) && defined(__has_include)
#  if __has_include(<infiniband/verbs.h>)
#    include <infiniband/verbs.h>
#    include <dlfcn.h>
#    define DS4_RDMA_HAVE_VERBS 1
#  endif
#endif

#ifdef DS4_RDMA_HAVE_VERBS

/* dlopen handle plus the librdma entry points we resolve. Field names are the
 * verbs function names minus the ibv_ prefix. */
typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    int (*query_qp)(struct ibv_qp *, struct ibv_qp_attr *, int, struct ibv_qp_init_attr *);
} ds4_rdma_verbs_api;

/* Resolve librdma into *api. Idempotent (api->handle guards a reload). Returns 1
 * if the provider and every symbol are present, 0 otherwise (caller falls back
 * to TCP). Never aborts. */
int ds4_rdma_verbs_load(ds4_rdma_verbs_api *api);

/* Apple RDMA-over-Thunderbolt link bring-up, shared by the tensor-parallel
 * (ds4_tp.c) and pipeline-parallel (ds4_distributed_rdma.c) transports. All three
 * encode the one recipe the provider accepts: a device with an ACTIVE port 1, the
 * RoCEv2 IPv4-mapped GID, and a UC QP taken to RTS with GRH addressing at
 * IBV_MTU_1024 (see Apple TN3205). */

/* Open the device whose port 1 is ACTIVE and carries the IPv4-mapped GID.
 * dev_name NULL picks the first usable device; want_gid_index >= 0 forces a GID
 * index, else the IPv4-mapped one is auto-selected. Fills ctx/port/gid/gid_index
 * and, when resolved_name != NULL, the chosen device name. The caller owns
 * *ctx_out on success. Returns 0, or -1 (err filled). */
int ds4_rdma_open_ipv4_link(const ds4_rdma_verbs_api *api,
                            const char *dev_name,
                            int want_gid_index,
                            struct ibv_context **ctx_out,
                            struct ibv_port_attr *port_out,
                            union ibv_gid *gid_out,
                            int *gid_index_out,
                            char *resolved_name, size_t resolved_cap,
                            char *err, size_t errlen);

/* UC QP -> INIT: pkey 0, the given port, LOCAL_WRITE|REMOTE_READ|REMOTE_WRITE
 * (the provider wants the remote flags even for two-sided SEND/RECV). Returns 0,
 * or -1 (err filled). */
int ds4_rdma_qp_to_init(const ds4_rdma_verbs_api *api, struct ibv_qp *qp,
                        uint8_t port_num, char *err, size_t errlen);

/* UC QP INIT -> RTR -> RTS with the provider recipe: IBV_MTU_1024, GRH global
 * addressing via the IPv4-mapped GID (sgid_index), hop_limit 1, no RC-only
 * attributes. peer_* come from the endpoint relayed over the TCP bootstrap; the
 * MR must already be registered (the provider maps it on the RTR transition).
 * Returns 0, or -1 (err filled). */
int ds4_rdma_qp_to_rtr_rts(const ds4_rdma_verbs_api *api, struct ibv_qp *qp,
                           uint8_t port_num, int gid_index, uint32_t local_psn,
                           const uint8_t peer_gid[16], uint32_t peer_qpn,
                           uint32_t peer_psn, uint16_t peer_lid,
                           char *err, size_t errlen);

#endif /* DS4_RDMA_HAVE_VERBS */

#endif /* DS4_RDMA_VERBS_H */
