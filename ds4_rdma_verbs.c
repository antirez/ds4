/* Shared runtime loader and link bring-up for Apple's librdma (see
 * ds4_rdma_verbs.h). */

#include "ds4_rdma_verbs.h"

#ifndef DS4_RDMA_HAVE_VERBS
/* Off Apple there is no provider to load; keep this a non-empty translation
 * unit (ISO C forbids an empty one). */
typedef int ds4_rdma_verbs_no_provider;
#endif

#ifdef DS4_RDMA_HAVE_VERBS

#include <errno.h>
#include <stdio.h>
#include <string.h>

int ds4_rdma_verbs_load(ds4_rdma_verbs_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define DS4_RDMA_SYM(field, name)                                      \
    do {                                                              \
        api->field = (__typeof__(api->field))dlsym(h, name);         \
        if (!api->field) { dlclose(h); return 0; }                   \
    } while (0)
    DS4_RDMA_SYM(get_device_list, "ibv_get_device_list");
    DS4_RDMA_SYM(free_device_list, "ibv_free_device_list");
    DS4_RDMA_SYM(get_device_name, "ibv_get_device_name");
    DS4_RDMA_SYM(open_device, "ibv_open_device");
    DS4_RDMA_SYM(close_device, "ibv_close_device");
    DS4_RDMA_SYM(query_device, "ibv_query_device");
    DS4_RDMA_SYM(query_port, "ibv_query_port");
    DS4_RDMA_SYM(query_gid, "ibv_query_gid");
    DS4_RDMA_SYM(alloc_pd, "ibv_alloc_pd");
    DS4_RDMA_SYM(dealloc_pd, "ibv_dealloc_pd");
    DS4_RDMA_SYM(reg_mr, "ibv_reg_mr");
    DS4_RDMA_SYM(dereg_mr, "ibv_dereg_mr");
    DS4_RDMA_SYM(create_cq, "ibv_create_cq");
    DS4_RDMA_SYM(destroy_cq, "ibv_destroy_cq");
    DS4_RDMA_SYM(create_qp, "ibv_create_qp");
    DS4_RDMA_SYM(destroy_qp, "ibv_destroy_qp");
    DS4_RDMA_SYM(modify_qp, "ibv_modify_qp");
    DS4_RDMA_SYM(query_qp, "ibv_query_qp");
#undef DS4_RDMA_SYM
    api->handle = h;
    return 1;
}

/* RoCEv2 IPv4-mapped GID: ::ffff:a.b.c.d - over Thunderbolt the link is a
 * link-local IPv4 address surfaced as this mapped GID, and the driver connects
 * through no other. */
static int rdma_gid_is_ipv4(const union ibv_gid *g) {
    for (int i = 0; i < 10; i++) if (g->raw[i]) return 0;
    return g->raw[10] == 0xff && g->raw[11] == 0xff;
}

int ds4_rdma_open_ipv4_link(const ds4_rdma_verbs_api *api,
                            const char *dev_name,
                            int want_gid_index,
                            struct ibv_context **ctx_out,
                            struct ibv_port_attr *port_out,
                            union ibv_gid *gid_out,
                            int *gid_index_out,
                            char *resolved_name, size_t resolved_cap,
                            char *err, size_t errlen) {
    int ndev = 0;
    struct ibv_device **list = api->get_device_list(&ndev);
    if (!list || ndev <= 0) {
        if (list) api->free_device_list(list);
        if (errlen) snprintf(err, errlen, "no RDMA devices found");
        return -1;
    }
    char last_err[192];
    last_err[0] = '\0';
    int rc = -1;
    for (int i = 0; i < ndev; i++) {
        const char *name = api->get_device_name(list[i]);
        if (dev_name && dev_name[0] && strcmp(name, dev_name) != 0) continue;
        struct ibv_context *ctx = api->open_device(list[i]);
        if (!ctx) {
            snprintf(last_err, sizeof(last_err), "%s: ibv_open_device failed", name);
            continue;
        }
        /* One port per Thunderbolt device: query port 1 directly. Accept it if
         * ACTIVE, or unconditionally when the device was named explicitly. */
        struct ibv_port_attr pa;
        if (api->query_port(ctx, 1, &pa) != 0) {
            snprintf(last_err, sizeof(last_err), "%s: query_port(1) failed", name);
            api->close_device(ctx);
            continue;
        }
        if (pa.state != IBV_PORT_ACTIVE && !(dev_name && dev_name[0])) {
            snprintf(last_err, sizeof(last_err), "%s: port 1 not active (state %d); is the peer "
                     "up and rdma_ctl enabled on both machines?", name, (int)pa.state);
            api->close_device(ctx);
            continue;
        }
        union ibv_gid gid;
        int gid_index = -1;
        if (want_gid_index >= 0) {
            gid_index = want_gid_index;
            if (api->query_gid(ctx, 1, gid_index, &gid) != 0) {
                snprintf(last_err, sizeof(last_err), "%s: query_gid(%d) failed", name, gid_index);
                api->close_device(ctx);
                continue;
            }
        } else {
            for (int g = 0; g < pa.gid_tbl_len; g++) {
                union ibv_gid t;
                if (api->query_gid(ctx, 1, g, &t) != 0) continue;
                if (rdma_gid_is_ipv4(&t)) { gid = t; gid_index = g; break; }
            }
            if (gid_index < 0) {
                snprintf(last_err, sizeof(last_err), "%s: no IPv4-mapped GID on the active port; "
                         "give the Thunderbolt interface its own IPv4 (e.g. sudo ifconfig en1 inet "
                         "10.99.0.2/30 alias) on both machines", name);
                api->close_device(ctx);
                continue;
            }
        }
        *ctx_out = ctx;
        if (port_out) *port_out = pa;
        if (gid_out) *gid_out = gid;
        if (gid_index_out) *gid_index_out = gid_index;
        if (resolved_name && resolved_cap) snprintf(resolved_name, resolved_cap, "%s", name);
        rc = 0;
        break;
    }
    api->free_device_list(list);
    if (rc != 0 && errlen) {
        if (dev_name && dev_name[0]) {
            snprintf(err, errlen, "RDMA device %s not usable: %s",
                     dev_name, last_err[0] ? last_err : "not found");
        } else {
            snprintf(err, errlen, "no usable RDMA link (%s)",
                     last_err[0] ? last_err : "no active port with an IPv4-mapped GID");
        }
    }
    return rc;
}

int ds4_rdma_qp_to_init(const ds4_rdma_verbs_api *api, struct ibv_qp *qp,
                        uint8_t port_num, char *err, size_t errlen) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = port_num;
    a.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    if (api->modify_qp(qp, &a,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        if (errlen) snprintf(err, errlen, "UC QP -> INIT failed (UC may be unsupported): %s",
                             strerror(errno));
        return -1;
    }
    return 0;
}

int ds4_rdma_qp_to_rtr_rts(const ds4_rdma_verbs_api *api, struct ibv_qp *qp,
                           uint8_t port_num, int gid_index, uint32_t local_psn,
                           const uint8_t peer_gid[16], uint32_t peer_qpn,
                           uint32_t peer_psn, uint16_t peer_lid,
                           char *err, size_t errlen) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_1024;               /* the value the UC provider accepts */
    a.dest_qp_num = peer_qpn;
    a.rq_psn = peer_psn;
    a.ah_attr.dlid = peer_lid;
    a.ah_attr.port_num = port_num;
    a.ah_attr.is_global = 1;                  /* address by GID via a GRH */
    memcpy(a.ah_attr.grh.dgid.raw, peer_gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)gid_index;
    a.ah_attr.grh.hop_limit = 1;              /* point-to-point link */
    /* UC: no MAX_DEST_RD_ATOMIC / MIN_RNR_TIMER (RC-only). */
    if (api->modify_qp(qp, &a,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN) != 0) {
        if (errlen) snprintf(err, errlen, "UC QP -> RTR failed: %s", strerror(errno));
        return -1;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = local_psn;
    /* UC: no timeout / retry_cnt / rnr_retry / max_rd_atomic. */
    if (api->modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
        if (errlen) snprintf(err, errlen, "UC QP -> RTS failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

#endif /* DS4_RDMA_HAVE_VERBS */
