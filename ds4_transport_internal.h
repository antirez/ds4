#ifndef DS4_TRANSPORT_INTERNAL_H
#define DS4_TRANSPORT_INTERNAL_H

#include "ds4_transport.h"

#include <pthread.h>

typedef struct ds4_transport_ops {
    const char *name;
    uint32_t caps;
    int (*send_bulk)(ds4_transport *t, const void *buf, size_t len);
    int (*recv_bulk)(ds4_transport *t, void *buf, size_t len);
    int (*send_bulk_desc)(ds4_transport *t,
                          const ds4_transport_bulk_desc *desc,
                          const void *buf,
                          size_t len);
    int (*recv_bulk_desc)(ds4_transport *t,
                          const ds4_transport_bulk_desc *desc,
                          void *buf,
                          size_t len);
    int (*mapped_leases_supported)(const ds4_transport *t);
    int (*tx_lease_acquire)(ds4_transport *t,
                            const ds4_transport_bulk_desc *desc,
                            ds4_transport_lease *lease,
                            char *err,
                            size_t errlen);
    int (*rx_lease_acquire)(ds4_transport *t,
                            const ds4_transport_bulk_desc *desc,
                            ds4_transport_lease *lease,
                            char *err,
                            size_t errlen);
    int (*tx_lease_mark_control_sent)(ds4_transport_lease *lease);
    int (*lease_commit)(ds4_transport_lease *lease, int gpu_quiesced);
    int (*lease_abort)(ds4_transport_lease *lease);
    int (*lease_abort_finish)(ds4_transport_lease *lease);
    int (*check_prepare)(ds4_transport *t);
    int (*configure)(ds4_transport *t,
                     uint32_t peer_frame_size,
                     uint32_t peer_ring_size,
                     char *err,
                     size_t errlen);
    size_t (*max_oob_bytes)(const ds4_transport *t);
    void (*close)(ds4_transport *t);
} ds4_transport_ops;

enum {
    DS4_TRANSPORT_LEASE_TX = 1,
    DS4_TRANSPORT_LEASE_RX = 2,
};

enum {
    DS4_TRANSPORT_LEASE_ACTIVE = 1,
    DS4_TRANSPORT_LEASE_COMMITTED = 2,
    DS4_TRANSPORT_LEASE_ABORTED = 3,
};

struct ds4_transport_lease {
    ds4_transport *transport;
    ds4_transport_bulk_desc desc;
    void *host_ptr;
    void *device_ptr;
    size_t bytes;
    uint32_t direction;
    uint32_t state;
    int control_sent;

    uint32_t first;
    uint32_t nframes;
    uint32_t event_bytes;
};

struct ds4_transport {
    const ds4_transport_ops *ops;
    int control_fd;
    int owns_control_fd;
    void *ctx;

    pthread_mutex_t ref_mu;
    unsigned int refs;

    pthread_mutex_t link_mu;
    pthread_mutex_t send_mu;
    pthread_mutex_t recv_mu;
    uint64_t generation;
    uint64_t tx_prepare_sequence;
    uint64_t tx_send_sequence;
    uint64_t rx_sequence;
    uint32_t local_frame_size;
    uint32_t local_ring_size;
    uint32_t peer_frame_size;
    uint32_t peer_ring_size;
    int configured;
    int fatal_error;
};

ds4_transport *ds4_transport_internal_create(
        const ds4_transport_ops *ops,
        int control_fd,
        int owns_control_fd,
        void *ctx,
        uint32_t local_frame_size,
        uint32_t local_ring_size,
        char *err,
        size_t errlen);

void ds4_transport_internal_fail(ds4_transport *t, int error_code);
int ds4_transport_internal_error(const ds4_transport *t);

#endif
