/* =========================================================================
 * ds4_transport.c - Persistent boundary-tensor transport core.
 * ========================================================================= */

#include "ds4_transport_internal.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define DS4_NHI_ENVELOPE_BYTES 64u

typedef char ds4_bulk_desc_must_be_64_bytes[
    sizeof(ds4_transport_bulk_desc) == DS4_TRANSPORT_BULK_DESC_BYTES ? 1 : -1];
typedef char ds4_bulk_desc_wire_must_be_64_bytes[
    sizeof(ds4_transport_bulk_desc_wire) == DS4_TRANSPORT_BULK_DESC_BYTES ? 1 : -1];

static void transport_set_err(char *err, size_t errlen, const char *message) {
    if (err && errlen) snprintf(err, errlen, "%s", message);
}

static int bulk_kind_valid(uint32_t kind, uint32_t bits) {
    switch (kind) {
    case DS4_TRANSPORT_BULK_INPUT_HIDDEN:
    case DS4_TRANSPORT_BULK_RESULT_HIDDEN:
        return bits == 8u || bits == 16u || bits == 32u;
    case DS4_TRANSPORT_BULK_RESULT_LOGITS:
        return bits == 32u;
    default:
        return 0;
    }
}

static int validate_expected_desc(
        const ds4_transport_bulk_desc *desc,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        char *err,
        size_t errlen) {
    if (!desc || !bulk_kind_valid(kind, element_bits) || payload_bytes == 0u) {
        transport_set_err(err, errlen, "invalid bulk descriptor expectation");
        errno = EINVAL;
        return -1;
    }
    if (desc->kind != kind || desc->session_id != session_id ||
        desc->request_id != request_id ||
        desc->payload_bytes != payload_bytes ||
        desc->element_bits != element_bits) {
        transport_set_err(err, errlen, "bulk descriptor metadata mismatch");
        errno = EPROTO;
        return -1;
    }
    if (!bulk_kind_valid(desc->kind, desc->element_bits) ||
        desc->generation == 0 || desc->sequence == 0 || desc->flags != 0 ||
        desc->reserved[0] != 0 || desc->reserved[1] != 0) {
        transport_set_err(err, errlen, "invalid bulk descriptor fields");
        errno = EPROTO;
        return -1;
    }
    return 0;
}

int ds4_transport_tcp_write(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    if (len != 0 && !buf) {
        errno = EINVAL;
        return -1;
    }
    while (len > 0) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        ssize_t n = send(fd, p, len, flags);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = EPIPE;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

int ds4_transport_tcp_read(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    if (len != 0 && !buf) {
        errno = EINVAL;
        return -1;
    }
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int tcp_send_bulk(ds4_transport *t, const void *buf, size_t len) {
    return ds4_transport_tcp_write(t->control_fd, buf, len);
}

static int tcp_recv_bulk(ds4_transport *t, void *buf, size_t len) {
    return ds4_transport_tcp_read(t->control_fd, buf, len);
}

static int tcp_configure(ds4_transport *t,
                         uint32_t peer_frame_size,
                         uint32_t peer_ring_size,
                         char *err,
                         size_t errlen) {
    (void)t;
    (void)peer_frame_size;
    (void)peer_ring_size;
    (void)err;
    (void)errlen;
    return 0;
}

static size_t tcp_max_oob_bytes(const ds4_transport *t) {
    (void)t;
    return 0;
}

static void tcp_close(ds4_transport *t) {
    (void)t;
}

static const ds4_transport_ops tcp_ops = {
    .name = "tcp",
    .caps = DS4_TRANSPORT_CAP_STREAM,
    .send_bulk = tcp_send_bulk,
    .recv_bulk = tcp_recv_bulk,
    .configure = tcp_configure,
    .max_oob_bytes = tcp_max_oob_bytes,
    .close = tcp_close,
};

ds4_transport *ds4_transport_internal_create(
        const ds4_transport_ops *ops,
        int control_fd,
        int owns_control_fd,
        void *ctx,
        uint32_t local_frame_size,
        uint32_t local_ring_size,
        char *err,
        size_t errlen) {
    if (!ops || control_fd < 0) {
        transport_set_err(err, errlen, "invalid transport creation arguments");
        errno = EINVAL;
        return NULL;
    }
    ds4_transport *t = calloc(1, sizeof(*t));
    if (!t) {
        transport_set_err(err, errlen, "out of memory creating transport");
        return NULL;
    }

    int rc = pthread_mutex_init(&t->ref_mu, NULL);
    if (rc != 0) goto fail_ref;
    rc = pthread_mutex_init(&t->link_mu, NULL);
    if (rc != 0) goto fail_link;
    rc = pthread_mutex_init(&t->send_mu, NULL);
    if (rc != 0) goto fail_send;
    rc = pthread_mutex_init(&t->recv_mu, NULL);
    if (rc != 0) goto fail_recv;

    t->ops = ops;
    t->control_fd = control_fd;
    t->owns_control_fd = owns_control_fd;
    t->ctx = ctx;
    t->refs = 1;
    t->local_frame_size = local_frame_size;
    t->local_ring_size = local_ring_size;
    t->tx_prepare_sequence = 1;
    t->tx_send_sequence = 1;
    t->rx_sequence = 1;
#ifdef SO_NOSIGPIPE
    {
        int one = 1;
        (void)setsockopt(control_fd, SOL_SOCKET, SO_NOSIGPIPE,
                         &one, sizeof(one));
    }
#endif
    return t;

fail_recv:
    pthread_mutex_destroy(&t->send_mu);
fail_send:
    pthread_mutex_destroy(&t->link_mu);
fail_link:
    pthread_mutex_destroy(&t->ref_mu);
fail_ref:
    free(t);
    transport_set_err(err, errlen, "failed to initialize transport mutexes");
    errno = rc;
    return NULL;
}

static ds4_transport *tcp_create(int fd, int owns_fd,
                                 char *err, size_t errlen) {
    return ds4_transport_internal_create(&tcp_ops, fd, owns_fd, NULL,
                                         0, 0, err, errlen);
}

ds4_transport *ds4_transport_tcp_create(int fd, char *err, size_t errlen) {
    return tcp_create(fd, 0, err, errlen);
}

ds4_transport *ds4_transport_tcp_dup(int fd, char *err, size_t errlen) {
    int copy = dup(fd);
    if (copy < 0) {
        transport_set_err(err, errlen, "failed to duplicate TCP transport fd");
        return NULL;
    }
    ds4_transport *t = tcp_create(copy, 1, err, errlen);
    if (!t) close(copy);
    return t;
}

ds4_transport *ds4_transport_retain(ds4_transport *t) {
    if (!t) return NULL;
    pthread_mutex_lock(&t->ref_mu);
    if (t->refs != UINT_MAX) t->refs++;
    pthread_mutex_unlock(&t->ref_mu);
    return t;
}

void ds4_transport_release(ds4_transport *t) {
    if (!t) return;
    pthread_mutex_lock(&t->ref_mu);
    const int destroy = t->refs != 0 && --t->refs == 0;
    pthread_mutex_unlock(&t->ref_mu);
    if (!destroy) return;
    if (t->ops && t->ops->close) t->ops->close(t);
    if (t->owns_control_fd && t->control_fd >= 0) close(t->control_fd);
    pthread_mutex_destroy(&t->recv_mu);
    pthread_mutex_destroy(&t->send_mu);
    pthread_mutex_destroy(&t->link_mu);
    pthread_mutex_destroy(&t->ref_mu);
    free(t);
}

int ds4_transport_control_fd(const ds4_transport *t) {
    return t ? t->control_fd : -1;
}

void ds4_transport_internal_fail(ds4_transport *t, int error_code) {
    if (!t) return;
    if (error_code <= 0) error_code = EIO;
    pthread_mutex_lock(&t->link_mu);
    if (t->fatal_error == 0) t->fatal_error = error_code;
    pthread_mutex_unlock(&t->link_mu);
    if (t->control_fd >= 0) (void)shutdown(t->control_fd, SHUT_RDWR);
}

int ds4_transport_internal_error(const ds4_transport *t) {
    if (!t) return EINVAL;
    ds4_transport *mutable_t = (ds4_transport *)t;
    pthread_mutex_lock(&mutable_t->link_mu);
    int error_code = mutable_t->fatal_error;
    pthread_mutex_unlock(&mutable_t->link_mu);
    return error_code;
}

int ds4_transport_configure_link(
        ds4_transport *t,
        uint64_t generation,
        uint32_t peer_frame_size,
        uint32_t peer_ring_size,
        char *err,
        size_t errlen) {
    if (!t || generation == 0) {
        transport_set_err(err, errlen, "invalid transport link generation");
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&t->link_mu);
    if (t->fatal_error != 0) {
        errno = t->fatal_error;
        pthread_mutex_unlock(&t->link_mu);
        transport_set_err(err, errlen, "transport link has failed");
        return -1;
    }
    if (t->configured) {
        const int same = t->generation == generation &&
            t->peer_frame_size == peer_frame_size &&
            t->peer_ring_size == peer_ring_size;
        pthread_mutex_unlock(&t->link_mu);
        if (same) return 0;
        transport_set_err(err, errlen, "transport link is already configured");
        errno = EBUSY;
        return -1;
    }
    int rc = t->ops && t->ops->configure
        ? t->ops->configure(t, peer_frame_size, peer_ring_size, err, errlen)
        : 0;
    if (rc == 0) {
        t->generation = generation;
        t->peer_frame_size = peer_frame_size;
        t->peer_ring_size = peer_ring_size;
        t->tx_prepare_sequence = 1;
        t->tx_send_sequence = 1;
        t->rx_sequence = 1;
        t->configured = 1;
    }
    pthread_mutex_unlock(&t->link_mu);
    return rc;
}

uint64_t ds4_transport_generation(const ds4_transport *t) {
    if (!t) return 0;
    ds4_transport *mutable_t = (ds4_transport *)t;
    pthread_mutex_lock(&mutable_t->link_mu);
    uint64_t generation = mutable_t->generation;
    pthread_mutex_unlock(&mutable_t->link_mu);
    return generation;
}

uint32_t ds4_transport_frame_size(const ds4_transport *t) {
    return t ? t->local_frame_size : 0;
}

uint32_t ds4_transport_ring_size(const ds4_transport *t) {
    return t ? t->local_ring_size : 0;
}

size_t ds4_transport_max_oob_bytes(const ds4_transport *t) {
    if (!t || !t->ops || !t->ops->max_oob_bytes) return 0;
    return t->ops->max_oob_bytes(t);
}

int ds4_transport_can_oob(const ds4_transport *t, size_t payload_bytes) {
    if (!t || payload_bytes == 0 ||
        (t->ops->caps & DS4_TRANSPORT_CAP_ZEROCOPY) == 0) return 0;
    ds4_transport *mutable_t = (ds4_transport *)t;
    pthread_mutex_lock(&mutable_t->link_mu);
    const int configured = mutable_t->configured &&
        mutable_t->generation != 0 && mutable_t->fatal_error == 0;
    pthread_mutex_unlock(&mutable_t->link_mu);
    if (!configured) return 0;
    const size_t max_bytes = ds4_transport_max_oob_bytes(t);
    return max_bytes != 0 && payload_bytes <= max_bytes;
}

int ds4_transport_prepare_bulk(
        ds4_transport *t,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        ds4_transport_bulk_desc *desc,
        char *err,
        size_t errlen) {
    if (!t || !desc || payload_bytes == 0 ||
        !bulk_kind_valid(kind, element_bits)) {
        transport_set_err(err, errlen, "invalid bulk descriptor request");
        errno = EINVAL;
        return -1;
    }

    memset(desc, 0, sizeof(*desc));
    pthread_mutex_lock(&t->link_mu);
    if (!t->configured || t->generation == 0 || t->fatal_error != 0 ||
        t->tx_prepare_sequence == 0 || t->tx_prepare_sequence == UINT64_MAX) {
        int saved = t->fatal_error ? t->fatal_error : EINVAL;
        pthread_mutex_unlock(&t->link_mu);
        transport_set_err(err, errlen, "transport link is not ready for bulk descriptors");
        errno = saved;
        return -1;
    }
    if (t->ops && t->ops->check_prepare && t->ops->check_prepare(t) != 0) {
        int saved = errno ? errno : EIO;
        pthread_mutex_unlock(&t->link_mu);
        transport_set_err(err, errlen,
                          "transport backend cannot prepare more bulk traffic");
        errno = saved;
        return -1;
    }
    desc->kind = kind;
    desc->generation = t->generation;
    desc->sequence = t->tx_prepare_sequence++;
    desc->session_id = session_id;
    desc->request_id = request_id;
    desc->payload_bytes = payload_bytes;
    desc->element_bits = element_bits;

    const size_t max_oob = t->ops && t->ops->max_oob_bytes
        ? t->ops->max_oob_bytes(t) : 0;
    if ((t->ops->caps & DS4_TRANSPORT_CAP_ZEROCOPY) != 0 &&
        max_oob != 0 && payload_bytes <= max_oob) {
        const uint64_t total = (uint64_t)DS4_NHI_ENVELOPE_BYTES + payload_bytes;
        desc->mode = DS4_TRANSPORT_BULK_NHI_OOB;
        desc->frame_count = (uint32_t)((total + t->local_frame_size - 1u) /
                                       t->local_frame_size);
    } else {
        desc->mode = DS4_TRANSPORT_BULK_TCP_INLINE;
    }
    pthread_mutex_unlock(&t->link_mu);
    return 0;
}

int ds4_transport_validate_inline_desc(
        const ds4_transport_bulk_desc *desc,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        char *err,
        size_t errlen) {
    if (validate_expected_desc(desc, kind, session_id, request_id,
                               payload_bytes, element_bits, err, errlen) != 0)
        return -1;
    if (desc->mode != DS4_TRANSPORT_BULK_TCP_INLINE ||
        desc->frame_count != 0) {
        transport_set_err(err, errlen, "invalid inline bulk descriptor");
        errno = EPROTO;
        return -1;
    }
    return 0;
}

int ds4_transport_validate_oob_desc(
        const ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        char *err,
        size_t errlen) {
    if (!t || validate_expected_desc(desc, kind, session_id, request_id,
                                     payload_bytes, element_bits,
                                     err, errlen) != 0)
        return -1;
    if (desc->mode != DS4_TRANSPORT_BULK_NHI_OOB ||
        (t->ops->caps & DS4_TRANSPORT_CAP_ZEROCOPY) == 0) {
        transport_set_err(err, errlen, "invalid out-of-band bulk descriptor mode");
        errno = EPROTO;
        return -1;
    }
    ds4_transport *mutable_t = (ds4_transport *)t;
    pthread_mutex_lock(&mutable_t->link_mu);
    const uint64_t generation = mutable_t->generation;
    const int configured = mutable_t->configured;
    pthread_mutex_unlock(&mutable_t->link_mu);
    const uint64_t total = (uint64_t)DS4_NHI_ENVELOPE_BYTES + payload_bytes;
    const uint32_t frames = t->local_frame_size == 0 ? 0 :
        (uint32_t)((total + t->local_frame_size - 1u) / t->local_frame_size);
    if (!configured || desc->generation != generation || frames == 0 ||
        desc->frame_count != frames || !ds4_transport_can_oob(t, payload_bytes)) {
        transport_set_err(err, errlen, "out-of-band bulk descriptor geometry mismatch");
        errno = EPROTO;
        return -1;
    }
    return 0;
}

static void wire_put_u32(uint8_t *p, uint32_t value) {
    value = htonl(value);
    memcpy(p, &value, sizeof(value));
}

static uint32_t wire_get_u32(const uint8_t *p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return ntohl(value);
}

static void wire_put_u64(uint8_t *p, uint64_t value) {
    wire_put_u32(p, (uint32_t)(value >> 32));
    wire_put_u32(p + 4, (uint32_t)value);
}

static uint64_t wire_get_u64(const uint8_t *p) {
    return ((uint64_t)wire_get_u32(p) << 32) | wire_get_u32(p + 4);
}

int ds4_transport_bulk_desc_encode(
        const ds4_transport_bulk_desc *desc,
        ds4_transport_bulk_desc_wire *wire) {
    if (!desc || !wire) {
        errno = EINVAL;
        return -1;
    }
    wire_put_u32(wire->bytes + 0, desc->mode);
    wire_put_u32(wire->bytes + 4, desc->kind);
    wire_put_u64(wire->bytes + 8, desc->generation);
    wire_put_u64(wire->bytes + 16, desc->sequence);
    wire_put_u64(wire->bytes + 24, desc->session_id);
    wire_put_u64(wire->bytes + 32, desc->request_id);
    wire_put_u32(wire->bytes + 40, desc->payload_bytes);
    wire_put_u32(wire->bytes + 44, desc->element_bits);
    wire_put_u32(wire->bytes + 48, desc->frame_count);
    wire_put_u32(wire->bytes + 52, desc->flags);
    wire_put_u32(wire->bytes + 56, desc->reserved[0]);
    wire_put_u32(wire->bytes + 60, desc->reserved[1]);
    return 0;
}

int ds4_transport_bulk_desc_decode(
        const ds4_transport_bulk_desc_wire *wire,
        ds4_transport_bulk_desc *desc) {
    if (!wire || !desc) {
        errno = EINVAL;
        return -1;
    }
    memset(desc, 0, sizeof(*desc));
    desc->mode = wire_get_u32(wire->bytes + 0);
    desc->kind = wire_get_u32(wire->bytes + 4);
    desc->generation = wire_get_u64(wire->bytes + 8);
    desc->sequence = wire_get_u64(wire->bytes + 16);
    desc->session_id = wire_get_u64(wire->bytes + 24);
    desc->request_id = wire_get_u64(wire->bytes + 32);
    desc->payload_bytes = wire_get_u32(wire->bytes + 40);
    desc->element_bits = wire_get_u32(wire->bytes + 44);
    desc->frame_count = wire_get_u32(wire->bytes + 48);
    desc->flags = wire_get_u32(wire->bytes + 52);
    desc->reserved[0] = wire_get_u32(wire->bytes + 56);
    desc->reserved[1] = wire_get_u32(wire->bytes + 60);
    return 0;
}

int ds4_transport_send_bulk(ds4_transport *t, const void *buf, size_t len) {
    if (!t || !t->ops || !t->ops->send_bulk || (len != 0 && !buf)) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&t->send_mu);
    int rc = t->ops->send_bulk(t, buf, len);
    pthread_mutex_unlock(&t->send_mu);
    return rc;
}

int ds4_transport_recv_bulk(ds4_transport *t, void *buf, size_t len) {
    if (!t || !t->ops || !t->ops->recv_bulk || (len != 0 && !buf)) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&t->recv_mu);
    int rc = t->ops->recv_bulk(t, buf, len);
    pthread_mutex_unlock(&t->recv_mu);
    return rc;
}

static int validate_directional_desc(ds4_transport *t,
                                     const ds4_transport_bulk_desc *desc,
                                     size_t len,
                                     int sending) {
    if (!t || !desc || len != desc->payload_bytes ||
        desc->payload_bytes == 0 ||
        !bulk_kind_valid(desc->kind, desc->element_bits) ||
        desc->flags != 0 || desc->reserved[0] != 0 ||
        desc->reserved[1] != 0) {
        errno = EPROTO;
        return -1;
    }
    pthread_mutex_lock(&t->link_mu);
    const uint64_t expected = sending ? t->tx_send_sequence : t->rx_sequence;
    const int valid = t->configured && t->fatal_error == 0 &&
        desc->generation == t->generation && desc->sequence == expected;
    int saved = t->fatal_error ? t->fatal_error : EPROTO;
    pthread_mutex_unlock(&t->link_mu);
    if (!valid) {
        errno = saved;
        return -1;
    }
    if (desc->mode == DS4_TRANSPORT_BULK_TCP_INLINE) {
        if (desc->frame_count == 0) return 0;
        errno = EPROTO;
        return -1;
    }
    if (desc->mode == DS4_TRANSPORT_BULK_NHI_OOB)
        return ds4_transport_validate_oob_desc(t, desc, desc->kind,
                                               desc->session_id,
                                               desc->request_id,
                                               desc->payload_bytes,
                                               desc->element_bits,
                                               NULL, 0);
    errno = EPROTO;
    return -1;
}

int ds4_transport_send_bulk_desc(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        const void *buf,
        size_t len) {
    if (!t || !desc || (len != 0 && !buf)) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&t->send_mu);
    int rc = validate_directional_desc(t, desc, len, 1);
    if (rc == 0) {
        if (desc->mode == DS4_TRANSPORT_BULK_TCP_INLINE) {
            rc = ds4_transport_tcp_write(t->control_fd, buf, len);
        } else if (!t->ops || !t->ops->send_bulk_desc) {
            errno = ENOTSUP;
            rc = -1;
        } else {
            rc = t->ops->send_bulk_desc(t, desc, buf, len);
        }
    }
    if (rc == 0) {
        pthread_mutex_lock(&t->link_mu);
        t->tx_send_sequence++;
        pthread_mutex_unlock(&t->link_mu);
    } else {
        int saved = errno ? errno : EIO;
        ds4_transport_internal_fail(t, saved);
        errno = saved;
    }
    pthread_mutex_unlock(&t->send_mu);
    return rc;
}

int ds4_transport_recv_bulk_desc(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        void *buf,
        size_t len) {
    if (!t || !desc || (len != 0 && !buf)) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&t->recv_mu);
    int rc = validate_directional_desc(t, desc, len, 0);
    if (rc == 0) {
        if (desc->mode == DS4_TRANSPORT_BULK_TCP_INLINE) {
            rc = ds4_transport_tcp_read(t->control_fd, buf, len);
        } else if (!t->ops || !t->ops->recv_bulk_desc) {
            errno = ENOTSUP;
            rc = -1;
        } else {
            rc = t->ops->recv_bulk_desc(t, desc, buf, len);
        }
    }
    if (rc == 1) {
        pthread_mutex_lock(&t->link_mu);
        t->rx_sequence++;
        pthread_mutex_unlock(&t->link_mu);
    } else if (rc <= 0) {
        int saved = rc == 0 ? ECONNRESET : (errno ? errno : EIO);
        ds4_transport_internal_fail(t, saved);
        errno = saved;
    }
    pthread_mutex_unlock(&t->recv_mu);
    return rc;
}

int ds4_transport_mapped_leases_supported(const ds4_transport *t) {
    return t && t->ops && t->ops->mapped_leases_supported
        ? t->ops->mapped_leases_supported(t) : 0;
}

static int transport_lease_acquire(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        ds4_transport_lease **out,
        uint32_t direction,
        char *err,
        size_t errlen) {
    if (!out) {
        transport_set_err(err, errlen, "missing mapped lease output");
        errno = EINVAL;
        return -1;
    }
    *out = NULL;
    if (!t || !desc ||
        (direction != DS4_TRANSPORT_LEASE_TX &&
         direction != DS4_TRANSPORT_LEASE_RX)) {
        transport_set_err(err, errlen, "invalid mapped lease request");
        errno = EINVAL;
        return -1;
    }
    if (desc->mode != DS4_TRANSPORT_BULK_NHI_OOB ||
        !ds4_transport_mapped_leases_supported(t)) {
        transport_set_err(err, errlen,
                          "mapped leases are unavailable for this descriptor");
        errno = ENOTSUP;
        return -1;
    }

    ds4_transport_lease *lease = calloc(1, sizeof(*lease));
    if (!lease) {
        transport_set_err(err, errlen, "out of memory creating mapped lease");
        return -1;
    }
    lease->transport = ds4_transport_retain(t);
    lease->desc = *desc;
    lease->bytes = desc->payload_bytes;
    lease->direction = direction;
    lease->state = DS4_TRANSPORT_LEASE_ACTIVE;

    pthread_mutex_t *direction_mu = direction == DS4_TRANSPORT_LEASE_TX
        ? &t->send_mu : &t->recv_mu;
    pthread_mutex_lock(direction_mu);
    int rc = validate_directional_desc(t, desc, desc->payload_bytes,
                                       direction == DS4_TRANSPORT_LEASE_TX);
    if (rc == 0) {
        if (direction == DS4_TRANSPORT_LEASE_TX) {
            if (!t->ops->tx_lease_acquire) {
                errno = ENOTSUP;
                rc = -1;
            } else {
                rc = t->ops->tx_lease_acquire(t, desc, lease, err, errlen);
            }
        } else {
            if (!t->ops->rx_lease_acquire) {
                errno = ENOTSUP;
                rc = -1;
            } else {
                rc = t->ops->rx_lease_acquire(t, desc, lease, err, errlen);
            }
        }
    }
    pthread_mutex_unlock(direction_mu);

    if (rc != 0) {
        const int saved = errno ? errno : EIO;
        /* A received descriptor is already committed by the peer. Invalid
         * metadata/event failures are fatal, but ENOTSUP deliberately leaves
         * the event untouched so the copy receive path can consume it. */
        if (direction == DS4_TRANSPORT_LEASE_RX && saved != ENOTSUP &&
            saved != EBUSY)
            ds4_transport_internal_fail(t, saved);
        ds4_transport_release(lease->transport);
        free(lease);
        errno = saved;
        return -1;
    }
    *out = lease;
    return 0;
}

int ds4_transport_tx_lease_acquire(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        ds4_transport_lease **out,
        char *err,
        size_t errlen) {
    return transport_lease_acquire(t, desc, out, DS4_TRANSPORT_LEASE_TX,
                                   err, errlen);
}

int ds4_transport_rx_lease_acquire(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        ds4_transport_lease **out,
        char *err,
        size_t errlen) {
    return transport_lease_acquire(t, desc, out, DS4_TRANSPORT_LEASE_RX,
                                   err, errlen);
}

void *ds4_transport_lease_host_ptr(const ds4_transport_lease *lease) {
    return lease && lease->state == DS4_TRANSPORT_LEASE_ACTIVE
        ? lease->host_ptr : NULL;
}

void *ds4_transport_lease_device_ptr(const ds4_transport_lease *lease) {
    return lease && lease->state == DS4_TRANSPORT_LEASE_ACTIVE
        ? lease->device_ptr : NULL;
}

size_t ds4_transport_lease_bytes(const ds4_transport_lease *lease) {
    return lease && lease->state == DS4_TRANSPORT_LEASE_ACTIVE
        ? lease->bytes : 0;
}

int ds4_transport_tx_lease_mark_control_sent(ds4_transport_lease *lease) {
    if (!lease || lease->state != DS4_TRANSPORT_LEASE_ACTIVE ||
        lease->direction != DS4_TRANSPORT_LEASE_TX || !lease->transport ||
        !lease->transport->ops ||
        !lease->transport->ops->tx_lease_mark_control_sent) {
        errno = EINVAL;
        return -1;
    }
    ds4_transport *t = lease->transport;
    /* This call is the caller's assertion that the complete control record
     * has already been sent. Preserve that irreversible fact even if the
     * backend reports a simultaneous failure while recording it. */
    lease->control_sent = 1;
    pthread_mutex_lock(&t->send_mu);
    int rc = t->ops->tx_lease_mark_control_sent(lease);
    pthread_mutex_unlock(&t->send_mu);
    if (rc != 0) {
        int saved = errno ? errno : EIO;
        /* The caller invokes this only after the TCP control record is fully
         * visible to the peer, so inability to mark it is already fatal. */
        ds4_transport_internal_fail(t, saved);
        errno = saved;
    }
    return rc;
}

static int transport_advance_lease_sequence(ds4_transport_lease *lease) {
    ds4_transport *t = lease->transport;
    pthread_mutex_lock(&t->link_mu);
    uint64_t *sequence = lease->direction == DS4_TRANSPORT_LEASE_TX
        ? &t->tx_send_sequence : &t->rx_sequence;
    const int valid = t->configured && t->fatal_error == 0 &&
        t->generation == lease->desc.generation &&
        *sequence == lease->desc.sequence;
    if (valid) (*sequence)++;
    int saved = t->fatal_error ? t->fatal_error : EPROTO;
    pthread_mutex_unlock(&t->link_mu);
    if (!valid) {
        errno = saved;
        return -1;
    }
    return 0;
}

static int transport_lease_commit_impl(ds4_transport_lease *lease,
                                       int gpu_quiesced) {
    if (!lease || lease->state != DS4_TRANSPORT_LEASE_ACTIVE ||
        !lease->transport || !lease->transport->ops ||
        !lease->transport->ops->lease_commit) {
        errno = EINVAL;
        return -1;
    }
    if (lease->direction == DS4_TRANSPORT_LEASE_TX &&
        !lease->control_sent) {
        errno = EPERM;
        return -1;
    }

    ds4_transport *t = lease->transport;
    pthread_mutex_t *direction_mu =
        lease->direction == DS4_TRANSPORT_LEASE_TX
        ? &t->send_mu : &t->recv_mu;
    pthread_mutex_lock(direction_mu);
    int rc = validate_directional_desc(
        t, &lease->desc, lease->desc.payload_bytes,
        lease->direction == DS4_TRANSPORT_LEASE_TX);
    if (rc == 0) {
        rc = t->ops->lease_commit(lease, gpu_quiesced);
    } else {
        const int validation_error = errno ? errno : EPROTO;
        /* Drop the backend's active pointer even when a concurrent fatal
         * condition made the core descriptor check fail before commit. */
        (void)t->ops->lease_abort(lease);
        errno = validation_error;
    }
    if (rc == 0) rc = transport_advance_lease_sequence(lease);
    if (rc == 0) {
        lease->state = DS4_TRANSPORT_LEASE_COMMITTED;
    } else {
        int saved = errno ? errno : EIO;
        /* RX descriptors and marked TX descriptors are already committed on
         * TCP. Any ownership, submit, repost, or sequence failure abandons
         * the generation; there is no safe inline retry. */
        ds4_transport_internal_fail(t, saved);
        lease->state = DS4_TRANSPORT_LEASE_ABORTED;
        errno = saved;
    }
    lease->host_ptr = NULL;
    lease->device_ptr = NULL;
    lease->bytes = 0;
    pthread_mutex_unlock(direction_mu);
    return rc;
}

int ds4_transport_lease_commit(ds4_transport_lease *lease) {
    return transport_lease_commit_impl(lease, 0);
}

int ds4_transport_lease_commit_gpu_quiesced(ds4_transport_lease *lease) {
    return transport_lease_commit_impl(lease, 1);
}

int ds4_transport_lease_abort(ds4_transport_lease *lease) {
    if (!lease || lease->state != DS4_TRANSPORT_LEASE_ACTIVE ||
        !lease->transport || !lease->transport->ops ||
        !lease->transport->ops->lease_abort ||
        (lease->direction == DS4_TRANSPORT_LEASE_TX &&
         !lease->control_sent &&
         !lease->transport->ops->lease_abort_finish)) {
        errno = EINVAL;
        return -1;
    }
    ds4_transport *t = lease->transport;
    pthread_mutex_t *direction_mu =
        lease->direction == DS4_TRANSPORT_LEASE_TX
        ? &t->send_mu : &t->recv_mu;
    const int recoverable = lease->direction == DS4_TRANSPORT_LEASE_TX &&
        !lease->control_sent;

    pthread_mutex_lock(direction_mu);
    int rc = t->ops->lease_abort(lease);
    if (rc == 0 && recoverable) {
        pthread_mutex_lock(&t->link_mu);
        const int can_rollback = t->configured && t->fatal_error == 0 &&
            t->generation == lease->desc.generation &&
            t->tx_send_sequence == lease->desc.sequence &&
            lease->desc.sequence != UINT64_MAX &&
            t->tx_prepare_sequence == lease->desc.sequence + 1u;
        /* The backend keeps its active marker until this point. Holding the
         * link lock while clearing it closes the check_prepare(ctx) race: a
         * new prepare observes either the old active lease or the rolled-back
         * sequence, never the transient state between them. */
        const int finish_rc = t->ops->lease_abort_finish(lease);
        if (can_rollback && finish_rc == 0)
            t->tx_prepare_sequence = lease->desc.sequence;
        pthread_mutex_unlock(&t->link_mu);
        if (!can_rollback || finish_rc != 0) {
            if (finish_rc == 0 || errno == 0) errno = EPROTO;
            rc = -1;
        }
    }
    if (rc != 0 || !recoverable) {
        int saved = errno ? errno : ECANCELED;
        ds4_transport_internal_fail(t, saved);
        errno = saved;
        rc = -1;
    }
    lease->state = DS4_TRANSPORT_LEASE_ABORTED;
    lease->host_ptr = NULL;
    lease->device_ptr = NULL;
    lease->bytes = 0;
    pthread_mutex_unlock(direction_mu);
    return rc;
}

void ds4_transport_lease_release(ds4_transport_lease *lease) {
    if (!lease) return;
    if (lease->state == DS4_TRANSPORT_LEASE_ACTIVE)
        (void)ds4_transport_lease_abort(lease);
    ds4_transport_release(lease->transport);
    free(lease);
}

const char *ds4_transport_name(const ds4_transport *t) {
    return t && t->ops ? t->ops->name : "none";
}

uint32_t ds4_transport_caps(const ds4_transport *t) {
    uint32_t caps = t && t->ops ? t->ops->caps : 0;
    if (ds4_transport_mapped_leases_supported(t))
        caps |= DS4_TRANSPORT_CAP_GPU_MAPPED;
    return caps;
}
