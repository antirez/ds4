/* =========================================================================
 * ds4_transport_nhi.c - CPU-copy USB4STREAM zero-copy-slot transport.
 * ========================================================================= */

#include "ds4_transport_internal.h"
#include "ds4_gpu.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __linux__

#include "ds4_tbstream_uapi.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DS4_NHI_ENVELOPE_MAGIC 0x44344e48u /* D4NH */
#define DS4_NHI_ENVELOPE_VERSION 1u
#define DS4_NHI_ENVELOPE_BYTES 64u
#define DS4_NHI_REAP_BATCH 64u
#define DS4_NHI_DEFAULT_TIMEOUT_SEC 30u

typedef char ds4_nhi_envelope_is_64_bytes[
    DS4_NHI_ENVELOPE_BYTES == DS4_TRANSPORT_BULK_DESC_BYTES ? 1 : -1];

typedef struct {
    uint32_t first;
    uint32_t nframes;
    uint64_t sequence;
} ds4_nhi_pending_tx;

typedef struct {
    int device_fd;
    int wake_fd;
    void *mapping;
    size_t mapping_bytes;
    unsigned char *tx_pool;
    unsigned char *rx_pool;
    void *gpu_mapping;
    size_t pool_bytes;
    uint32_t frame_size;
    uint32_t ring_size;
    uint32_t tx_frame_limit;
    uint32_t timeout_sec;

    pthread_mutex_t mu;
    pthread_cond_t tx_cv;
    pthread_cond_t rx_cv;
    pthread_t dispatcher;
    int dispatcher_started;
    int stopping;
    int failed;
    int peer_closed;
    int activated;
    int gpu_mapping_registered;

    ds4_transport_lease *active_tx_lease;
    ds4_transport_lease *active_rx_lease;

    uint32_t tx_next;
    uint32_t tx_inflight;
    ds4_nhi_pending_tx *tx_pending;
    uint32_t tx_head;
    uint32_t tx_tail;
    uint32_t tx_count;

    struct tbstream_zc_event *rx_events;
    uint32_t rx_head;
    uint32_t rx_tail;
    uint32_t rx_count;
    uint32_t rx_next;
    uint32_t rx_event_next;

    ds4_transport *owner;
} ds4_nhi_ctx;

static int nhi_trace_enabled(void) {
    const char *value = getenv("DS4_DIST_NHI_TRACE");
    return value && value[0] && strcmp(value, "0") != 0;
}

/* Direct GPU access to NHI-owned pages is deliberately opt-in. The ordinary
 * NHI backend still uses the mmap pools and avoids the socket stack, but
 * stages model tensors through CPU copies. Model-independent mapping gates are
 * useful qualification coverage; they are not sufficient evidence that a
 * platform's GPU/NHI peer-DMA interaction is reliable under model load. */
static int nhi_mapped_model_io_enabled(void) {
    const char *value = getenv("DS4_DIST_NHI_MAPPED");
    return value &&
        (!strcmp(value, "1") || !strcmp(value, "true") ||
         !strcmp(value, "yes") || !strcmp(value, "on"));
}

static void nhi_trace(ds4_nhi_ctx *ctx, const char *format, ...) {
    if (!nhi_trace_enabled()) return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    fprintf(stderr, "ds4: NHI trace: %lld.%06ld ctx=%p ",
            (long long)now.tv_sec, now.tv_nsec / 1000, (void *)ctx);
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
}

static void nhi_set_err(char *err, size_t errlen, const char *message) {
    if (err && errlen) snprintf(err, errlen, "%s", message);
}

static uint32_t nhi_timeout_sec(void) {
    const char *text = getenv("DS4_DIST_NHI_TIMEOUT_SEC");
    if (!text || !text[0]) return DS4_NHI_DEFAULT_TIMEOUT_SEC;
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value == 0 || value > 3600) return DS4_NHI_DEFAULT_TIMEOUT_SEC;
    return (uint32_t)value;
}

static void nhi_wire_put_u32(unsigned char *p, uint32_t value) {
    value = htonl(value);
    memcpy(p, &value, sizeof(value));
}

static uint32_t nhi_wire_get_u32(const unsigned char *p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return ntohl(value);
}

static void nhi_wire_put_u64(unsigned char *p, uint64_t value) {
    nhi_wire_put_u32(p, (uint32_t)(value >> 32));
    nhi_wire_put_u32(p + 4, (uint32_t)value);
}

static uint64_t nhi_wire_get_u64(const unsigned char *p) {
    return ((uint64_t)nhi_wire_get_u32(p) << 32) |
           nhi_wire_get_u32(p + 4);
}

/* The NHI message repeats every identity field from the TCP descriptor. The
 * UAPI event has no application tag, so accepting an event based on FIFO
 * position alone would make stale or malformed traffic indistinguishable. */
static void nhi_envelope_encode(
        const ds4_transport_bulk_desc *desc,
        unsigned char envelope[DS4_NHI_ENVELOPE_BYTES]) {
    memset(envelope, 0, DS4_NHI_ENVELOPE_BYTES);
    nhi_wire_put_u32(envelope + 0, DS4_NHI_ENVELOPE_MAGIC);
    nhi_wire_put_u32(envelope + 4, DS4_NHI_ENVELOPE_VERSION);
    nhi_wire_put_u32(envelope + 8, desc->kind);
    nhi_wire_put_u32(envelope + 12, desc->element_bits);
    nhi_wire_put_u64(envelope + 16, desc->generation);
    nhi_wire_put_u64(envelope + 24, desc->sequence);
    nhi_wire_put_u64(envelope + 32, desc->session_id);
    nhi_wire_put_u64(envelope + 40, desc->request_id);
    nhi_wire_put_u32(envelope + 48, desc->payload_bytes);
    nhi_wire_put_u32(envelope + 52, desc->frame_count);
    nhi_wire_put_u32(envelope + 56, desc->flags);
    nhi_wire_put_u32(envelope + 60, 0);
}

static int nhi_envelope_matches(
        const unsigned char envelope[DS4_NHI_ENVELOPE_BYTES],
        const ds4_transport_bulk_desc *desc) {
    return nhi_wire_get_u32(envelope + 0) == DS4_NHI_ENVELOPE_MAGIC &&
        nhi_wire_get_u32(envelope + 4) == DS4_NHI_ENVELOPE_VERSION &&
        nhi_wire_get_u32(envelope + 8) == desc->kind &&
        nhi_wire_get_u32(envelope + 12) == desc->element_bits &&
        nhi_wire_get_u64(envelope + 16) == desc->generation &&
        nhi_wire_get_u64(envelope + 24) == desc->sequence &&
        nhi_wire_get_u64(envelope + 32) == desc->session_id &&
        nhi_wire_get_u64(envelope + 40) == desc->request_id &&
        nhi_wire_get_u32(envelope + 48) == desc->payload_bytes &&
        nhi_wire_get_u32(envelope + 52) == desc->frame_count &&
        nhi_wire_get_u32(envelope + 56) == desc->flags &&
        nhi_wire_get_u32(envelope + 60) == 0;
}

static void nhi_ring_copy_in(ds4_nhi_ctx *ctx,
                             uint32_t first,
                             size_t message_offset,
                             const void *source,
                             size_t bytes) {
    const unsigned char *src = source;
    size_t offset = (size_t)first * ctx->frame_size + message_offset;
    offset %= ctx->pool_bytes;
    while (bytes != 0) {
        size_t chunk = ctx->pool_bytes - offset;
        if (chunk > bytes) chunk = bytes;
        memcpy(ctx->tx_pool + offset, src, chunk);
        src += chunk;
        bytes -= chunk;
        offset = 0;
    }
}

static void nhi_ring_copy_out(ds4_nhi_ctx *ctx,
                              uint32_t first,
                              size_t message_offset,
                              void *destination,
                              size_t bytes) {
    unsigned char *dst = destination;
    size_t offset = (size_t)first * ctx->frame_size + message_offset;
    offset %= ctx->pool_bytes;
    while (bytes != 0) {
        size_t chunk = ctx->pool_bytes - offset;
        if (chunk > bytes) chunk = bytes;
        memcpy(dst, ctx->rx_pool + offset, chunk);
        dst += chunk;
        bytes -= chunk;
        offset = 0;
    }
}

static void nhi_broadcast_locked(ds4_nhi_ctx *ctx) {
    pthread_cond_broadcast(&ctx->tx_cv);
    pthread_cond_broadcast(&ctx->rx_cv);
}

static int nhi_fail_locked(ds4_nhi_ctx *ctx, int error_code) {
    if (error_code <= 0) error_code = EIO;
    if (ctx->failed == 0) ctx->failed = error_code;
    nhi_broadcast_locked(ctx);
    return ctx->failed;
}

static void nhi_fail(ds4_nhi_ctx *ctx, int error_code) {
    pthread_mutex_lock(&ctx->mu);
    int saved = nhi_fail_locked(ctx, error_code);
    const int activated = ctx->activated;
    pthread_mutex_unlock(&ctx->mu);
    /* An opened endpoint is only a negotiation candidate until configure.
     * Candidate failure must not tear down the shared TCP HELLO socket: AUTO
     * may still select descriptor-framed TCP before the ACK commits NHI. */
    if (activated) ds4_transport_internal_fail(ctx->owner, saved);
}

static int nhi_event_rx_valid(ds4_nhi_ctx *ctx,
                              const struct tbstream_zc_event *ev) {
    if (ev->nframes == 0 || ev->nframes > ctx->tx_frame_limit ||
        ev->first >= ctx->ring_size || ev->first != ctx->rx_event_next)
        return 0;
    const uint64_t minimum = (uint64_t)(ev->nframes - 1u) * ctx->frame_size + 1u;
    const uint64_t maximum = (uint64_t)ev->nframes * ctx->frame_size;
    return ev->bytes >= minimum && ev->bytes <= maximum;
}

static int nhi_dispatch_event(ds4_nhi_ctx *ctx,
                              const struct tbstream_zc_event *ev) {
    int failure = 0;
    int activated = 0;
    pthread_mutex_lock(&ctx->mu);
    nhi_trace(ctx,
              "event type=%u first=%u nframes=%u bytes=%u tx_next=%u "
              "tx_inflight=%u tx_count=%u rx_next=%u rx_event_next=%u "
              "rx_count=%u",
              ev->type, ev->first, ev->nframes, ev->bytes,
              ctx->tx_next, ctx->tx_inflight, ctx->tx_count,
              ctx->rx_next, ctx->rx_event_next, ctx->rx_count);
    if (ctx->stopping || ctx->failed) {
        const int already_failed = ctx->failed;
        pthread_mutex_unlock(&ctx->mu);
        if (already_failed) errno = already_failed;
        return already_failed ? -1 : 0;
    }
    switch (ev->type) {
    case TBSTREAM_ZC_EV_TX_DONE:
        if (ev->bytes != 0 || ctx->tx_count == 0) {
            failure = nhi_fail_locked(ctx, EPROTO);
            break;
        }
        {
            ds4_nhi_pending_tx *pending = &ctx->tx_pending[ctx->tx_head];
            if (ev->first != pending->first ||
                ev->nframes != pending->nframes ||
                ev->nframes > ctx->tx_inflight) {
                failure = nhi_fail_locked(ctx, EPROTO);
                break;
            }
            ctx->tx_inflight -= ev->nframes;
            ctx->tx_head = (ctx->tx_head + 1u) % ctx->ring_size;
            ctx->tx_count--;
            pthread_cond_broadcast(&ctx->tx_cv);
        }
        break;
    case TBSTREAM_ZC_EV_RX:
        if (!nhi_event_rx_valid(ctx, ev) || ctx->rx_count >= ctx->ring_size) {
            failure = nhi_fail_locked(ctx, EPROTO);
            break;
        }
        ctx->rx_events[ctx->rx_tail] = *ev;
        ctx->rx_tail = (ctx->rx_tail + 1u) % ctx->ring_size;
        ctx->rx_count++;
        ctx->rx_event_next = (ctx->rx_event_next + ev->nframes) % ctx->ring_size;
        pthread_cond_broadcast(&ctx->rx_cv);
        break;
    case TBSTREAM_ZC_EV_CLOSE:
        ctx->peer_closed = 1;
        nhi_broadcast_locked(ctx);
        break;
    default:
        failure = nhi_fail_locked(ctx, EPROTO);
        break;
    }
    activated = ctx->activated;
    pthread_mutex_unlock(&ctx->mu);

    if (failure && activated)
        ds4_transport_internal_fail(ctx->owner, failure);
    return failure ? -1 : 0;
}

static int nhi_reap_available(ds4_nhi_ctx *ctx) {
    struct tbstream_zc_event events[DS4_NHI_REAP_BATCH];
    for (;;) {
        struct tbstream_zc_reap reap;
        memset(&reap, 0, sizeof(reap));
        reap.max = DS4_NHI_REAP_BATCH;
        reap.flags = TBSTREAM_ZC_REAP_NONBLOCK;
        reap.events = (uint64_t)(uintptr_t)events;
        int count = ioctl(ctx->device_fd, TBSTREAM_ZC_REAP, &reap);
        if (count < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) return 0;
            return -1;
        }
        if (count <= 0 || count > (int)DS4_NHI_REAP_BATCH) {
            errno = EPROTO;
            return -1;
        }
        for (int i = 0; i < count; i++) {
            if (nhi_dispatch_event(ctx, &events[i]) != 0) return -1;
        }
    }
}

static void *nhi_dispatch_main(void *arg) {
    ds4_nhi_ctx *ctx = arg;
    for (;;) {
        struct pollfd fds[3];
        memset(fds, 0, sizeof(fds));
        fds[0].fd = ctx->device_fd;
        fds[0].events = POLLIN;
        fds[1].fd = ctx->wake_fd;
        fds[1].events = POLLIN;
        fds[2].fd = ctx->owner->control_fd;
#ifdef POLLRDHUP
        fds[2].events = POLLRDHUP;
#endif
        int rc = poll(fds, 3, -1);
        if (rc < 0) {
            if (errno == EINTR) continue;
            nhi_fail(ctx, errno);
            break;
        }
        if (fds[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) break;
        pthread_mutex_lock(&ctx->mu);
        int stopping = ctx->stopping;
        pthread_mutex_unlock(&ctx->mu);
        if (stopping) break;

        if (fds[2].revents & (POLLERR | POLLHUP | POLLNVAL
#ifdef POLLRDHUP
                              | POLLRDHUP
#endif
                              )) {
            nhi_fail(ctx, ECONNRESET);
            break;
        }
        if (fds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            nhi_fail(ctx, ENXIO);
            break;
        }
        if ((fds[0].revents & POLLIN) && nhi_reap_available(ctx) != 0) {
            int saved = errno ? errno : EIO;
            nhi_fail(ctx, saved);
            break;
        }
    }
    return NULL;
}

static int nhi_raw_send(ds4_transport *t, const void *buf, size_t len) {
    return ds4_transport_tcp_write(t->control_fd, buf, len);
}

static int nhi_raw_recv(ds4_transport *t, void *buf, size_t len) {
    return ds4_transport_tcp_read(t->control_fd, buf, len);
}

/* Backend liveness is intentionally separate from max_oob_bytes(). Geometry
 * remains stable so a failed NHI link can never masquerade as an oversized
 * payload and silently downgrade to TCP. Queued inbound completions may still
 * drain after peer CLOSE, but no new outbound descriptor is prepared. */
static int nhi_check_prepare(ds4_transport *t) {
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&ctx->mu);
    int saved = ctx->failed;
    if (saved == 0 && (ctx->peer_closed || ctx->stopping)) saved = EPIPE;
    if (saved == 0 && ctx->active_tx_lease) saved = EBUSY;
    pthread_mutex_unlock(&ctx->mu);
    if (saved != 0) {
        errno = saved;
        return -1;
    }
    return 0;
}

static int nhi_configure(ds4_transport *t,
                         uint32_t peer_frame_size,
                         uint32_t peer_ring_size,
                         char *err,
                         size_t errlen) {
    ds4_nhi_ctx *ctx = t->ctx;
    if (!ctx || peer_frame_size != ctx->frame_size || peer_ring_size < 2u) {
        nhi_set_err(err, errlen, "NHI peer pool geometry is incompatible");
        errno = EPROTO;
        return -1;
    }
    uint32_t shared_ring = peer_ring_size < ctx->ring_size
        ? peer_ring_size : ctx->ring_size;
    pthread_mutex_lock(&ctx->mu);
    if (ctx->failed || ctx->peer_closed || ctx->stopping) {
        int saved = ctx->failed ? ctx->failed : EPIPE;
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "NHI candidate failed before activation");
        errno = saved;
        return -1;
    }
    if (ctx->tx_inflight != 0 || ctx->tx_count != 0 || ctx->rx_count != 0) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "NHI link has traffic before negotiation");
        errno = EBUSY;
        return -1;
    }
    ctx->tx_frame_limit = shared_ring - 1u;
    ctx->activated = 1;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static size_t nhi_max_oob_bytes(const ds4_transport *t) {
    const ds4_nhi_ctx *ctx_const = t ? t->ctx : NULL;
    if (!ctx_const) return 0;
    ds4_nhi_ctx *ctx = (ds4_nhi_ctx *)ctx_const;
    pthread_mutex_lock(&ctx->mu);
    /* This is negotiated geometry, not a liveness probe. Once NHI has been
     * selected for a link, backend failure must make the next OOB operation
     * fail; it must never silently turn a prepared message into TCP fallback
     * after the connection generation may already have NHI traffic. */
    const uint64_t capacity =
        (uint64_t)ctx->tx_frame_limit * ctx->frame_size;
    pthread_mutex_unlock(&ctx->mu);
    if (capacity <= DS4_NHI_ENVELOPE_BYTES) return 0;
    uint64_t payload = capacity - DS4_NHI_ENVELOPE_BYTES;
    if (payload > (uint64_t)UINT32_MAX - DS4_NHI_ENVELOPE_BYTES)
        payload = (uint64_t)UINT32_MAX - DS4_NHI_ENVELOPE_BYTES;
    return payload > SIZE_MAX ? SIZE_MAX : (size_t)payload;
}

static void nhi_deadline(struct timespec *deadline, uint32_t timeout_sec) {
    clock_gettime(CLOCK_REALTIME, deadline);
    deadline->tv_sec += (time_t)timeout_sec;
}

static int nhi_wait_for_tx_locked(ds4_nhi_ctx *ctx, uint32_t nframes) {
    struct timespec deadline;
    nhi_deadline(&deadline, ctx->timeout_sec);
    while (!ctx->stopping && !ctx->failed && !ctx->peer_closed &&
           nframes > ctx->tx_frame_limit - ctx->tx_inflight) {
        int rc = pthread_cond_timedwait(&ctx->tx_cv, &ctx->mu, &deadline);
        if (rc == ETIMEDOUT) {
            nhi_trace(ctx,
                      "TX wait timeout need=%u inflight=%u limit=%u count=%u",
                      nframes, ctx->tx_inflight, ctx->tx_frame_limit,
                      ctx->tx_count);
            errno = ETIMEDOUT;
            return -1;
        }
        if (rc != 0) {
            errno = rc;
            return -1;
        }
    }
    if (ctx->failed) {
        errno = ctx->failed;
        return -1;
    }
    if (ctx->stopping || ctx->peer_closed) {
        errno = EPIPE;
        return -1;
    }
    return 0;
}

static int nhi_send_desc(ds4_transport *t,
                         const ds4_transport_bulk_desc *desc,
                         const void *buf,
                         size_t len) {
    ds4_nhi_ctx *ctx = t->ctx;
    const uint64_t total = (uint64_t)DS4_NHI_ENVELOPE_BYTES + len;
    const uint32_t nframes = (uint32_t)((total + ctx->frame_size - 1u) /
                                        ctx->frame_size);
    const uint32_t last_len = (uint32_t)(total -
        (uint64_t)(nframes - 1u) * ctx->frame_size);
    if (total > UINT32_MAX || nframes == 0 ||
        nframes != desc->frame_count ||
        nframes > ctx->tx_frame_limit) {
        errno = EMSGSIZE;
        return -1;
    }

    unsigned char envelope[DS4_NHI_ENVELOPE_BYTES];
    nhi_envelope_encode(desc, envelope);

    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_tx_lease) {
        pthread_mutex_unlock(&ctx->mu);
        errno = EBUSY;
        return -1;
    }
    if (nhi_wait_for_tx_locked(ctx, nframes) != 0) {
        const int saved_errno = errno ? errno : EIO;
        int saved = nhi_fail_locked(ctx, saved_errno);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = saved_errno;
        return -1;
    }
    if (ctx->tx_count >= ctx->ring_size) {
        int saved = nhi_fail_locked(ctx, EOVERFLOW);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = saved;
        return -1;
    }

    const uint32_t first = ctx->tx_next;
    nhi_ring_copy_in(ctx, first, 0, envelope, sizeof(envelope));
    nhi_ring_copy_in(ctx, first, sizeof(envelope), buf, len);

    struct tbstream_zc_tx tx;
    memset(&tx, 0, sizeof(tx));
    tx.nframes = nframes;
    tx.last_len = last_len;
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_SUBMIT_TX, &tx) != 0) {
        int saved_errno = errno ? errno : EIO;
        int saved = nhi_fail_locked(ctx, saved_errno);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = saved_errno;
        return -1;
    }
    if (tx.first != first) {
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = EPROTO;
        return -1;
    }

    ds4_nhi_pending_tx *pending = &ctx->tx_pending[ctx->tx_tail];
    pending->first = first;
    pending->nframes = nframes;
    pending->sequence = desc->sequence;
    ctx->tx_tail = (ctx->tx_tail + 1u) % ctx->ring_size;
    ctx->tx_count++;
    ctx->tx_inflight += nframes;
    ctx->tx_next = (ctx->tx_next + nframes) % ctx->ring_size;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_wait_for_rx_locked(ds4_nhi_ctx *ctx) {
    struct timespec deadline;
    nhi_deadline(&deadline, ctx->timeout_sec);
    while (ctx->rx_count == 0 && !ctx->failed && !ctx->peer_closed &&
           !ctx->stopping) {
        int rc = pthread_cond_timedwait(&ctx->rx_cv, &ctx->mu, &deadline);
        if (rc == ETIMEDOUT) {
            nhi_trace(ctx,
                      "RX wait timeout next=%u event_next=%u count=%u",
                      ctx->rx_next, ctx->rx_event_next, ctx->rx_count);
            errno = ETIMEDOUT;
            return -1;
        }
        if (rc != 0) {
            errno = rc;
            return -1;
        }
    }
    if (ctx->rx_count != 0) return 1;
    if (ctx->failed) {
        errno = ctx->failed;
        return -1;
    }
    if (ctx->peer_closed) return 0;
    errno = ECANCELED;
    return -1;
}

static int nhi_recv_desc(ds4_transport *t,
                         const ds4_transport_bulk_desc *desc,
                         void *buf,
                         size_t len) {
    ds4_nhi_ctx *ctx = t->ctx;
    const uint64_t total = (uint64_t)DS4_NHI_ENVELOPE_BYTES + len;
    const uint32_t expected_frames = (uint32_t)(
        (total + ctx->frame_size - 1u) / ctx->frame_size);
    if (expected_frames == 0 || expected_frames != desc->frame_count ||
        expected_frames > ctx->tx_frame_limit || total > UINT32_MAX) {
        errno = EMSGSIZE;
        return -1;
    }

    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_rx_lease) {
        pthread_mutex_unlock(&ctx->mu);
        errno = EBUSY;
        return -1;
    }
    int wait_rc = nhi_wait_for_rx_locked(ctx);
    if (wait_rc <= 0) {
        if (wait_rc < 0) {
            const int saved_errno = errno ? errno : EIO;
            int saved = nhi_fail_locked(ctx, saved_errno);
            pthread_mutex_unlock(&ctx->mu);
            ds4_transport_internal_fail(t, saved);
            errno = saved_errno;
            return -1;
        }
        pthread_mutex_unlock(&ctx->mu);
        return wait_rc;
    }

    const struct tbstream_zc_event ev = ctx->rx_events[ctx->rx_head];
    if (ev.first != ctx->rx_next || ev.nframes != expected_frames ||
        ev.bytes != (uint32_t)total) {
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = EPROTO;
        return -1;
    }

    unsigned char envelope[DS4_NHI_ENVELOPE_BYTES];
    nhi_ring_copy_out(ctx, ev.first, 0, envelope, sizeof(envelope));
    if (!nhi_envelope_matches(envelope, desc)) {
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = EPROTO;
        return -1;
    }
    nhi_ring_copy_out(ctx, ev.first, sizeof(envelope), buf, len);

    uint32_t repost = ev.nframes;
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_POST_RX, &repost) != 0) {
        int saved_errno = errno ? errno : EIO;
        int saved = nhi_fail_locked(ctx, saved_errno);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = saved_errno;
        return -1;
    }
    ctx->rx_head = (ctx->rx_head + 1u) % ctx->ring_size;
    ctx->rx_count--;
    ctx->rx_next = (ctx->rx_next + ev.nframes) % ctx->ring_size;
    pthread_mutex_unlock(&ctx->mu);
    return 1;
}

static int nhi_mapped_leases_supported(const ds4_transport *t) {
    const ds4_nhi_ctx *ctx_const = t ? t->ctx : NULL;
    if (!ctx_const) return 0;
    ds4_nhi_ctx *ctx = (ds4_nhi_ctx *)ctx_const;
    pthread_mutex_lock(&ctx->mu);
    const int supported = nhi_mapped_model_io_enabled() &&
        ctx->gpu_mapping_registered &&
        ctx->gpu_mapping != NULL;
    pthread_mutex_unlock(&ctx->mu);
    return supported;
}

static int nhi_tx_lease_acquire(ds4_transport *t,
                                const ds4_transport_bulk_desc *desc,
                                ds4_transport_lease *lease,
                                char *err,
                                size_t errlen) {
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx || !desc || !lease) {
        nhi_set_err(err, errlen, "invalid NHI TX lease request");
        errno = EINVAL;
        return -1;
    }
    const uint64_t total =
        (uint64_t)DS4_NHI_ENVELOPE_BYTES + desc->payload_bytes;
    const uint32_t nframes = (uint32_t)(
        (total + ctx->frame_size - 1u) / ctx->frame_size);
    if (total > UINT32_MAX || nframes == 0 ||
        nframes != desc->frame_count || nframes > ctx->tx_frame_limit) {
        nhi_set_err(err, errlen, "invalid NHI TX lease geometry");
        errno = EMSGSIZE;
        return -1;
    }

    pthread_mutex_lock(&ctx->mu);
    if (!ctx->gpu_mapping_registered || !ctx->gpu_mapping) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "NHI GPU mapping is unavailable");
        errno = ENOTSUP;
        return -1;
    }
    if (ctx->active_tx_lease) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "an NHI TX lease is already active");
        errno = EBUSY;
        return -1;
    }
    if (ctx->failed || ctx->peer_closed || ctx->stopping) {
        int saved = ctx->failed ? ctx->failed : EPIPE;
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "NHI TX endpoint is unavailable");
        errno = saved;
        return -1;
    }

    const uint32_t first = ctx->tx_next;
    const size_t payload_offset =
        (size_t)first * ctx->frame_size + DS4_NHI_ENVELOPE_BYTES;
    if (payload_offset > ctx->pool_bytes ||
        desc->payload_bytes > ctx->pool_bytes - payload_offset) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen,
                    "NHI TX payload wraps and requires the copy path");
        errno = ENOTSUP;
        return -1;
    }
    if (nhi_wait_for_tx_locked(ctx, nframes) != 0) {
        const int saved_errno = errno ? errno : EIO;
        int saved = nhi_fail_locked(ctx, saved_errno);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = saved_errno;
        return -1;
    }

    unsigned char envelope[DS4_NHI_ENVELOPE_BYTES];
    nhi_envelope_encode(desc, envelope);
    nhi_ring_copy_in(ctx, first, 0, envelope, sizeof(envelope));

    const size_t mapping_offset =
        (size_t)(ctx->tx_pool - (unsigned char *)ctx->mapping) +
        payload_offset;
    lease->host_ptr = ctx->tx_pool + payload_offset;
    lease->device_ptr = (unsigned char *)ctx->gpu_mapping + mapping_offset;
    lease->bytes = desc->payload_bytes;
    lease->first = first;
    lease->nframes = nframes;
    lease->event_bytes = (uint32_t)total;
    ctx->active_tx_lease = lease;
    nhi_trace(ctx,
              "TX acquire seq=%llu request=%llu first=%u nframes=%u bytes=%u "
              "inflight=%u count=%u",
              (unsigned long long)desc->sequence,
              (unsigned long long)desc->request_id,
              first, nframes, (uint32_t)total,
              ctx->tx_inflight, ctx->tx_count);
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_rx_lease_acquire(ds4_transport *t,
                                const ds4_transport_bulk_desc *desc,
                                ds4_transport_lease *lease,
                                char *err,
                                size_t errlen) {
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx || !desc || !lease) {
        nhi_set_err(err, errlen, "invalid NHI RX lease request");
        errno = EINVAL;
        return -1;
    }
    const uint64_t total =
        (uint64_t)DS4_NHI_ENVELOPE_BYTES + desc->payload_bytes;
    const uint32_t nframes = (uint32_t)(
        (total + ctx->frame_size - 1u) / ctx->frame_size);
    if (total > UINT32_MAX || nframes == 0 ||
        nframes != desc->frame_count || nframes > ctx->tx_frame_limit) {
        nhi_set_err(err, errlen, "invalid NHI RX lease geometry");
        errno = EMSGSIZE;
        return -1;
    }

    pthread_mutex_lock(&ctx->mu);
    if (!ctx->gpu_mapping_registered || !ctx->gpu_mapping) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "NHI GPU mapping is unavailable");
        errno = ENOTSUP;
        return -1;
    }
    if (ctx->active_rx_lease) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen, "an NHI RX lease is already active");
        errno = EBUSY;
        return -1;
    }

    nhi_trace(ctx,
              "RX acquire wait seq=%llu request=%llu nframes=%u bytes=%u "
              "next=%u event_next=%u count=%u",
              (unsigned long long)desc->sequence,
              (unsigned long long)desc->request_id,
              nframes, (uint32_t)total, ctx->rx_next,
              ctx->rx_event_next, ctx->rx_count);
    int wait_rc = nhi_wait_for_rx_locked(ctx);
    if (wait_rc <= 0) {
        const int saved_errno = wait_rc == 0
            ? ECONNRESET : (errno ? errno : EIO);
        if (wait_rc < 0) {
            int saved = nhi_fail_locked(ctx, saved_errno);
            pthread_mutex_unlock(&ctx->mu);
            ds4_transport_internal_fail(t, saved);
        } else {
            pthread_mutex_unlock(&ctx->mu);
        }
        errno = saved_errno;
        return -1;
    }

    const struct tbstream_zc_event ev = ctx->rx_events[ctx->rx_head];
    if (ev.first != ctx->rx_next || ev.nframes != nframes ||
        ev.bytes != (uint32_t)total) {
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        nhi_set_err(err, errlen, "NHI RX lease event mismatch");
        errno = EPROTO;
        return -1;
    }

    unsigned char envelope[DS4_NHI_ENVELOPE_BYTES];
    nhi_ring_copy_out(ctx, ev.first, 0, envelope, sizeof(envelope));
    if (!nhi_envelope_matches(envelope, desc)) {
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        nhi_set_err(err, errlen, "NHI RX lease envelope mismatch");
        errno = EPROTO;
        return -1;
    }

    const size_t payload_offset =
        (size_t)ev.first * ctx->frame_size + DS4_NHI_ENVELOPE_BYTES;
    if (payload_offset > ctx->pool_bytes ||
        desc->payload_bytes > ctx->pool_bytes - payload_offset) {
        pthread_mutex_unlock(&ctx->mu);
        nhi_set_err(err, errlen,
                    "NHI RX payload wraps and requires the copy path");
        errno = ENOTSUP;
        return -1;
    }

    const size_t mapping_offset =
        (size_t)(ctx->rx_pool - (unsigned char *)ctx->mapping) +
        payload_offset;
    lease->host_ptr = ctx->rx_pool + payload_offset;
    lease->device_ptr = (unsigned char *)ctx->gpu_mapping + mapping_offset;
    lease->bytes = desc->payload_bytes;
    lease->first = ev.first;
    lease->nframes = ev.nframes;
    lease->event_bytes = ev.bytes;
    ctx->active_rx_lease = lease;
    nhi_trace(ctx,
              "RX acquire ready seq=%llu request=%llu first=%u nframes=%u "
              "bytes=%u count=%u",
              (unsigned long long)desc->sequence,
              (unsigned long long)desc->request_id,
              ev.first, ev.nframes, ev.bytes, ctx->rx_count);
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_tx_lease_mark_control_sent(ds4_transport_lease *lease) {
    if (!lease || !lease->transport ||
        lease->direction != DS4_TRANSPORT_LEASE_TX) {
        errno = EINVAL;
        return -1;
    }
    ds4_nhi_ctx *ctx = lease->transport->ctx;
    if (!ctx) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&ctx->mu);
    /* The caller has already written TCP before entering this notification.
     * Record that fact even when a concurrent backend failure is discovered. */
    lease->control_sent = 1;
    if (ctx->active_tx_lease != lease) {
        pthread_mutex_unlock(&ctx->mu);
        errno = EINVAL;
        return -1;
    }
    if (ctx->failed || ctx->peer_closed || ctx->stopping) {
        int saved = ctx->failed ? ctx->failed : EPIPE;
        pthread_mutex_unlock(&ctx->mu);
        errno = saved;
        return -1;
    }
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_mapped_sync(const char *label) {
    if (ds4_gpu_host_mapped_synchronize(label)) return 0;
    errno = EIO;
    return -1;
}

static int nhi_finish_lease_failure(ds4_nhi_ctx *ctx,
                                    ds4_transport_lease *lease,
                                    int error_code) {
    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_tx_lease == lease) ctx->active_tx_lease = NULL;
    if (ctx->active_rx_lease == lease) ctx->active_rx_lease = NULL;
    int saved = nhi_fail_locked(ctx, error_code);
    pthread_mutex_unlock(&ctx->mu);
    ds4_transport_internal_fail(lease->transport, saved);
    errno = error_code > 0 ? error_code : saved;
    return -1;
}

static int nhi_tx_lease_commit(ds4_transport_lease *lease,
                               int gpu_quiesced) {
    ds4_transport *t = lease ? lease->transport : NULL;
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx || !lease->control_sent) {
        errno = EPERM;
        return -1;
    }
    if (gpu_quiesced) {
        nhi_trace(ctx,
                  "TX commit sync skipped (caller GPU-quiesced) "
                  "seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
    } else {
        nhi_trace(ctx,
                  "TX commit sync begin seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
        if (nhi_mapped_sync("NHI TX mapped ownership") != 0)
            return nhi_finish_lease_failure(ctx, lease, EIO);
        nhi_trace(ctx,
                  "TX commit sync done seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
    }

    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_tx_lease != lease || !ctx->gpu_mapping_registered ||
        ctx->failed || ctx->peer_closed || ctx->stopping ||
        lease->first != ctx->tx_next || lease->nframes == 0 ||
        ctx->tx_inflight > ctx->tx_frame_limit ||
        lease->nframes > ctx->tx_frame_limit - ctx->tx_inflight ||
        ctx->tx_count >= ctx->ring_size) {
        int error_code = ctx->failed ? ctx->failed
            : (ctx->peer_closed ? EPIPE
               : (ctx->stopping ? ECANCELED : EPROTO));
        ctx->active_tx_lease = NULL;
        int saved = nhi_fail_locked(ctx, error_code);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = error_code;
        return -1;
    }

    struct tbstream_zc_tx tx;
    memset(&tx, 0, sizeof(tx));
    tx.nframes = lease->nframes;
    tx.last_len = lease->event_bytes -
        (lease->nframes - 1u) * ctx->frame_size;
    nhi_trace(ctx,
              "TX submit begin seq=%llu request=%llu first=%u nframes=%u",
              (unsigned long long)lease->desc.sequence,
              (unsigned long long)lease->desc.request_id,
              lease->first, lease->nframes);
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_SUBMIT_TX, &tx) != 0) {
        int error_code = errno ? errno : EIO;
        ctx->active_tx_lease = NULL;
        int saved = nhi_fail_locked(ctx, error_code);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = error_code;
        return -1;
    }
    nhi_trace(ctx,
              "TX submit done seq=%llu request=%llu expected_first=%u got_first=%u "
              "nframes=%u last_len=%u",
              (unsigned long long)lease->desc.sequence,
              (unsigned long long)lease->desc.request_id,
              lease->first, tx.first, tx.nframes, tx.last_len);
    if (tx.first != lease->first) {
        ctx->active_tx_lease = NULL;
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = EPROTO;
        return -1;
    }

    ds4_nhi_pending_tx *pending = &ctx->tx_pending[ctx->tx_tail];
    pending->first = lease->first;
    pending->nframes = lease->nframes;
    pending->sequence = lease->desc.sequence;
    ctx->tx_tail = (ctx->tx_tail + 1u) % ctx->ring_size;
    ctx->tx_count++;
    ctx->tx_inflight += lease->nframes;
    ctx->tx_next = (ctx->tx_next + lease->nframes) % ctx->ring_size;
    ctx->active_tx_lease = NULL;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_rx_lease_commit(ds4_transport_lease *lease,
                               int gpu_quiesced) {
    ds4_transport *t = lease ? lease->transport : NULL;
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx) {
        errno = EINVAL;
        return -1;
    }
    if (gpu_quiesced) {
        nhi_trace(ctx,
                  "RX commit sync skipped (caller GPU-quiesced) "
                  "seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
    } else {
        nhi_trace(ctx,
                  "RX commit sync begin seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
        if (nhi_mapped_sync("NHI RX mapped ownership") != 0)
            return nhi_finish_lease_failure(ctx, lease, EIO);
        nhi_trace(ctx,
                  "RX commit sync done seq=%llu request=%llu first=%u nframes=%u",
                  (unsigned long long)lease->desc.sequence,
                  (unsigned long long)lease->desc.request_id,
                  lease->first, lease->nframes);
    }

    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_rx_lease != lease || !ctx->gpu_mapping_registered ||
        ctx->failed || ctx->stopping || ctx->rx_count == 0) {
        int error_code = ctx->failed ? ctx->failed : EPROTO;
        ctx->active_rx_lease = NULL;
        int saved = nhi_fail_locked(ctx, error_code);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = error_code;
        return -1;
    }
    const struct tbstream_zc_event *ev = &ctx->rx_events[ctx->rx_head];
    if (ev->first != lease->first || ev->first != ctx->rx_next ||
        ev->nframes != lease->nframes || ev->bytes != lease->event_bytes) {
        ctx->active_rx_lease = NULL;
        int saved = nhi_fail_locked(ctx, EPROTO);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = EPROTO;
        return -1;
    }

    uint32_t repost = ev->nframes;
    nhi_trace(ctx,
              "RX repost begin seq=%llu request=%llu first=%u nframes=%u",
              (unsigned long long)lease->desc.sequence,
              (unsigned long long)lease->desc.request_id,
              ev->first, ev->nframes);
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_POST_RX, &repost) != 0) {
        int error_code = errno ? errno : EIO;
        ctx->active_rx_lease = NULL;
        int saved = nhi_fail_locked(ctx, error_code);
        pthread_mutex_unlock(&ctx->mu);
        ds4_transport_internal_fail(t, saved);
        errno = error_code;
        return -1;
    }
    nhi_trace(ctx,
              "RX repost done seq=%llu request=%llu first=%u nframes=%u count=%u",
              (unsigned long long)lease->desc.sequence,
              (unsigned long long)lease->desc.request_id,
              ev->first, ev->nframes, ctx->rx_count);
    ctx->rx_head = (ctx->rx_head + 1u) % ctx->ring_size;
    ctx->rx_count--;
    ctx->rx_next = (ctx->rx_next + ev->nframes) % ctx->ring_size;
    ctx->active_rx_lease = NULL;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_lease_commit(ds4_transport_lease *lease, int gpu_quiesced) {
    if (!lease) {
        errno = EINVAL;
        return -1;
    }
    if (lease->direction == DS4_TRANSPORT_LEASE_TX)
        return nhi_tx_lease_commit(lease, gpu_quiesced);
    if (lease->direction == DS4_TRANSPORT_LEASE_RX)
        return nhi_rx_lease_commit(lease, gpu_quiesced);
    errno = EINVAL;
    return -1;
}

static int nhi_lease_abort(ds4_transport_lease *lease) {
    ds4_transport *t = lease ? lease->transport : NULL;
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx) {
        errno = EINVAL;
        return -1;
    }

    const int recoverable = lease->direction == DS4_TRANSPORT_LEASE_TX &&
        !lease->control_sent;
    /* Even a locally recoverable abort may follow queued GPU writes. Fence
     * them before making tx_next reusable by the copy path or a new lease. */
    if (recoverable &&
        nhi_mapped_sync("NHI TX mapped abort ownership") != 0)
        return nhi_finish_lease_failure(ctx, lease, EIO);

    pthread_mutex_lock(&ctx->mu);
    ds4_transport_lease **active =
        lease->direction == DS4_TRANSPORT_LEASE_TX
        ? &ctx->active_tx_lease : &ctx->active_rx_lease;
    if (*active != lease) {
        pthread_mutex_unlock(&ctx->mu);
        errno = EINVAL;
        return -1;
    }
    if (recoverable) {
        pthread_mutex_unlock(&ctx->mu);
        return 0;
    }

    *active = NULL;
    int saved = nhi_fail_locked(ctx, ECANCELED);
    pthread_mutex_unlock(&ctx->mu);
    ds4_transport_internal_fail(t, saved);
    errno = ECANCELED;
    return -1;
}

static int nhi_lease_abort_finish(ds4_transport_lease *lease) {
    ds4_transport *t = lease ? lease->transport : NULL;
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx || lease->direction != DS4_TRANSPORT_LEASE_TX ||
        lease->control_sent) {
        errno = EINVAL;
        return -1;
    }
    pthread_mutex_lock(&ctx->mu);
    if (ctx->active_tx_lease != lease) {
        pthread_mutex_unlock(&ctx->mu);
        errno = EINVAL;
        return -1;
    }
    ctx->active_tx_lease = NULL;
    pthread_mutex_unlock(&ctx->mu);
    return 0;
}

static int nhi_unregister_gpu_mapping(ds4_nhi_ctx *ctx) {
    if (!ctx || !ctx->gpu_mapping_registered) return 1;
    if (!ds4_gpu_host_unregister_mapped(ctx->mapping)) {
        fprintf(stderr,
                "ds4: NHI mapped-host unregister failed; preserving mmap\n");
        return 0;
    }
    ctx->gpu_mapping_registered = 0;
    ctx->gpu_mapping = NULL;
    return 1;
}

static void nhi_close(ds4_transport *t) {
    ds4_nhi_ctx *ctx = t ? t->ctx : NULL;
    if (!ctx) return;

    pthread_mutex_lock(&ctx->mu);
    ctx->stopping = 1;
    nhi_broadcast_locked(ctx);
    pthread_mutex_unlock(&ctx->mu);
    if (ctx->wake_fd >= 0) {
        uint64_t one = 1;
        (void)write(ctx->wake_fd, &one, sizeof(one));
    }
    if (ctx->dispatcher_started)
        (void)pthread_join(ctx->dispatcher, NULL);
    const int mapping_unregistered = nhi_unregister_gpu_mapping(ctx);
    if (ctx->mapping != MAP_FAILED && mapping_unregistered)
        (void)munmap(ctx->mapping, ctx->mapping_bytes);
    if (ctx->device_fd >= 0) close(ctx->device_fd);
    if (ctx->wake_fd >= 0) close(ctx->wake_fd);
    pthread_cond_destroy(&ctx->rx_cv);
    pthread_cond_destroy(&ctx->tx_cv);
    pthread_mutex_destroy(&ctx->mu);
    free(ctx->rx_events);
    free(ctx->tx_pending);
    free(ctx);
    t->ctx = NULL;
}

static const ds4_transport_ops nhi_ops = {
    .name = "nhi-cpu-copy",
    .caps = DS4_TRANSPORT_CAP_STREAM | DS4_TRANSPORT_CAP_ZEROCOPY,
    .send_bulk = nhi_raw_send,
    .recv_bulk = nhi_raw_recv,
    .send_bulk_desc = nhi_send_desc,
    .recv_bulk_desc = nhi_recv_desc,
    .mapped_leases_supported = nhi_mapped_leases_supported,
    .tx_lease_acquire = nhi_tx_lease_acquire,
    .rx_lease_acquire = nhi_rx_lease_acquire,
    .tx_lease_mark_control_sent = nhi_tx_lease_mark_control_sent,
    .lease_commit = nhi_lease_commit,
    .lease_abort = nhi_lease_abort,
    .lease_abort_finish = nhi_lease_abort_finish,
    .check_prepare = nhi_check_prepare,
    .configure = nhi_configure,
    .max_oob_bytes = nhi_max_oob_bytes,
    .close = nhi_close,
};

static void nhi_unstarted_cleanup(ds4_nhi_ctx *ctx,
                                  int mutex_ready,
                                  int tx_cv_ready,
                                  int rx_cv_ready) {
    if (!ctx) return;
    const int mapping_unregistered = nhi_unregister_gpu_mapping(ctx);
    if (ctx->mapping != MAP_FAILED && mapping_unregistered)
        (void)munmap(ctx->mapping, ctx->mapping_bytes);
    if (ctx->device_fd >= 0) close(ctx->device_fd);
    if (ctx->wake_fd >= 0) close(ctx->wake_fd);
    if (rx_cv_ready) pthread_cond_destroy(&ctx->rx_cv);
    if (tx_cv_ready) pthread_cond_destroy(&ctx->tx_cv);
    if (mutex_ready) pthread_mutex_destroy(&ctx->mu);
    free(ctx->rx_events);
    free(ctx->tx_pending);
    free(ctx);
}

static ds4_transport *nhi_create_common(
        int control_fd,
        int owns_control_fd,
        const char *device_path,
        char *err,
        size_t errlen) {
    if (control_fd < 0 || !device_path || !device_path[0]) {
        nhi_set_err(err, errlen, "invalid NHI transport arguments");
        errno = EINVAL;
        if (owns_control_fd && control_fd >= 0) close(control_fd);
        return NULL;
    }

    ds4_nhi_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        nhi_set_err(err, errlen, "out of memory creating NHI transport");
        if (owns_control_fd) close(control_fd);
        return NULL;
    }
    ctx->device_fd = -1;
    ctx->wake_fd = -1;
    ctx->mapping = MAP_FAILED;
    ctx->timeout_sec = nhi_timeout_sec();
    int mutex_ready = 0, tx_cv_ready = 0, rx_cv_ready = 0;

    ctx->device_fd = open(device_path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (ctx->device_fd < 0) {
        if (err && errlen)
            snprintf(err, errlen, "open %s: %s", device_path, strerror(errno));
        goto fail;
    }
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_ENABLE) != 0) {
        if (err && errlen)
            snprintf(err, errlen, "%s: TBSTREAM_ZC_ENABLE: %s",
                     device_path, strerror(errno));
        goto fail;
    }
    struct tbstream_zc_info info;
    memset(&info, 0, sizeof(info));
    if (ioctl(ctx->device_fd, TBSTREAM_ZC_GET_INFO, &info) != 0) {
        if (err && errlen)
            snprintf(err, errlen, "%s: TBSTREAM_ZC_GET_INFO: %s",
                     device_path, strerror(errno));
        goto fail;
    }
    if (info.frame_size != TBSTREAM_ZC_FRAME_SIZE || info.ring_size < 2u ||
        (size_t)info.ring_size > SIZE_MAX / info.frame_size) {
        nhi_set_err(err, errlen, "invalid NHI zero-copy pool geometry");
        errno = EPROTO;
        goto fail;
    }
    ctx->frame_size = info.frame_size;
    ctx->ring_size = info.ring_size;
    ctx->pool_bytes = (size_t)info.ring_size * info.frame_size;
    if (ctx->pool_bytes > SIZE_MAX / 2u || info.tx_pool_offset != 0 ||
        info.rx_pool_offset != ctx->pool_bytes) {
        nhi_set_err(err, errlen, "invalid NHI zero-copy pool offsets");
        errno = EPROTO;
        goto fail;
    }
    ctx->mapping_bytes = ctx->pool_bytes * 2u;
    ctx->mapping = mmap(NULL, ctx->mapping_bytes,
                        PROT_READ | PROT_WRITE, MAP_SHARED,
                        ctx->device_fd, 0);
    if (ctx->mapping == MAP_FAILED) {
        if (err && errlen)
            snprintf(err, errlen, "%s: mmap zero-copy pools: %s",
                     device_path, strerror(errno));
        goto fail;
    }
#ifdef MADV_DONTFORK
    (void)madvise(ctx->mapping, ctx->mapping_bytes, MADV_DONTFORK);
#endif
    ctx->tx_pool = (unsigned char *)ctx->mapping + info.tx_pool_offset;
    ctx->rx_pool = (unsigned char *)ctx->mapping + info.rx_pool_offset;
    const int mapped_requested = nhi_mapped_model_io_enabled();
#ifdef DS4_ROCM_BUILD
    if (mapped_requested && ds4_gpu_host_mapping_supported()) {
        void *gpu_mapping = NULL;
        if (ds4_gpu_host_register_mapped(
                ctx->mapping, (uint64_t)ctx->mapping_bytes, &gpu_mapping)) {
            ctx->gpu_mapping = gpu_mapping;
            ctx->gpu_mapping_registered = 1;
        }
    }
#endif
    ctx->tx_frame_limit = info.ring_size - 1u;
    ctx->tx_pending = calloc(info.ring_size, sizeof(*ctx->tx_pending));
    ctx->rx_events = calloc(info.ring_size, sizeof(*ctx->rx_events));
    if (!ctx->tx_pending || !ctx->rx_events) {
        nhi_set_err(err, errlen, "out of memory creating NHI event queues");
        goto fail;
    }
    int rc = pthread_mutex_init(&ctx->mu, NULL);
    if (rc != 0) {
        errno = rc;
        nhi_set_err(err, errlen, "failed to initialize NHI state mutex");
        goto fail;
    }
    mutex_ready = 1;
    rc = pthread_cond_init(&ctx->tx_cv, NULL);
    if (rc != 0) {
        errno = rc;
        nhi_set_err(err, errlen, "failed to initialize NHI TX condition");
        goto fail;
    }
    tx_cv_ready = 1;
    rc = pthread_cond_init(&ctx->rx_cv, NULL);
    if (rc != 0) {
        errno = rc;
        nhi_set_err(err, errlen, "failed to initialize NHI RX condition");
        goto fail;
    }
    rx_cv_ready = 1;
    ctx->wake_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (ctx->wake_fd < 0) {
        nhi_set_err(err, errlen, "failed to create NHI dispatcher eventfd");
        goto fail;
    }

    ds4_transport *t = ds4_transport_internal_create(
        &nhi_ops, control_fd, owns_control_fd, ctx,
        info.frame_size, info.ring_size, err, errlen);
    if (!t) goto fail;
    ctx->owner = t;
    nhi_trace(ctx,
              "created device=%s frame_size=%u ring_size=%u "
              "mapped_requested=%d mapped=%d",
              device_path, ctx->frame_size, ctx->ring_size,
              mapped_requested, ctx->gpu_mapping_registered);
    rc = pthread_create(&ctx->dispatcher, NULL, nhi_dispatch_main, ctx);
    if (rc != 0) {
        errno = rc;
        nhi_set_err(err, errlen, "failed to start NHI completion dispatcher");
        ds4_transport_release(t);
        return NULL;
    }
    ctx->dispatcher_started = 1;
    return t;

fail:
    {
        int saved = errno ? errno : EIO;
        nhi_unstarted_cleanup(ctx, mutex_ready, tx_cv_ready, rx_cv_ready);
        if (owns_control_fd) close(control_fd);
        errno = saved;
        return NULL;
    }
}

ds4_transport *ds4_transport_nhi_create(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen) {
    return nhi_create_common(control_fd, 0, device_path, err, errlen);
}

ds4_transport *ds4_transport_nhi_dup(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen) {
    if (control_fd < 0) {
        nhi_set_err(err, errlen, "invalid NHI control fd");
        errno = EINVAL;
        return NULL;
    }
    int copy = fcntl(control_fd, F_DUPFD_CLOEXEC, 0);
    if (copy < 0) {
        nhi_set_err(err, errlen, "failed to duplicate NHI control fd");
        return NULL;
    }
    return nhi_create_common(copy, 1, device_path, err, errlen);
}

#else

ds4_transport *ds4_transport_nhi_create(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen) {
    (void)control_fd;
    (void)device_path;
    if (err && errlen)
        snprintf(err, errlen, "nhi transport unavailable on this platform");
    errno = ENOTSUP;
    return NULL;
}

ds4_transport *ds4_transport_nhi_dup(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen) {
    return ds4_transport_nhi_create(control_fd, device_path, err, errlen);
}

#endif
