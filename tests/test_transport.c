#include "ds4_transport_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int checks;
static int failures;

#define CHECK(expr) do { \
    checks++; \
    if (!(expr)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

typedef struct {
    unsigned char host[256];
    unsigned char device[256];
    ds4_transport_lease *tx;
    ds4_transport_lease *rx;
    unsigned int conservative_commits;
    unsigned int gpu_quiesced_commits;
} fake_lease_ctx;

static int fake_mapped_supported(const ds4_transport *t) {
    return t && t->ctx != NULL;
}

static int fake_tx_acquire(ds4_transport *t,
                           const ds4_transport_bulk_desc *desc,
                           ds4_transport_lease *lease,
                           char *err,
                           size_t errlen) {
    (void)err;
    (void)errlen;
    fake_lease_ctx *ctx = t->ctx;
    if (ctx->tx) {
        errno = EBUSY;
        return -1;
    }
    ctx->tx = lease;
    lease->host_ptr = ctx->host;
    lease->device_ptr = ctx->device;
    lease->bytes = desc->payload_bytes;
    return 0;
}

static int fake_rx_acquire(ds4_transport *t,
                           const ds4_transport_bulk_desc *desc,
                           ds4_transport_lease *lease,
                           char *err,
                           size_t errlen) {
    (void)err;
    (void)errlen;
    fake_lease_ctx *ctx = t->ctx;
    if (ctx->rx) {
        errno = EBUSY;
        return -1;
    }
    ctx->rx = lease;
    lease->host_ptr = ctx->host;
    lease->device_ptr = ctx->device;
    lease->bytes = desc->payload_bytes;
    return 0;
}

static int fake_mark_control(ds4_transport_lease *lease) {
    fake_lease_ctx *ctx = lease->transport->ctx;
    if (ctx->tx != lease) {
        errno = EINVAL;
        return -1;
    }
    lease->control_sent = 1;
    return 0;
}

static int fake_lease_commit(ds4_transport_lease *lease,
                             int gpu_quiesced) {
    fake_lease_ctx *ctx = lease->transport->ctx;
    ds4_transport_lease **active =
        lease->direction == DS4_TRANSPORT_LEASE_TX ? &ctx->tx : &ctx->rx;
    if (*active != lease) {
        errno = EINVAL;
        return -1;
    }
    *active = NULL;
    if (gpu_quiesced)
        ctx->gpu_quiesced_commits++;
    else
        ctx->conservative_commits++;
    return 0;
}

static int fake_lease_abort(ds4_transport_lease *lease) {
    fake_lease_ctx *ctx = lease->transport->ctx;
    ds4_transport_lease **active =
        lease->direction == DS4_TRANSPORT_LEASE_TX ? &ctx->tx : &ctx->rx;
    if (*active != lease) {
        errno = EINVAL;
        return -1;
    }
    if (lease->direction == DS4_TRANSPORT_LEASE_TX &&
        !lease->control_sent)
        return 0;
    *active = NULL;
    errno = ECANCELED;
    return -1;
}

static int fake_lease_abort_finish(ds4_transport_lease *lease) {
    fake_lease_ctx *ctx = lease->transport->ctx;
    if (ctx->tx != lease || lease->control_sent) {
        errno = EINVAL;
        return -1;
    }
    ctx->tx = NULL;
    return 0;
}

static int fake_check_prepare(ds4_transport *t) {
    fake_lease_ctx *ctx = t->ctx;
    if (!ctx->tx) return 0;
    errno = EBUSY;
    return -1;
}

static int fake_configure(ds4_transport *t,
                          uint32_t peer_frame_size,
                          uint32_t peer_ring_size,
                          char *err,
                          size_t errlen) {
    (void)t;
    (void)err;
    (void)errlen;
    if (peer_frame_size == 4096u && peer_ring_size == 8u) return 0;
    errno = EPROTO;
    return -1;
}

static size_t fake_max_oob(const ds4_transport *t) {
    (void)t;
    return 4096u * 7u - DS4_TRANSPORT_BULK_DESC_BYTES;
}

static void fake_close(ds4_transport *t) {
    free(t->ctx);
    t->ctx = NULL;
}

static const ds4_transport_ops fake_lease_ops = {
    .name = "fake-mapped",
    .caps = DS4_TRANSPORT_CAP_STREAM | DS4_TRANSPORT_CAP_ZEROCOPY,
    .mapped_leases_supported = fake_mapped_supported,
    .tx_lease_acquire = fake_tx_acquire,
    .rx_lease_acquire = fake_rx_acquire,
    .tx_lease_mark_control_sent = fake_mark_control,
    .lease_commit = fake_lease_commit,
    .lease_abort = fake_lease_abort,
    .lease_abort_finish = fake_lease_abort_finish,
    .check_prepare = fake_check_prepare,
    .configure = fake_configure,
    .max_oob_bytes = fake_max_oob,
    .close = fake_close,
};

static void test_descriptor_sequences(void) {
    int sv[2] = {-1, -1};
    char err[128] = "";
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    ds4_transport *left = ds4_transport_tcp_create(sv[0], err, sizeof(err));
    ds4_transport *right = ds4_transport_tcp_create(sv[1], err, sizeof(err));
    CHECK(left != NULL);
    CHECK(right != NULL);
    if (!left || !right) goto done;

    const uint64_t generation = 0x1122334455667788ull;
    CHECK(ds4_transport_configure_link(left, generation, 0, 0,
                                       err, sizeof(err)) == 0);
    CHECK(ds4_transport_configure_link(right, generation, 0, 0,
                                       err, sizeof(err)) == 0);
    CHECK(ds4_transport_generation(left) == generation);
    CHECK(ds4_transport_generation(right) == generation);

    static const char payload[] = "descriptor-framed payload";
    ds4_transport_bulk_desc desc;
    CHECK(ds4_transport_prepare_bulk(left,
                                     DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                                     17, 23, (uint32_t)sizeof(payload), 32,
                                     &desc, err, sizeof(err)) == 0);
    CHECK(desc.mode == DS4_TRANSPORT_BULK_TCP_INLINE);
    CHECK(desc.generation == generation);
    CHECK(desc.sequence == 1);
    CHECK(desc.frame_count == 0);

    ds4_transport_bulk_desc_wire wire;
    ds4_transport_bulk_desc decoded;
    CHECK(ds4_transport_bulk_desc_encode(&desc, &wire) == 0);
    CHECK(ds4_transport_bulk_desc_decode(&wire, &decoded) == 0);
    CHECK(memcmp(&decoded, &desc, sizeof(desc)) == 0);
    CHECK(ds4_transport_validate_inline_desc(
              &decoded, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              17, 23, (uint32_t)sizeof(payload), 32,
              err, sizeof(err)) == 0);

    /* TCP exposes the lease API as a clean, non-consuming fallback decision.
     * The prepared descriptor remains valid for the ordinary tagged path. */
    CHECK(ds4_transport_mapped_leases_supported(left) == 0);
    CHECK((ds4_transport_caps(left) & DS4_TRANSPORT_CAP_GPU_MAPPED) == 0);
    ds4_transport_lease *lease = (ds4_transport_lease *)(uintptr_t)1u;
    errno = 0;
    CHECK(ds4_transport_tx_lease_acquire(left, &desc, &lease,
                                         err, sizeof(err)) == -1);
    CHECK(errno == ENOTSUP);
    CHECK(lease == NULL);
    CHECK(ds4_transport_lease_host_ptr(NULL) == NULL);
    CHECK(ds4_transport_lease_device_ptr(NULL) == NULL);
    CHECK(ds4_transport_lease_bytes(NULL) == 0);
    ds4_transport_lease_release(NULL);

    char received[sizeof(payload)] = {0};
    CHECK(ds4_transport_send_bulk_desc(left, &desc,
                                       payload, sizeof(payload)) == 0);
    CHECK(ds4_transport_recv_bulk_desc(right, &decoded,
                                       received, sizeof(received)) == 1);
    CHECK(memcmp(received, payload, sizeof(payload)) == 0);

    /* Directional sequences are independent on the full-duplex link. */
    ds4_transport_bulk_desc reverse_desc;
    CHECK(ds4_transport_prepare_bulk(right,
                                     DS4_TRANSPORT_BULK_RESULT_HIDDEN,
                                     17, 23, (uint32_t)sizeof(payload), 32,
                                     &reverse_desc, err, sizeof(err)) == 0);
    CHECK(reverse_desc.sequence == 1);
    memset(received, 0, sizeof(received));
    CHECK(ds4_transport_send_bulk_desc(right, &reverse_desc,
                                       payload, sizeof(payload)) == 0);
    CHECK(ds4_transport_recv_bulk_desc(left, &reverse_desc,
                                       received, sizeof(received)) == 1);
    CHECK(memcmp(received, payload, sizeof(payload)) == 0);

    /* A descriptor from another connection generation is rejected before
     * reading any payload, and poisons this generation against retry. */
    ds4_transport_bulk_desc next_desc;
    CHECK(ds4_transport_prepare_bulk(left,
                                     DS4_TRANSPORT_BULK_INPUT_HIDDEN,
                                     17, 29, (uint32_t)sizeof(payload), 32,
                                     &next_desc, err, sizeof(err)) == 0);
    CHECK(ds4_transport_send_bulk_desc(left, &next_desc,
                                       payload, sizeof(payload)) == 0);
    ds4_transport_bulk_desc stale = next_desc;
    stale.generation--;
    errno = 0;
    CHECK(ds4_transport_recv_bulk_desc(right, &stale,
                                       received, sizeof(received)) == -1);
    CHECK(errno == EPROTO);
    errno = 0;
    CHECK(ds4_transport_recv_bulk_desc(right, &next_desc,
                                       received, sizeof(received)) == -1);
    CHECK(errno == EPROTO);

done:
    ds4_transport_release(right);
    ds4_transport_release(left);
    if (sv[1] >= 0) close(sv[1]);
    if (sv[0] >= 0) close(sv[0]);
}

static void test_mapped_lease_core(void) {
    int sv[2] = {-1, -1};
    char err[128] = "";
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    fake_lease_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    if (!ctx) goto done;
    ds4_transport *t = ds4_transport_internal_create(
        &fake_lease_ops, sv[0], 0, ctx, 4096u, 8u, err, sizeof(err));
    CHECK(t != NULL);
    if (!t) {
        free(ctx);
        goto done;
    }
    CHECK(ds4_transport_configure_link(t, 77u, 4096u, 8u,
                                       err, sizeof(err)) == 0);
    CHECK(ds4_transport_mapped_leases_supported(t) == 1);
    CHECK((ds4_transport_caps(t) & DS4_TRANSPORT_CAP_GPU_MAPPED) != 0);

    ds4_transport_bulk_desc desc;
    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 5u, 128u, 16u,
              &desc, err, sizeof(err)) == 0);
    CHECK(desc.mode == DS4_TRANSPORT_BULK_NHI_OOB);
    CHECK(desc.sequence == 1u);

    ds4_transport_lease *lease = NULL;
    CHECK(ds4_transport_tx_lease_acquire(t, &desc, &lease,
                                         err, sizeof(err)) == 0);
    CHECK(lease != NULL);
    CHECK(ds4_transport_lease_host_ptr(lease) == ctx->host);
    CHECK(ds4_transport_lease_device_ptr(lease) == ctx->device);
    CHECK(ds4_transport_lease_bytes(lease) == 128u);

    /* A lease blocks a second prepare, and commit cannot precede the control
     * record. Neither condition consumes the directional sequence. */
    ds4_transport_bulk_desc blocked;
    errno = 0;
    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 6u, 128u, 16u,
              &blocked, err, sizeof(err)) == -1);
    CHECK(errno == EBUSY);
    errno = 0;
    CHECK(ds4_transport_lease_commit(lease) == -1);
    CHECK(errno == EPERM);
    errno = 0;
    CHECK(ds4_transport_lease_commit_gpu_quiesced(lease) == -1);
    CHECK(errno == EPERM);

    CHECK(ds4_transport_lease_abort(lease) == 0);
    CHECK(ds4_transport_lease_host_ptr(lease) == NULL);
    ds4_transport_lease_release(lease);
    lease = NULL;

    /* Recoverable abort rolls back exactly the immediately outstanding
     * prepare, so the next descriptor reuses sequence one. */
    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 7u, 128u, 16u,
              &desc, err, sizeof(err)) == 0);
    CHECK(desc.sequence == 1u);
    CHECK(ds4_transport_tx_lease_acquire(t, &desc, &lease,
                                         err, sizeof(err)) == 0);
    CHECK(ds4_transport_tx_lease_mark_control_sent(lease) == 0);
    CHECK(ds4_transport_lease_commit(lease) == 0);
    CHECK(ctx->conservative_commits == 1u);
    CHECK(ctx->gpu_quiesced_commits == 0u);
    CHECK(ds4_transport_lease_bytes(lease) == 0);
    ds4_transport_lease_release(lease);
    lease = NULL;

    /* RX commit independently advances the receive sequence. */
    CHECK(ds4_transport_rx_lease_acquire(t, &desc, &lease,
                                         err, sizeof(err)) == 0);
    CHECK(ds4_transport_lease_commit_gpu_quiesced(lease) == 0);
    CHECK(ctx->conservative_commits == 1u);
    CHECK(ctx->gpu_quiesced_commits == 1u);
    ds4_transport_lease_release(lease);
    lease = NULL;

    /* The scoped fast path has the same TX control and sequence rules while
     * forwarding the caller's quiescence assertion to the backend. */
    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 8u, 128u, 16u,
              &desc, err, sizeof(err)) == 0);
    CHECK(desc.sequence == 2u);
    CHECK(ds4_transport_tx_lease_acquire(t, &desc, &lease,
                                         err, sizeof(err)) == 0);
    CHECK(ds4_transport_tx_lease_mark_control_sent(lease) == 0);
    CHECK(ds4_transport_lease_commit_gpu_quiesced(lease) == 0);
    CHECK(ctx->conservative_commits == 1u);
    CHECK(ctx->gpu_quiesced_commits == 2u);
    ds4_transport_lease_release(lease);
    lease = NULL;

    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 9u, 128u, 16u,
              &desc, err, sizeof(err)) == 0);
    CHECK(desc.sequence == 3u);
    CHECK(ds4_transport_tx_lease_acquire(t, &desc, &lease,
                                         err, sizeof(err)) == 0);
    CHECK(ds4_transport_tx_lease_mark_control_sent(lease) == 0);
    errno = 0;
    CHECK(ds4_transport_lease_abort(lease) == -1);
    CHECK(errno == ECANCELED);
    ds4_transport_lease_release(lease);
    errno = 0;
    CHECK(ds4_transport_prepare_bulk(
              t, DS4_TRANSPORT_BULK_INPUT_HIDDEN, 3u, 10u, 128u, 16u,
              &desc, err, sizeof(err)) == -1);
    CHECK(errno == ECANCELED);

    ds4_transport_release(t);

done:
    if (sv[1] >= 0) close(sv[1]);
    if (sv[0] >= 0) close(sv[0]);
}

int main(void) {
    int sv[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return 1;

    char err[128] = "";
    ds4_transport *left = ds4_transport_tcp_dup(sv[0], err, sizeof(err));
    ds4_transport *right = ds4_transport_tcp_create(sv[1], err, sizeof(err));
    CHECK(left != NULL);
    CHECK(right != NULL);
    if (!left || !right) return 1;

    CHECK(ds4_transport_retain(left) == left);
    CHECK(ds4_transport_control_fd(left) >= 0);
    CHECK(ds4_transport_control_fd(left) != sv[0]);
    CHECK(strcmp(ds4_transport_name(left), "tcp") == 0);
    CHECK(ds4_transport_caps(left) == DS4_TRANSPORT_CAP_STREAM);

    /* The owned transport keeps its duplicate alive after the monitored
     * original descriptor is closed. */
    close(sv[0]);
    sv[0] = -1;

    static const char forward[] = "persistent transport";
    char buf[sizeof(forward)] = {0};
    CHECK(ds4_transport_send_bulk(left, forward, sizeof(forward)) == 0);
    CHECK(ds4_transport_recv_bulk(right, buf, sizeof(buf)) == 1);
    CHECK(memcmp(buf, forward, sizeof(forward)) == 0);

    static const char reverse[] = "full duplex";
    memset(buf, 0, sizeof(buf));
    CHECK(ds4_transport_send_bulk(right, reverse, sizeof(reverse)) == 0);
    CHECK(ds4_transport_recv_bulk(left, buf, sizeof(reverse)) == 1);
    CHECK(memcmp(buf, reverse, sizeof(reverse)) == 0);

    CHECK(ds4_transport_send_bulk(left, NULL, 0) == 0);
    CHECK(ds4_transport_recv_bulk(right, NULL, 0) == 1);

    ds4_transport_release(left);
    CHECK(ds4_transport_send_bulk(left, forward, sizeof(forward)) == 0);
    memset(buf, 0, sizeof(buf));
    CHECK(ds4_transport_recv_bulk(right, buf, sizeof(forward)) == 1);
    CHECK(memcmp(buf, forward, sizeof(forward)) == 0);

    CHECK(shutdown(ds4_transport_control_fd(left), SHUT_WR) == 0);
    CHECK(ds4_transport_recv_bulk(right, buf, 1) == 0);

    ds4_transport_release(right);
    right = NULL;
    close(sv[1]);
    sv[1] = -1;

    /* A closed peer is an ordinary write failure, never process-wide
     * SIGPIPE. This covers Linux MSG_NOSIGNAL and BSD/macOS SO_NOSIGPIPE. */
    CHECK(ds4_transport_send_bulk(left, forward, sizeof(forward)) == -1);
    ds4_transport_release(left);
    left = NULL;

    errno = 0;
    CHECK(ds4_transport_tcp_create(-1, err, sizeof(err)) == NULL);
    CHECK(errno == EINVAL);

    test_descriptor_sequences();
    test_mapped_lease_core();

    fprintf(stderr, "test_transport: %d/%d checks passed (%d failed)\n",
            checks - failures, checks, failures);
    return failures ? 1 : 0;
}
