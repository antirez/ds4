#ifndef DS4_TP_IO_H
#define DS4_TP_IO_H

/* Both directions advance independently of socket or USB ring capacity. */
#include <errno.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static double ds4_tp_io_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int ds4_tp_io_exchange(int fd, bool device, const void *out, void *in,
        uint64_t bytes, uint64_t timeout_ms, const atomic_bool *cancelled) {
    if (fd < 0 || !out || !in || !bytes || !timeout_ms || bytes > SIZE_MAX) {
        errno = EINVAL;
        return 0;
    }
    const double deadline = ds4_tp_io_now() + (double)timeout_ms / 1000.0;
    uint64_t sent = 0, received = 0;
    while (sent < bytes || received < bytes) {
        if (cancelled && atomic_load_explicit(cancelled, memory_order_acquire)) {
            errno = ECANCELED;
            goto fail;
        }
        const double remaining = deadline - ds4_tp_io_now();
        if (remaining <= 0) { errno = ETIMEDOUT; goto fail; }
        bool progress = false;
        if (sent < bytes) {
            size_t n = bytes - sent > 2097152u ? 2097152u : (size_t)(bytes - sent);
            ssize_t r = device ? write(fd, (const char *)out + sent, n) :
                send(fd, (const char *)out + sent, n, MSG_DONTWAIT | MSG_NOSIGNAL);
            if (r > 0) { sent += (uint64_t)r; progress = true; }
            else if (!r) { errno = EPIPE; goto fail; }
            else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) goto fail;
        }
        if (received < bytes) {
            size_t n = bytes - received > 2097152u ? 2097152u : (size_t)(bytes - received);
            ssize_t r = device ? read(fd, (char *)in + received, n) :
                recv(fd, (char *)in + received, n, MSG_DONTWAIT);
            if (r > 0) { received += (uint64_t)r; progress = true; }
            else if (!r) { errno = ECONNRESET; goto fail; }
            else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) goto fail;
        }
        if (!progress) {
            struct pollfd p = {fd, (short)((sent < bytes ? POLLOUT : 0) |
                                           (received < bytes ? POLLIN : 0)), 0};
            int wait_ms = remaining > .05 ? 50 : (int)(remaining * 1000.0) + 1;
            int rc = poll(&p, 1, wait_ms);
            if (rc < 0 && errno != EINTR) goto fail;
            if (rc > 0 && (p.revents & POLLNVAL)) { errno = EBADF; goto fail; }
            if (rc > 0 && (p.revents & POLLERR)) { errno = EIO; goto fail; }
            /* HUP can coexist with buffered data. Let read report EOF. */
        }
    }
    return 1;
fail:
    {
        int saved_errno = errno;
        fprintf(stderr, "ds4-tp: %s I/O failed: sent=%llu received=%llu expected=%llu: %s\n",
                device ? "device" : "socket", (unsigned long long)sent,
                (unsigned long long)received, (unsigned long long)bytes,
                strerror(saved_errno));
        errno = saved_errno;
    }
    return 0;
}
#endif
