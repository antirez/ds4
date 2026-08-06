#ifndef DS4_DIST_DCHAN_H
#define DS4_DIST_DCHAN_H

#include <stddef.h>

/* Distributed data-plane channel.
 *
 * The distributed protocol reads/writes WORK, RESULT and snapshot frames over
 * a channel handle instead of a raw socket fd, so the same protocol code runs
 * over either transport. Two implementations exist: a thin TCP wrapper over the
 * existing dist_write_full/dist_read_full helpers (behavior-identical to the
 * pre-seam code) and, on Apple builds, an RDMA UC-QP channel (see
 * ds4_distributed_rdma.c). Control/bootstrap traffic (HELLO, registration,
 * route) still uses raw fds: the RDMA connection is established during the TCP
 * bootstrap and only the steady-state data plane runs over the channel.
 *
 * read() returns the tri-state that dist_read_full uses (1 ok / 0 clean EOF /
 * -1 error) and write() returns 0 on success / -1 on error, so channel-based
 * frame code keeps the exact control flow of the fd-based code it replaces.
 */
typedef struct ds4_dist_dchan ds4_dist_dchan;

/* Per-implementation operations. ctx is the implementation's private state.
 * begin_message and flush are optional (NULL for transports that never
 * buffer). */
typedef struct {
    int  (*write)(void *ctx, const void *buf, size_t len);  /* 0 ok / -1 err        */
    int  (*read)(void *ctx, void *buf, size_t len);         /* 1 ok / 0 EOF / -1 err */
    void (*shutdown)(void *ctx);                            /* best-effort half-close */
    void (*close)(void *ctx);                               /* release transport + ctx */
    int  (*fd)(void *ctx);                                  /* socket fd, or -1        */
    void (*begin_message)(void *ctx, size_t bytes);         /* next `bytes` = 1 message */
    int  (*flush)(void *ctx);                               /* publish buffered prefix */
} ds4_dist_dchan_ops;

struct ds4_dist_dchan {
    const ds4_dist_dchan_ops *ops;
    void *ctx;
};

/* Dispatchers (defined in ds4_distributed.c). NULL channel is tolerated by
 * close/shutdown; write/read/fd require a valid channel. */
int  ds4_dist_dchan_write(ds4_dist_dchan *c, const void *buf, size_t len);
int  ds4_dist_dchan_read(ds4_dist_dchan *c, void *buf, size_t len);
/* Declare that the next `bytes` written form one logical message, so a buffering
 * transport knows where to flush. A datagram-framed transport (RDMA) coalesces
 * the small writes a message is built from into full frames and puts the last,
 * partial frame on the wire when the declared byte count is reached; without the
 * declaration it must flush every write. TCP ignores it. Not calling it is only
 * a performance bug, never a correctness one, and the whole message must be
 * written by the thread that declared it (true of every frame in the protocol,
 * which is already serialized per channel by the caller). */
void ds4_dist_dchan_begin_message(ds4_dist_dchan *c, size_t bytes);
int  ds4_dist_dchan_flush(ds4_dist_dchan *c);
void ds4_dist_dchan_shutdown(ds4_dist_dchan *c);
void ds4_dist_dchan_close(ds4_dist_dchan *c);
int  ds4_dist_dchan_fd(ds4_dist_dchan *c);

/* TCP channel taking ownership of a connected socket fd (closed by
 * ds4_dist_dchan_close). Returns NULL on allocation failure. */
ds4_dist_dchan *ds4_dist_dchan_from_fd(int fd);

#endif /* DS4_DIST_DCHAN_H */
