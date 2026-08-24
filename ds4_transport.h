#ifndef DS4_TRANSPORT_H
#define DS4_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

/* Bulk boundary-tensor transport. Control and negotiation traffic stay on the
 * distributed TCP socket; only hidden-state and batched-logit payloads move
 * through this vtable. A zero-copy NHI/USB4STREAM backend can replace that
 * path once peers negotiate an out-of-band bulk descriptor.
 */

typedef struct ds4_transport ds4_transport;
typedef struct ds4_transport_lease ds4_transport_lease;

#define DS4_TRANSPORT_CAP_STREAM   0x1u /* reliable byte stream (TCP) */
#define DS4_TRANSPORT_CAP_ZEROCOPY 0x2u /* out-of-band zero-copy payloads */
#define DS4_TRANSPORT_CAP_GPU_MAPPED 0x4u /* direct mapped-slot GPU leases */

#define DS4_TRANSPORT_BULK_DESC_BYTES 64u

typedef enum {
    DS4_TRANSPORT_BULK_NONE = 0,
    DS4_TRANSPORT_BULK_TCP_INLINE = 1,
    DS4_TRANSPORT_BULK_NHI_OOB = 2,
} ds4_transport_bulk_mode;

typedef enum {
    DS4_TRANSPORT_BULK_INPUT_HIDDEN = 1,
    DS4_TRANSPORT_BULK_RESULT_HIDDEN = 2,
    DS4_TRANSPORT_BULK_RESULT_LOGITS = 3,
} ds4_transport_bulk_kind;

/* Host-order v3 descriptor. The corresponding wire record is always exactly
 * 64 bytes; encode/decode helpers avoid exposing host padding or byte order.
 * Local NHI slot indices are deliberately absent because the peer RX cursor
 * is independent of the sender's TX cursor. */
typedef struct {
    uint32_t mode;
    uint32_t kind;
    uint64_t generation;
    uint64_t sequence;
    uint64_t session_id;
    uint64_t request_id;
    uint32_t payload_bytes;
    uint32_t element_bits;
    uint32_t frame_count;
    uint32_t flags;
    uint32_t reserved[2];
} ds4_transport_bulk_desc;

typedef struct {
    uint8_t bytes[DS4_TRANSPORT_BULK_DESC_BYTES];
} ds4_transport_bulk_desc_wire;

/* Byte-stream primitives shared by the control path and the TCP bulk backend.
 * write returns 0/-1; read returns 1 on success, 0 on peer close, -1 on error.
 */
int ds4_transport_tcp_write(int fd, const void *buf, size_t len);
int ds4_transport_tcp_read(int fd, void *buf, size_t len);

/* Create a persistent TCP bulk transport over an established control/data fd.
 * tcp_create() borrows fd; tcp_dup() owns a duplicate that is closed with the
 * final transport reference. Transports support concurrent full-duplex sender
 * and receiver paths.
 */
ds4_transport *ds4_transport_tcp_create(int fd, char *err, size_t errlen);
ds4_transport *ds4_transport_tcp_dup(int fd, char *err, size_t errlen);
ds4_transport *ds4_transport_retain(ds4_transport *t);
void ds4_transport_release(ds4_transport *t);
int ds4_transport_control_fd(const ds4_transport *t);

/* Open and map one exclusive /dev/tbstreamX zero-copy endpoint. create()
 * borrows control_fd; dup() owns a duplicate. Both are Linux-only and return
 * NULL with ENOTSUP on other platforms. The link must be configured with the
 * generation and peer geometry selected by HELLO before preparing OOB bulk. */
ds4_transport *ds4_transport_nhi_create(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen);
ds4_transport *ds4_transport_nhi_dup(
        int control_fd,
        const char *device_path,
        char *err,
        size_t errlen);

/* Configure the negotiated connection identity and peer receive geometry.
 * TCP accepts zero peer geometry. NHI requires its peer frame size to match
 * and limits each message and aggregate in-flight frames to the smaller ring.
 */
int ds4_transport_configure_link(
        ds4_transport *t,
        uint64_t generation,
        uint32_t peer_frame_size,
        uint32_t peer_ring_size,
        char *err,
        size_t errlen);
uint64_t ds4_transport_generation(const ds4_transport *t);
uint32_t ds4_transport_frame_size(const ds4_transport *t);
uint32_t ds4_transport_ring_size(const ds4_transport *t);
size_t ds4_transport_max_oob_bytes(const ds4_transport *t);
int ds4_transport_can_oob(const ds4_transport *t, size_t payload_bytes);

/* Allocate the next directional sequence and select NHI only when this link
 * is configured and the complete tagged message fits. Oversized payloads are
 * described as inline TCP before any OOB submission can become ambiguous.
 * The caller must serialize prepare -> complete TCP control frame -> tagged
 * send for each direction; this is the same per-link write lock required to
 * keep control frames ordered on the TCP socket. */
int ds4_transport_prepare_bulk(
        ds4_transport *t,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        ds4_transport_bulk_desc *desc,
        char *err,
        size_t errlen);
int ds4_transport_validate_inline_desc(
        const ds4_transport_bulk_desc *desc,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        char *err,
        size_t errlen);
int ds4_transport_validate_oob_desc(
        const ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        uint32_t kind,
        uint64_t session_id,
        uint64_t request_id,
        uint32_t payload_bytes,
        uint32_t element_bits,
        char *err,
        size_t errlen);
int ds4_transport_bulk_desc_encode(
        const ds4_transport_bulk_desc *desc,
        ds4_transport_bulk_desc_wire *wire);
int ds4_transport_bulk_desc_decode(
        const ds4_transport_bulk_desc_wire *wire,
        ds4_transport_bulk_desc *desc);

int ds4_transport_send_bulk(ds4_transport *t, const void *buf, size_t len);
int ds4_transport_recv_bulk(ds4_transport *t, void *buf, size_t len);
int ds4_transport_send_bulk_desc(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        const void *buf,
        size_t len);
int ds4_transport_recv_bulk_desc(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        void *buf,
        size_t len);

/* Optional direct access to one mapped NHI payload. A lease retains its
 * transport and only exposes a single contiguous payload span. Acquire
 * returns ENOTSUP without consuming a cursor/event when mapping is absent or
 * the payload wraps, allowing the same prepared descriptor to use the copy
 * send/receive API instead. Only one lease per direction may be active.
 *
 * TX order is: acquire, fill through device_ptr, write the complete TCP
 * control frame, mark_control_sent, commit, release. Commit synchronizes GPU
 * ownership before NHI submission. Aborting before any TCP control write has
 * been attempted is safe and rolls back only that immediately outstanding
 * prepare sequence. A partial/failed control write must abandon the generation
 * (mark then abort, or shut down the socket); absence of mark_control_sent does
 * not make bytes already written recoverable. Aborting after the mark is fatal
 * because the peer already expects OOB data.
 *
 * RX acquire validates the queued event and tagged envelope before exposing
 * memory. Commit synchronizes GPU ownership, reposts the exact frames, and
 * advances receive sequence. An RX abort is necessarily fatal. Pointers and
 * byte count are valid only while the lease remains active.
 *
 * commit_gpu_quiesced() is the scoped fast path for a caller that has already
 * completed every GPU read or write touching the lease (or never exposed the
 * lease to GPU work). It preserves all transport and DMA ownership operations
 * while avoiding a redundant backend GPU synchronization. */
int ds4_transport_mapped_leases_supported(const ds4_transport *t);
int ds4_transport_tx_lease_acquire(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        ds4_transport_lease **out,
        char *err,
        size_t errlen);
int ds4_transport_rx_lease_acquire(
        ds4_transport *t,
        const ds4_transport_bulk_desc *desc,
        ds4_transport_lease **out,
        char *err,
        size_t errlen);
void *ds4_transport_lease_host_ptr(const ds4_transport_lease *lease);
void *ds4_transport_lease_device_ptr(const ds4_transport_lease *lease);
size_t ds4_transport_lease_bytes(const ds4_transport_lease *lease);
int ds4_transport_tx_lease_mark_control_sent(ds4_transport_lease *lease);
int ds4_transport_lease_commit(ds4_transport_lease *lease);
int ds4_transport_lease_commit_gpu_quiesced(ds4_transport_lease *lease);
int ds4_transport_lease_abort(ds4_transport_lease *lease);
void ds4_transport_lease_release(ds4_transport_lease *lease);

const char *ds4_transport_name(const ds4_transport *t);
uint32_t ds4_transport_caps(const ds4_transport *t);

#endif
