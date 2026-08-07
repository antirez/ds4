#ifndef DS4_DIST_V3_H
#define DS4_DIST_V3_H

/* Standalone negotiation and frame-layout helpers for DS4 distributed v3.
 * The common DS4E frame header stays in ds4_distributed.c. Bulk descriptor
 * ownership, encoding, sequencing, and transport validation live in
 * ds4_transport.[ch]; this module deliberately uses that canonical record.
 */

#include <stddef.h>
#include <stdint.h>

#include "ds4_transport.h"

#define DS4_DIST_V3_PROTOCOL_VERSION 3u

/* v2 currently uses message types 1..10. */
#define DS4_DIST_MSG_HELLO_V3  11u
#define DS4_DIST_MSG_HELLO_ACK 12u
#define DS4_DIST_MSG_HELLO_READY 13u

#define DS4_DIST_V3_CAP_BULK_DESC_V1    0x00000001u
#define DS4_DIST_V3_CAP_NHI_CPU_COPY_V1 0x00000002u
#define DS4_DIST_V3_CAP_NHI_TAGGED_V1   0x00000004u
#define DS4_DIST_V3_CAP_NHI_V1 \
    (DS4_DIST_V3_CAP_BULK_DESC_V1 | \
     DS4_DIST_V3_CAP_NHI_CPU_COPY_V1 | \
     DS4_DIST_V3_CAP_NHI_TAGGED_V1)

#define DS4_DIST_V3_HELLO_EXT_BYTES     32u
#define DS4_DIST_V3_HELLO_ACK_BYTES     48u
#define DS4_DIST_V3_HELLO_READY_BYTES   24u
#define DS4_DIST_V3_WORK_FIXED_BYTES    84u
#define DS4_DIST_V3_RESULT_FIXED_BYTES  40u

typedef enum {
    DS4_DIST_V3_POLICY_AUTO = 0,
    DS4_DIST_V3_POLICY_REQUIRE_NHI = 1,
} ds4_dist_v3_transport_policy;

typedef enum {
    DS4_DIST_V3_TRANSPORT_TCP = 1,
    DS4_DIST_V3_TRANSPORT_NHI = 2,
} ds4_dist_v3_transport_kind;

/* Appended to the existing v2 HELLO fixed record in a HELLO_V3 payload. */
typedef struct {
    uint32_t protocol_min;
    uint32_t protocol_max;
    uint32_t capabilities;
    uint32_t transport_policy;
    uint32_t nhi_frame_size;
    uint32_t nhi_ring_size;
    uint32_t nhi_max_payload;
    uint32_t reserved;
} ds4_dist_v3_hello_ext;

typedef struct {
    uint32_t protocol_version;
    uint32_t coordinator_caps;
    uint32_t selected_caps;
    uint32_t selected_transport;
    uint32_t generation_hi;
    uint32_t generation_lo;
    uint32_t nhi_frame_size;
    uint32_t nhi_ring_size;
    uint32_t nhi_max_payload;
    uint32_t reserved[3];
} ds4_dist_v3_hello_ack;

/* Sent by the worker only after it has installed the ACK-selected link. The
 * coordinator must not publish the worker or dispatch WORK before receiving
 * and validating this barrier. */
typedef struct {
    uint32_t protocol_version;
    uint32_t selected_transport;
    uint32_t generation_hi;
    uint32_t generation_lo;
    uint32_t reserved[2];
} ds4_dist_v3_hello_ready;

void ds4_dist_v3_u64_to_halves(uint64_t value, uint32_t *hi, uint32_t *lo);
uint64_t ds4_dist_v3_u64_from_halves(uint32_t hi, uint32_t lo);

void ds4_dist_v3_hello_ext_to_wire(
        ds4_dist_v3_hello_ext *wire,
        const ds4_dist_v3_hello_ext *host);
void ds4_dist_v3_hello_ext_from_wire(
        ds4_dist_v3_hello_ext *host,
        const ds4_dist_v3_hello_ext *wire);
void ds4_dist_v3_hello_ack_to_wire(
        ds4_dist_v3_hello_ack *wire,
        const ds4_dist_v3_hello_ack *host);
void ds4_dist_v3_hello_ack_from_wire(
        ds4_dist_v3_hello_ack *host,
        const ds4_dist_v3_hello_ack *wire);
void ds4_dist_v3_hello_ready_to_wire(
        ds4_dist_v3_hello_ready *wire,
        const ds4_dist_v3_hello_ready *host);
void ds4_dist_v3_hello_ready_from_wire(
        ds4_dist_v3_hello_ready *host,
        const ds4_dist_v3_hello_ready *wire);

/* Validate a host-order offer or ACK. Unknown advertised capability bits are
 * ignored. Selected bits must be a supported subset of both peers' offers.
 */
int ds4_dist_v3_hello_ext_validate(
        const ds4_dist_v3_hello_ext *hello,
        char *err,
        size_t errlen);
int ds4_dist_v3_hello_ack_validate(
        const ds4_dist_v3_hello_ack *ack,
        const ds4_dist_v3_hello_ext *local_offer,
        char *err,
        size_t errlen);
int ds4_dist_v3_hello_ready_validate(
        const ds4_dist_v3_hello_ready *ready,
        const ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen);

/* Construct the host-order ACK selected from two already validated offers.
 * generation must be nonzero. NHI is selected only if both sides advertise
 * the complete NHI v1 bundle and have compatible geometry. AUTO otherwise
 * selects v3 TCP; REQUIRE_NHI rejects instead.
 */
int ds4_dist_v3_negotiate(
        const ds4_dist_v3_hello_ext *worker,
        const ds4_dist_v3_hello_ext *coordinator,
        uint64_t generation,
        ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen);

/* Frame sizes exclude the common 12-byte DS4E frame header. control_bytes is
 * the number of bytes retained on TCP when mode is NHI_OOB. Descriptors are
 * always present and use the canonical DS4_TRANSPORT_BULK_DESC_BYTES size.
 */
int ds4_dist_v3_work_frame_sizes(
        uint32_t token_bytes,
        uint32_t route_bytes,
        uint32_t bulk_payload_bytes,
        uint32_t mode,
        uint32_t *control_bytes,
        uint32_t *frame_bytes);
int ds4_dist_v3_result_frame_sizes(
        uint32_t telemetry_bytes,
        uint32_t control_payload_bytes,
        uint32_t bulk_payload_bytes,
        uint32_t mode,
        uint32_t *control_bytes,
        uint32_t *frame_bytes);

#endif
