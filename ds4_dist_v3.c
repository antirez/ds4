/* Standalone DS4 distributed protocol v3 negotiation and frame helpers. */

#include "ds4_dist_v3.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>

typedef char ds4_dist_v3_hello_ext_size_check[
    sizeof(ds4_dist_v3_hello_ext) == DS4_DIST_V3_HELLO_EXT_BYTES ? 1 : -1];
typedef char ds4_dist_v3_hello_ack_size_check[
    sizeof(ds4_dist_v3_hello_ack) == DS4_DIST_V3_HELLO_ACK_BYTES ? 1 : -1];
typedef char ds4_dist_v3_hello_ready_size_check[
    sizeof(ds4_dist_v3_hello_ready) == DS4_DIST_V3_HELLO_READY_BYTES ? 1 : -1];

static int v3_fail(int error, char *err, size_t errlen, const char *message) {
    errno = error;
    if (err && errlen != 0) snprintf(err, errlen, "%s", message);
    return -1;
}

static int v3_add_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (!out) {
        errno = EINVAL;
        return -1;
    }
    if (a > UINT32_MAX - b) {
        errno = EOVERFLOW;
        return -1;
    }
    *out = a + b;
    return 0;
}

static int v3_mode_valid(uint32_t mode) {
    return mode == DS4_TRANSPORT_BULK_NONE ||
           mode == DS4_TRANSPORT_BULK_TCP_INLINE ||
           mode == DS4_TRANSPORT_BULK_NHI_OOB;
}

static int v3_policy_valid(uint32_t policy) {
    return policy == DS4_DIST_V3_POLICY_AUTO ||
           policy == DS4_DIST_V3_POLICY_REQUIRE_NHI;
}

#define DS4_DIST_V3_CAP_NHI_OPTIONAL \
    (DS4_DIST_V3_CAP_NHI_CPU_COPY_V1 | DS4_DIST_V3_CAP_NHI_TAGGED_V1)

static int v3_nhi_geometry_valid(
        uint32_t frame_size,
        uint32_t ring_size,
        uint32_t max_payload) {
    uint64_t capacity;
    if (frame_size == 0 || ring_size < 2u || max_payload == 0) return 0;
    capacity = (uint64_t)(ring_size - 1u) * frame_size;
    return capacity > DS4_TRANSPORT_BULK_DESC_BYTES &&
        max_payload <= capacity - DS4_TRANSPORT_BULK_DESC_BYTES;
}

void ds4_dist_v3_u64_to_halves(uint64_t value, uint32_t *hi, uint32_t *lo) {
    if (hi) *hi = (uint32_t)(value >> 32);
    if (lo) *lo = (uint32_t)value;
}

uint64_t ds4_dist_v3_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

void ds4_dist_v3_hello_ext_to_wire(
        ds4_dist_v3_hello_ext *wire,
        const ds4_dist_v3_hello_ext *host) {
    ds4_dist_v3_hello_ext value;
    if (!wire || !host) return;
    value = *host;
    value.protocol_min = htonl(value.protocol_min);
    value.protocol_max = htonl(value.protocol_max);
    value.capabilities = htonl(value.capabilities);
    value.transport_policy = htonl(value.transport_policy);
    value.nhi_frame_size = htonl(value.nhi_frame_size);
    value.nhi_ring_size = htonl(value.nhi_ring_size);
    value.nhi_max_payload = htonl(value.nhi_max_payload);
    value.reserved = htonl(value.reserved);
    *wire = value;
}

void ds4_dist_v3_hello_ext_from_wire(
        ds4_dist_v3_hello_ext *host,
        const ds4_dist_v3_hello_ext *wire) {
    ds4_dist_v3_hello_ext value;
    if (!host || !wire) return;
    value = *wire;
    value.protocol_min = ntohl(value.protocol_min);
    value.protocol_max = ntohl(value.protocol_max);
    value.capabilities = ntohl(value.capabilities);
    value.transport_policy = ntohl(value.transport_policy);
    value.nhi_frame_size = ntohl(value.nhi_frame_size);
    value.nhi_ring_size = ntohl(value.nhi_ring_size);
    value.nhi_max_payload = ntohl(value.nhi_max_payload);
    value.reserved = ntohl(value.reserved);
    *host = value;
}

void ds4_dist_v3_hello_ack_to_wire(
        ds4_dist_v3_hello_ack *wire,
        const ds4_dist_v3_hello_ack *host) {
    ds4_dist_v3_hello_ack value;
    uint32_t i;
    if (!wire || !host) return;
    value = *host;
    value.protocol_version = htonl(value.protocol_version);
    value.coordinator_caps = htonl(value.coordinator_caps);
    value.selected_caps = htonl(value.selected_caps);
    value.selected_transport = htonl(value.selected_transport);
    value.generation_hi = htonl(value.generation_hi);
    value.generation_lo = htonl(value.generation_lo);
    value.nhi_frame_size = htonl(value.nhi_frame_size);
    value.nhi_ring_size = htonl(value.nhi_ring_size);
    value.nhi_max_payload = htonl(value.nhi_max_payload);
    for (i = 0; i < 3u; i++) value.reserved[i] = htonl(value.reserved[i]);
    *wire = value;
}

void ds4_dist_v3_hello_ack_from_wire(
        ds4_dist_v3_hello_ack *host,
        const ds4_dist_v3_hello_ack *wire) {
    ds4_dist_v3_hello_ack value;
    uint32_t i;
    if (!host || !wire) return;
    value = *wire;
    value.protocol_version = ntohl(value.protocol_version);
    value.coordinator_caps = ntohl(value.coordinator_caps);
    value.selected_caps = ntohl(value.selected_caps);
    value.selected_transport = ntohl(value.selected_transport);
    value.generation_hi = ntohl(value.generation_hi);
    value.generation_lo = ntohl(value.generation_lo);
    value.nhi_frame_size = ntohl(value.nhi_frame_size);
    value.nhi_ring_size = ntohl(value.nhi_ring_size);
    value.nhi_max_payload = ntohl(value.nhi_max_payload);
    for (i = 0; i < 3u; i++) value.reserved[i] = ntohl(value.reserved[i]);
    *host = value;
}

void ds4_dist_v3_hello_ready_to_wire(
        ds4_dist_v3_hello_ready *wire,
        const ds4_dist_v3_hello_ready *host) {
    ds4_dist_v3_hello_ready value;
    uint32_t i;
    if (!wire || !host) return;
    value = *host;
    value.protocol_version = htonl(value.protocol_version);
    value.selected_transport = htonl(value.selected_transport);
    value.generation_hi = htonl(value.generation_hi);
    value.generation_lo = htonl(value.generation_lo);
    for (i = 0; i < 2u; i++) value.reserved[i] = htonl(value.reserved[i]);
    *wire = value;
}

void ds4_dist_v3_hello_ready_from_wire(
        ds4_dist_v3_hello_ready *host,
        const ds4_dist_v3_hello_ready *wire) {
    ds4_dist_v3_hello_ready value;
    uint32_t i;
    if (!host || !wire) return;
    value = *wire;
    value.protocol_version = ntohl(value.protocol_version);
    value.selected_transport = ntohl(value.selected_transport);
    value.generation_hi = ntohl(value.generation_hi);
    value.generation_lo = ntohl(value.generation_lo);
    for (i = 0; i < 2u; i++) value.reserved[i] = ntohl(value.reserved[i]);
    *host = value;
}

int ds4_dist_v3_hello_ext_validate(
        const ds4_dist_v3_hello_ext *hello,
        char *err,
        size_t errlen) {
    uint32_t nhi_bits;
    if (err && errlen != 0) err[0] = '\0';
    if (!hello)
        return v3_fail(EINVAL, err, errlen, "missing v3 HELLO extension");
    if (hello->protocol_min > DS4_DIST_V3_PROTOCOL_VERSION ||
        hello->protocol_max < DS4_DIST_V3_PROTOCOL_VERSION ||
        hello->protocol_min > hello->protocol_max)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO has no supported protocol revision");
    if (!v3_policy_valid(hello->transport_policy))
        return v3_fail(EPROTO, err, errlen, "v3 HELLO has an invalid transport policy");
    if (hello->reserved != 0)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO reserved field is nonzero");
    if ((hello->capabilities & DS4_DIST_V3_CAP_BULK_DESC_V1) == 0)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO lacks bulk descriptor support");

    nhi_bits = hello->capabilities & DS4_DIST_V3_CAP_NHI_OPTIONAL;
    if (nhi_bits != 0 && nhi_bits != DS4_DIST_V3_CAP_NHI_OPTIONAL)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO advertises an incomplete NHI capability bundle");
    if ((hello->capabilities & DS4_DIST_V3_CAP_NHI_V1) ==
        DS4_DIST_V3_CAP_NHI_V1) {
        if (!v3_nhi_geometry_valid(hello->nhi_frame_size,
                                   hello->nhi_ring_size,
                                   hello->nhi_max_payload))
            return v3_fail(EPROTO, err, errlen, "v3 HELLO has invalid NHI geometry");
    } else if (hello->nhi_frame_size != 0 || hello->nhi_ring_size != 0 ||
               hello->nhi_max_payload != 0) {
        return v3_fail(EPROTO, err, errlen, "v3 HELLO has NHI geometry without NHI capability");
    }
    if (hello->transport_policy == DS4_DIST_V3_POLICY_REQUIRE_NHI &&
        (hello->capabilities & DS4_DIST_V3_CAP_NHI_V1) !=
            DS4_DIST_V3_CAP_NHI_V1)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO requires unavailable NHI capability");
    return 0;
}

int ds4_dist_v3_hello_ack_validate(
        const ds4_dist_v3_hello_ack *ack,
        const ds4_dist_v3_hello_ext *local_offer,
        char *err,
        size_t errlen) {
    uint64_t generation;
    uint32_t common_caps;
    uint32_t i;
    if (err && errlen != 0) err[0] = '\0';
    if (!ack || !local_offer)
        return v3_fail(EINVAL, err, errlen, "missing v3 HELLO ACK validation parameters");
    if (ds4_dist_v3_hello_ext_validate(local_offer, err, errlen) != 0) return -1;
    if (ack->protocol_version != DS4_DIST_V3_PROTOCOL_VERSION)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO ACK selected an unsupported revision");
    for (i = 0; i < 3u; i++) {
        if (ack->reserved[i] != 0)
            return v3_fail(EPROTO, err, errlen, "v3 HELLO ACK reserved field is nonzero");
    }
    generation = ds4_dist_v3_u64_from_halves(ack->generation_hi,
                                              ack->generation_lo);
    if (generation == 0)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO ACK has a zero generation");
    if ((ack->coordinator_caps & DS4_DIST_V3_CAP_BULK_DESC_V1) == 0 ||
        ((ack->coordinator_caps & DS4_DIST_V3_CAP_NHI_OPTIONAL) != 0 &&
         (ack->coordinator_caps & DS4_DIST_V3_CAP_NHI_OPTIONAL) !=
             DS4_DIST_V3_CAP_NHI_OPTIONAL))
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO ACK advertises invalid coordinator capabilities");
    common_caps = local_offer->capabilities & ack->coordinator_caps;
    if ((ack->selected_caps & ~common_caps) != 0 ||
        (ack->selected_caps & DS4_DIST_V3_CAP_BULK_DESC_V1) == 0)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO ACK selected unsupported capabilities");

    if (ack->selected_transport == DS4_DIST_V3_TRANSPORT_TCP) {
        if (local_offer->transport_policy == DS4_DIST_V3_POLICY_REQUIRE_NHI)
            return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO ACK downgraded required NHI to TCP");
        if (ack->nhi_frame_size != 0 || ack->nhi_ring_size != 0 ||
            ack->nhi_max_payload != 0)
            return v3_fail(EPROTO, err, errlen, "v3 TCP ACK contains NHI geometry");
        if ((ack->selected_caps &
             (DS4_DIST_V3_CAP_NHI_CPU_COPY_V1 |
              DS4_DIST_V3_CAP_NHI_TAGGED_V1)) != 0)
            return v3_fail(EPROTO, err, errlen, "v3 TCP ACK selected NHI capabilities");
        return 0;
    }
    if (ack->selected_transport != DS4_DIST_V3_TRANSPORT_NHI)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO ACK selected an invalid transport");
    if ((ack->selected_caps & DS4_DIST_V3_CAP_NHI_V1) != DS4_DIST_V3_CAP_NHI_V1)
        return v3_fail(EPROTO, err, errlen, "v3 NHI ACK lacks the complete capability bundle");
    if (ack->nhi_frame_size != local_offer->nhi_frame_size ||
        ack->nhi_ring_size > local_offer->nhi_ring_size ||
        ack->nhi_max_payload > local_offer->nhi_max_payload ||
        !v3_nhi_geometry_valid(ack->nhi_frame_size,
                               ack->nhi_ring_size,
                               ack->nhi_max_payload))
        return v3_fail(EPROTO, err, errlen, "v3 HELLO ACK has incompatible NHI geometry");
    return 0;
}

int ds4_dist_v3_hello_ready_validate(
        const ds4_dist_v3_hello_ready *ready,
        const ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen) {
    uint64_t ready_generation;
    uint64_t ack_generation;
    uint32_t i;
    if (err && errlen != 0) err[0] = '\0';
    if (!ready || !ack)
        return v3_fail(EINVAL, err, errlen, "missing v3 HELLO READY validation parameters");
    if (ready->protocol_version != DS4_DIST_V3_PROTOCOL_VERSION ||
        ready->protocol_version != ack->protocol_version)
        return v3_fail(EPROTONOSUPPORT, err, errlen, "v3 HELLO READY protocol mismatch");
    if ((ready->selected_transport != DS4_DIST_V3_TRANSPORT_TCP &&
         ready->selected_transport != DS4_DIST_V3_TRANSPORT_NHI) ||
        ready->selected_transport != ack->selected_transport)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO READY transport mismatch");
    ready_generation = ds4_dist_v3_u64_from_halves(ready->generation_hi,
                                                    ready->generation_lo);
    ack_generation = ds4_dist_v3_u64_from_halves(ack->generation_hi,
                                                  ack->generation_lo);
    if (ready_generation == 0 || ack_generation == 0 ||
        ready_generation != ack_generation)
        return v3_fail(EPROTO, err, errlen, "v3 HELLO READY generation mismatch");
    for (i = 0; i < 2u; i++) {
        if (ready->reserved[i] != 0)
            return v3_fail(EPROTO, err, errlen, "v3 HELLO READY reserved field is nonzero");
    }
    return 0;
}

int ds4_dist_v3_negotiate(
        const ds4_dist_v3_hello_ext *worker,
        const ds4_dist_v3_hello_ext *coordinator,
        uint64_t generation,
        ds4_dist_v3_hello_ack *ack,
        char *err,
        size_t errlen) {
    uint32_t common_caps;
    int both_nhi;
    if (err && errlen != 0) err[0] = '\0';
    if (!worker || !coordinator || !ack || generation == 0)
        return v3_fail(EINVAL, err, errlen, "missing v3 negotiation parameters");
    if (ds4_dist_v3_hello_ext_validate(worker, err, errlen) != 0 ||
        ds4_dist_v3_hello_ext_validate(coordinator, err, errlen) != 0)
        return -1;

    common_caps = worker->capabilities & coordinator->capabilities;
    both_nhi = (common_caps & DS4_DIST_V3_CAP_NHI_V1) ==
        DS4_DIST_V3_CAP_NHI_V1 &&
        worker->nhi_frame_size == coordinator->nhi_frame_size;
    if (!both_nhi &&
        (worker->transport_policy == DS4_DIST_V3_POLICY_REQUIRE_NHI ||
         coordinator->transport_policy == DS4_DIST_V3_POLICY_REQUIRE_NHI))
        return v3_fail(EPROTONOSUPPORT, err, errlen, "required v3 NHI transport cannot be negotiated");

    *ack = (ds4_dist_v3_hello_ack){0};
    ack->protocol_version = DS4_DIST_V3_PROTOCOL_VERSION;
    ack->coordinator_caps = coordinator->capabilities;
    ack->selected_caps = common_caps & DS4_DIST_V3_CAP_BULK_DESC_V1;
    ds4_dist_v3_u64_to_halves(generation,
                              &ack->generation_hi, &ack->generation_lo);
    if (both_nhi) {
        uint32_t ring_size = worker->nhi_ring_size < coordinator->nhi_ring_size
            ? worker->nhi_ring_size : coordinator->nhi_ring_size;
        uint32_t max_payload = worker->nhi_max_payload < coordinator->nhi_max_payload
            ? worker->nhi_max_payload : coordinator->nhi_max_payload;
        ack->selected_transport = DS4_DIST_V3_TRANSPORT_NHI;
        ack->selected_caps |= DS4_DIST_V3_CAP_NHI_V1;
        ack->nhi_frame_size = worker->nhi_frame_size;
        ack->nhi_ring_size = ring_size;
        ack->nhi_max_payload = max_payload;
    } else {
        ack->selected_transport = DS4_DIST_V3_TRANSPORT_TCP;
    }
    return ds4_dist_v3_hello_ack_validate(ack, worker, err, errlen);
}

int ds4_dist_v3_work_frame_sizes(
        uint32_t token_bytes,
        uint32_t route_bytes,
        uint32_t bulk_payload_bytes,
        uint32_t mode,
        uint32_t *control_bytes,
        uint32_t *frame_bytes) {
    uint32_t control;
    uint32_t total;
    if (!control_bytes || !frame_bytes || !v3_mode_valid(mode) ||
        (mode == DS4_TRANSPORT_BULK_NONE && bulk_payload_bytes != 0) ||
        (mode != DS4_TRANSPORT_BULK_NONE && bulk_payload_bytes == 0)) {
        errno = EINVAL;
        return -1;
    }
    if (v3_add_u32(DS4_DIST_V3_WORK_FIXED_BYTES,
                   DS4_TRANSPORT_BULK_DESC_BYTES, &control) != 0 ||
        v3_add_u32(control, token_bytes, &control) != 0 ||
        v3_add_u32(control, route_bytes, &control) != 0)
        return -1;
    total = control;
    if (mode == DS4_TRANSPORT_BULK_TCP_INLINE &&
        v3_add_u32(total, bulk_payload_bytes, &total) != 0)
        return -1;
    *control_bytes = control;
    *frame_bytes = total;
    return 0;
}

int ds4_dist_v3_result_frame_sizes(
        uint32_t telemetry_bytes,
        uint32_t control_payload_bytes,
        uint32_t bulk_payload_bytes,
        uint32_t mode,
        uint32_t *control_bytes,
        uint32_t *frame_bytes) {
    uint32_t control;
    uint32_t total;
    if (!control_bytes || !frame_bytes || !v3_mode_valid(mode) ||
        (mode == DS4_TRANSPORT_BULK_NONE && bulk_payload_bytes != 0) ||
        (mode != DS4_TRANSPORT_BULK_NONE &&
         (bulk_payload_bytes == 0 || control_payload_bytes != 0))) {
        errno = EINVAL;
        return -1;
    }
    if (v3_add_u32(DS4_DIST_V3_RESULT_FIXED_BYTES,
                   DS4_TRANSPORT_BULK_DESC_BYTES, &control) != 0 ||
        v3_add_u32(control, telemetry_bytes, &control) != 0 ||
        v3_add_u32(control, control_payload_bytes, &control) != 0)
        return -1;
    total = control;
    if (mode == DS4_TRANSPORT_BULK_TCP_INLINE &&
        v3_add_u32(total, bulk_payload_bytes, &total) != 0)
        return -1;
    *control_bytes = control;
    *frame_bytes = total;
    return 0;
}
