#include "ds4_dist_v3.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int checks;
static int failures;

#define CHECK(expr) do { \
    checks++; \
    if (!(expr)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define NHI_FRAME_SIZE 4096u
#define NHI_RING_LARGE 4096u
#define NHI_RING_SMALL 2048u
#define NHI_MAX_LARGE \
    ((NHI_RING_LARGE - 1u) * NHI_FRAME_SIZE - DS4_TRANSPORT_BULK_DESC_BYTES)
#define NHI_MAX_SMALL \
    ((NHI_RING_SMALL - 1u) * NHI_FRAME_SIZE - DS4_TRANSPORT_BULK_DESC_BYTES)

static ds4_dist_v3_hello_ext tcp_offer(uint32_t policy) {
    ds4_dist_v3_hello_ext offer = {
        DS4_DIST_V3_PROTOCOL_VERSION,
        DS4_DIST_V3_PROTOCOL_VERSION,
        DS4_DIST_V3_CAP_BULK_DESC_V1,
        policy,
        0,
        0,
        0,
        0,
    };
    return offer;
}

static ds4_dist_v3_hello_ext nhi_offer(
        uint32_t policy,
        uint32_t ring_size,
        uint32_t max_payload) {
    ds4_dist_v3_hello_ext offer = {
        DS4_DIST_V3_PROTOCOL_VERSION,
        DS4_DIST_V3_PROTOCOL_VERSION,
        DS4_DIST_V3_CAP_NHI_V1,
        policy,
        NHI_FRAME_SIZE,
        ring_size,
        max_payload,
        0,
    };
    return offer;
}

static int hello_equal(
        const ds4_dist_v3_hello_ext *a,
        const ds4_dist_v3_hello_ext *b) {
    return a->protocol_min == b->protocol_min &&
        a->protocol_max == b->protocol_max &&
        a->capabilities == b->capabilities &&
        a->transport_policy == b->transport_policy &&
        a->nhi_frame_size == b->nhi_frame_size &&
        a->nhi_ring_size == b->nhi_ring_size &&
        a->nhi_max_payload == b->nhi_max_payload &&
        a->reserved == b->reserved;
}

static int ack_equal(
        const ds4_dist_v3_hello_ack *a,
        const ds4_dist_v3_hello_ack *b) {
    uint32_t i;
    if (a->protocol_version != b->protocol_version ||
        a->coordinator_caps != b->coordinator_caps ||
        a->selected_caps != b->selected_caps ||
        a->selected_transport != b->selected_transport ||
        a->generation_hi != b->generation_hi ||
        a->generation_lo != b->generation_lo ||
        a->nhi_frame_size != b->nhi_frame_size ||
        a->nhi_ring_size != b->nhi_ring_size ||
        a->nhi_max_payload != b->nhi_max_payload)
        return 0;
    for (i = 0; i < 3u; i++) {
        if (a->reserved[i] != b->reserved[i]) return 0;
    }
    return 1;
}

static int ready_equal(
        const ds4_dist_v3_hello_ready *a,
        const ds4_dist_v3_hello_ready *b) {
    return a->protocol_version == b->protocol_version &&
        a->selected_transport == b->selected_transport &&
        a->generation_hi == b->generation_hi &&
        a->generation_lo == b->generation_lo &&
        a->reserved[0] == b->reserved[0] &&
        a->reserved[1] == b->reserved[1];
}

static int desc_equal(
        const ds4_transport_bulk_desc *a,
        const ds4_transport_bulk_desc *b) {
    return a->mode == b->mode && a->kind == b->kind &&
        a->generation == b->generation && a->sequence == b->sequence &&
        a->session_id == b->session_id && a->request_id == b->request_id &&
        a->payload_bytes == b->payload_bytes &&
        a->element_bits == b->element_bits &&
        a->frame_count == b->frame_count && a->flags == b->flags &&
        a->reserved[0] == b->reserved[0] &&
        a->reserved[1] == b->reserved[1];
}

static void test_endian_round_trips(void) {
    ds4_dist_v3_hello_ext hello = {
        0x01020304u, 0x11121314u, 0x21222324u, 0x31323334u,
        0x41424344u, 0x51525354u, 0x61626364u, 0x71727374u,
    };
    ds4_dist_v3_hello_ext hello_wire;
    ds4_dist_v3_hello_ext hello_got;
    ds4_dist_v3_hello_ext_to_wire(&hello_wire, &hello);
    ds4_dist_v3_hello_ext_from_wire(&hello_got, &hello_wire);
    CHECK(hello_equal(&hello, &hello_got));

    ds4_dist_v3_hello_ack ack = {
        0x01020304u, 0x11121314u, 0x21222324u, 0x31323334u,
        0x41424344u, 0x51525354u, 0x61626364u, 0x71727374u,
        0x81828384u, {0x91929394u, 0xa1a2a3a4u, 0xb1b2b3b4u},
    };
    ds4_dist_v3_hello_ack ack_wire;
    ds4_dist_v3_hello_ack ack_got;
    ds4_dist_v3_hello_ack_to_wire(&ack_wire, &ack);
    ds4_dist_v3_hello_ack_from_wire(&ack_got, &ack_wire);
    CHECK(ack_equal(&ack, &ack_got));

    ds4_dist_v3_hello_ready ready = {
        0x01020304u,
        0x11121314u,
        0x21222324u,
        0x31323334u,
        {0x41424344u, 0x51525354u},
    };
    ds4_dist_v3_hello_ready ready_wire;
    ds4_dist_v3_hello_ready ready_got;
    ds4_dist_v3_hello_ready_to_wire(&ready_wire, &ready);
    ds4_dist_v3_hello_ready_from_wire(&ready_got, &ready_wire);
    CHECK(ready_equal(&ready, &ready_got));

    ds4_transport_bulk_desc desc = {
        DS4_TRANSPORT_BULK_NHI_OOB,
        DS4_TRANSPORT_BULK_RESULT_LOGITS,
        UINT64_C(0x0102030405060708),
        UINT64_C(0x1112131415161718),
        UINT64_C(0x2122232425262728),
        UINT64_C(0x3132333435363738),
        0x41424344u,
        32u,
        0x51525354u,
        0,
        {0, 0},
    };
    ds4_transport_bulk_desc_wire desc_wire;
    ds4_transport_bulk_desc desc_got;
    CHECK(ds4_transport_bulk_desc_encode(&desc, &desc_wire) == 0);
    CHECK(desc_wire.bytes[0] == 0 && desc_wire.bytes[1] == 0 &&
          desc_wire.bytes[2] == 0 &&
          desc_wire.bytes[3] == DS4_TRANSPORT_BULK_NHI_OOB);
    CHECK(desc_wire.bytes[8] == 0x01u && desc_wire.bytes[15] == 0x08u);
    CHECK(ds4_transport_bulk_desc_decode(&desc_wire, &desc_got) == 0);
    CHECK(desc_equal(&desc, &desc_got));
    CHECK(ds4_transport_bulk_desc_encode(NULL, &desc_wire) == -1);
    CHECK(ds4_transport_bulk_desc_decode(NULL, &desc_got) == -1);
}

static void test_offer_validation(void) {
    char err[160];
    ds4_dist_v3_hello_ext offer = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == 0);

    offer = nhi_offer(DS4_DIST_V3_POLICY_AUTO, NHI_RING_LARGE,
                      NHI_MAX_LARGE);
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == 0);

    offer.reserved = 1;
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    offer.capabilities = 0;
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer.capabilities = DS4_DIST_V3_CAP_BULK_DESC_V1 |
                         DS4_DIST_V3_CAP_NHI_CPU_COPY_V1;
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer = tcp_offer(DS4_DIST_V3_POLICY_REQUIRE_NHI);
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer = nhi_offer(DS4_DIST_V3_POLICY_AUTO, NHI_RING_LARGE,
                      NHI_MAX_LARGE + 1u);
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    offer.protocol_min = DS4_DIST_V3_PROTOCOL_VERSION + 1u;
    offer.protocol_max = offer.protocol_min;
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
    offer = tcp_offer(99u);
    CHECK(ds4_dist_v3_hello_ext_validate(&offer, err, sizeof(err)) == -1);
}

static void test_negotiation(void) {
    const uint64_t generation = UINT64_C(0x1020304050607080);
    char err[160];
    ds4_dist_v3_hello_ack ack;
    ds4_dist_v3_hello_ext worker = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    ds4_dist_v3_hello_ext coordinator = tcp_offer(DS4_DIST_V3_POLICY_AUTO);

    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation,
                                &ack, err, sizeof(err)) == 0);
    CHECK(ack.selected_transport == DS4_DIST_V3_TRANSPORT_TCP);
    CHECK(ack.selected_caps == DS4_DIST_V3_CAP_BULK_DESC_V1);
    CHECK(ds4_dist_v3_u64_from_halves(ack.generation_hi,
                                     ack.generation_lo) == generation);
    CHECK(ack.nhi_frame_size == 0 && ack.nhi_ring_size == 0 &&
          ack.nhi_max_payload == 0);
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == 0);

    worker = nhi_offer(DS4_DIST_V3_POLICY_AUTO, NHI_RING_LARGE,
                       NHI_MAX_LARGE);
    coordinator = nhi_offer(DS4_DIST_V3_POLICY_AUTO, NHI_RING_SMALL,
                            NHI_MAX_SMALL);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 1u,
                                &ack, err, sizeof(err)) == 0);
    CHECK(ack.selected_transport == DS4_DIST_V3_TRANSPORT_NHI);
    CHECK((ack.selected_caps & DS4_DIST_V3_CAP_NHI_V1) ==
          DS4_DIST_V3_CAP_NHI_V1);
    CHECK(ack.nhi_frame_size == NHI_FRAME_SIZE);
    CHECK(ack.nhi_ring_size == NHI_RING_SMALL);
    CHECK(ack.nhi_max_payload == NHI_MAX_SMALL);
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == 0);

    coordinator = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 2u,
                                &ack, err, sizeof(err)) == 0);
    CHECK(ack.selected_transport == DS4_DIST_V3_TRANSPORT_TCP);

    worker.transport_policy = DS4_DIST_V3_POLICY_REQUIRE_NHI;
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 3u,
                                &ack, err, sizeof(err)) == -1);
    CHECK(errno == EPROTONOSUPPORT);

    worker = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    coordinator = nhi_offer(DS4_DIST_V3_POLICY_REQUIRE_NHI,
                            NHI_RING_LARGE, NHI_MAX_LARGE);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 4u,
                                &ack, err, sizeof(err)) == -1);
    CHECK(ds4_dist_v3_negotiate(&worker, &worker, 0,
                                &ack, err, sizeof(err)) == -1);

    worker = nhi_offer(DS4_DIST_V3_POLICY_REQUIRE_NHI,
                       NHI_RING_LARGE, NHI_MAX_LARGE);
    coordinator = nhi_offer(DS4_DIST_V3_POLICY_AUTO,
                            NHI_RING_SMALL, NHI_MAX_SMALL);
    coordinator.nhi_frame_size = 8192u;
    coordinator.nhi_max_payload =
        (NHI_RING_SMALL - 1u) * coordinator.nhi_frame_size -
        DS4_TRANSPORT_BULK_DESC_BYTES;
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 5u,
                                &ack, err, sizeof(err)) == -1);

    worker = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    coordinator = tcp_offer(DS4_DIST_V3_POLICY_AUTO);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 6u,
                                &ack, err, sizeof(err)) == 0);
    ack.generation_hi = 0;
    ack.generation_lo = 0;
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == -1);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 7u,
                                &ack, err, sizeof(err)) == 0);
    ack.reserved[1] = 1;
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == -1);
    ack.reserved[1] = 0;
    ack.selected_caps |= DS4_DIST_V3_CAP_NHI_CPU_COPY_V1;
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == -1);

    worker = nhi_offer(DS4_DIST_V3_POLICY_REQUIRE_NHI,
                       NHI_RING_LARGE, NHI_MAX_LARGE);
    coordinator = nhi_offer(DS4_DIST_V3_POLICY_AUTO,
                            NHI_RING_LARGE, NHI_MAX_LARGE);
    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation + 8u,
                                &ack, err, sizeof(err)) == 0);
    ack.selected_transport = DS4_DIST_V3_TRANSPORT_TCP;
    ack.selected_caps = DS4_DIST_V3_CAP_BULK_DESC_V1;
    ack.nhi_frame_size = 0;
    ack.nhi_ring_size = 0;
    ack.nhi_max_payload = 0;
    CHECK(ds4_dist_v3_hello_ack_validate(&ack, &worker,
                                         err, sizeof(err)) == -1);
}

static void test_frame_sizes(void) {
    uint32_t control = 0;
    uint32_t frame = 0;
    const uint32_t work_control = DS4_DIST_V3_WORK_FIXED_BYTES +
        DS4_TRANSPORT_BULK_DESC_BYTES + 8u + 12u;
    const uint32_t result_control = DS4_DIST_V3_RESULT_FIXED_BYTES +
        DS4_TRANSPORT_BULK_DESC_BYTES + 40u;

    CHECK(ds4_dist_v3_work_frame_sizes(8, 12, 16,
                                      DS4_TRANSPORT_BULK_TCP_INLINE,
                                      &control, &frame) == 0);
    CHECK(control == work_control && frame == work_control + 16u);
    CHECK(ds4_dist_v3_work_frame_sizes(8, 12, 16,
                                      DS4_TRANSPORT_BULK_NHI_OOB,
                                      &control, &frame) == 0);
    CHECK(control == work_control && frame == work_control);
    CHECK(ds4_dist_v3_work_frame_sizes(8, 12, 0,
                                      DS4_TRANSPORT_BULK_NONE,
                                      &control, &frame) == 0);
    CHECK(control == work_control && frame == work_control);
    CHECK(ds4_dist_v3_work_frame_sizes(8, 12, 1,
                                      DS4_TRANSPORT_BULK_NONE,
                                      &control, &frame) == -1);
    CHECK(ds4_dist_v3_work_frame_sizes(8, 12, 0,
                                      DS4_TRANSPORT_BULK_TCP_INLINE,
                                      &control, &frame) == -1);
    CHECK(ds4_dist_v3_work_frame_sizes(UINT32_MAX, 0, 0,
                                      DS4_TRANSPORT_BULK_NONE,
                                      &control, &frame) == -1);
    CHECK(errno == EOVERFLOW);

    CHECK(ds4_dist_v3_result_frame_sizes(40, 0, 100,
                                        DS4_TRANSPORT_BULK_TCP_INLINE,
                                        &control, &frame) == 0);
    CHECK(control == result_control && frame == result_control + 100u);
    CHECK(ds4_dist_v3_result_frame_sizes(40, 0, 100,
                                        DS4_TRANSPORT_BULK_NHI_OOB,
                                        &control, &frame) == 0);
    CHECK(control == result_control && frame == result_control);
    CHECK(ds4_dist_v3_result_frame_sizes(40, 17, 0,
                                        DS4_TRANSPORT_BULK_NONE,
                                        &control, &frame) == 0);
    CHECK(control == result_control + 17u && frame == result_control + 17u);
    CHECK(ds4_dist_v3_result_frame_sizes(40, 1, 100,
                                        DS4_TRANSPORT_BULK_TCP_INLINE,
                                        &control, &frame) == -1);
    CHECK(ds4_dist_v3_result_frame_sizes(UINT32_MAX, 0, 0,
                                        DS4_TRANSPORT_BULK_NONE,
                                        &control, &frame) == -1);
    CHECK(errno == EOVERFLOW);
}

static void test_ready_barrier(void) {
    const uint64_t generation = UINT64_C(0x123456789abcdef0);
    char err[160];
    ds4_dist_v3_hello_ext worker = nhi_offer(
        DS4_DIST_V3_POLICY_AUTO, NHI_RING_LARGE, NHI_MAX_LARGE);
    ds4_dist_v3_hello_ext coordinator = nhi_offer(
        DS4_DIST_V3_POLICY_AUTO, NHI_RING_SMALL, NHI_MAX_SMALL);
    ds4_dist_v3_hello_ack ack;
    ds4_dist_v3_hello_ready ready = {0};

    CHECK(ds4_dist_v3_negotiate(&worker, &coordinator, generation,
                                &ack, err, sizeof(err)) == 0);
    ready.protocol_version = DS4_DIST_V3_PROTOCOL_VERSION;
    ready.selected_transport = ack.selected_transport;
    ready.generation_hi = ack.generation_hi;
    ready.generation_lo = ack.generation_lo;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == 0);

    ready.protocol_version++;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == -1);
    ready.protocol_version = DS4_DIST_V3_PROTOCOL_VERSION;
    ready.selected_transport = DS4_DIST_V3_TRANSPORT_TCP;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == -1);
    ready.selected_transport = ack.selected_transport;
    ready.generation_lo++;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == -1);
    ready.generation_hi = 0;
    ready.generation_lo = 0;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == -1);
    ready.generation_hi = ack.generation_hi;
    ready.generation_lo = ack.generation_lo;
    ready.reserved[1] = 1;
    CHECK(ds4_dist_v3_hello_ready_validate(&ready, &ack,
                                           err, sizeof(err)) == -1);
    CHECK(ds4_dist_v3_hello_ready_validate(NULL, &ack,
                                           err, sizeof(err)) == -1);
}

static void test_descriptor_validation(void) {
    char err[160];
    ds4_transport_bulk_desc desc = {
        DS4_TRANSPORT_BULK_TCP_INLINE,
        DS4_TRANSPORT_BULK_INPUT_HIDDEN,
        9,
        1,
        11,
        12,
        4096,
        16,
        0,
        0,
        {0, 0},
    };
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == 0);
    desc.frame_count = 1;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == -1);
    desc.frame_count = 0;
    desc.flags = 1;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == -1);
    desc.flags = 0;
    desc.generation = 0;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == -1);
    desc.generation = 9;
    desc.sequence = 0;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == -1);
    desc.sequence = 1;
    desc.reserved[1] = 1;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 12, 4096, 16, err, sizeof(err)) == -1);
    desc.reserved[1] = 0;
    CHECK(ds4_transport_validate_inline_desc(
              &desc, DS4_TRANSPORT_BULK_INPUT_HIDDEN,
              11, 13, 4096, 16, err, sizeof(err)) == -1);

    desc.mode = DS4_TRANSPORT_BULK_NHI_OOB;
    desc.frame_count = 2;
    ds4_transport_bulk_desc_wire wire;
    ds4_transport_bulk_desc got;
    CHECK(ds4_transport_bulk_desc_encode(&desc, &wire) == 0);
    CHECK(ds4_transport_bulk_desc_decode(&wire, &got) == 0);
    CHECK(desc_equal(&desc, &got));
}

int main(void) {
    CHECK(sizeof(ds4_dist_v3_hello_ext) == DS4_DIST_V3_HELLO_EXT_BYTES);
    CHECK(sizeof(ds4_dist_v3_hello_ack) == DS4_DIST_V3_HELLO_ACK_BYTES);
    CHECK(sizeof(ds4_dist_v3_hello_ready) == DS4_DIST_V3_HELLO_READY_BYTES);
    CHECK(sizeof(ds4_transport_bulk_desc_wire) ==
          DS4_TRANSPORT_BULK_DESC_BYTES);

    test_endian_round_trips();
    test_offer_validation();
    test_negotiation();
    test_frame_sizes();
    test_ready_barrier();
    test_descriptor_validation();

    fprintf(stderr, "test_dist_v3: %d/%d checks passed (%d failed)\n",
            checks - failures, checks, failures);
    return failures ? 1 : 0;
}
