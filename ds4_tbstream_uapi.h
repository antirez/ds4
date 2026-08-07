/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef DS4_TBSTREAM_UAPI_H
#define DS4_TBSTREAM_UAPI_H

/* Vendored userspace view of the zero-copy USB4STREAM UAPI deployed on the
 * Strix Halo peers. Keep ioctl numbers and record widths synchronized with
 * strix-rdma/kernel/zerocopy. */

#include <stdint.h>
#include <sys/ioctl.h>

#define TBSTREAM_ZC_FRAME_SIZE 4096u

struct tbstream_zc_info {
    uint32_t ring_size;
    uint32_t frame_size;
    uint64_t tx_pool_offset;
    uint64_t rx_pool_offset;
};

enum tbstream_zc_event_type {
    TBSTREAM_ZC_EV_RX = 0,
    TBSTREAM_ZC_EV_TX_DONE = 1,
    TBSTREAM_ZC_EV_CLOSE = 2,
};

struct tbstream_zc_event {
    uint32_t type;
    uint32_t first;
    uint32_t nframes;
    uint32_t bytes;
};

struct tbstream_zc_tx {
    uint32_t nframes;
    uint32_t last_len;
    uint32_t first;
    uint32_t reserved;
};

struct tbstream_zc_reap {
    uint32_t max;
    uint32_t flags;
    uint64_t events;
};

#define TBSTREAM_ZC_REAP_NONBLOCK 0x1u

struct tbstream_zc_rx {
    uint32_t nframes;
    uint32_t flags;
};

#define TBSTREAM_ZC_RX_F_INTERRUPT_BOUNDARIES 0x1u

#define TBSTREAM_ZC_MAGIC 0xb4
#define TBSTREAM_ZC_ENABLE _IO(TBSTREAM_ZC_MAGIC, 0x00)
#define TBSTREAM_ZC_GET_INFO \
    _IOR(TBSTREAM_ZC_MAGIC, 0x01, struct tbstream_zc_info)
#define TBSTREAM_ZC_SUBMIT_TX \
    _IOWR(TBSTREAM_ZC_MAGIC, 0x02, struct tbstream_zc_tx)
#define TBSTREAM_ZC_POST_RX \
    _IOW(TBSTREAM_ZC_MAGIC, 0x03, uint32_t)
#define TBSTREAM_ZC_REAP \
    _IOWR(TBSTREAM_ZC_MAGIC, 0x04, struct tbstream_zc_reap)
#define TBSTREAM_ZC_POST_RX_FLAGS \
    _IOW(TBSTREAM_ZC_MAGIC, 0x05, struct tbstream_zc_rx)

#endif
