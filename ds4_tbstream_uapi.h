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

/* Diagnostic ring/zero-copy stats (thunderbolt_stream patch >= 10) and the
 * experimental DMA-BUF import interface (patch 14). Keep layout synchronized
 * with strix-rdma/kernel/zerocopy; ds4_transport_nhi.c has compile-time size
 * asserts for the records ds4 uses. */

struct tbstream_zc_ring_stats {
    uint64_t descriptors_posted;
    uint64_t descriptors_completed;
    uint64_t interrupts;
    uint64_t work_runs;
    uint64_t kick_requests;
    uint64_t kick_pending;
    uint32_t flags;
    int32_t hop;
    int32_t irq;
    uint32_t vector;
    uint32_t size;
    uint32_t sw_head;
    uint32_t sw_tail;
    uint32_t hw_posted;
    uint32_t hw_completed;
    uint32_t queued;
    uint32_t in_flight;
    uint32_t tail_flags;
    uint32_t tail_length;
    uint32_t tail_eof;
    uint32_t tail_sof;
    uint32_t options;
    uint32_t interval_nsec;
    uint32_t reserved[2];
};

#define TBSTREAM_ZC_RING_F_TX 0x1u
#define TBSTREAM_ZC_RING_F_RUNNING 0x2u
#define TBSTREAM_ZC_RING_F_HW_VALID 0x20u

struct tbstream_zc_stats {
    uint32_t version;
    uint32_t struct_size;
    uint64_t tx_submit_calls;
    uint64_t tx_submit_frames;
    uint64_t tx_callbacks;
    uint64_t tx_terminal_callbacks;
    uint64_t tx_events;
    uint64_t tx_enqueue_errors;
    uint64_t rx_callbacks;
    uint64_t rx_data_more;
    uint64_t rx_data;
    uint64_t rx_close;
    uint64_t rx_events;
    uint64_t rx_repost_calls;
    uint64_t rx_repost_frames;
    uint64_t rx_repost_errors;
    uint64_t reap_calls;
    uint64_t reaped_events;
    uint64_t event_drops;
    uint64_t crc_errors;
    uint64_t overrun_errors;
    uint64_t canceled_callbacks;
    uint64_t failures;
    uint64_t tx_prod;
    uint64_t tx_cons;
    uint64_t rx_prod;
    uint64_t rx_cons;
    uint32_t tx_pending;
    uint32_t tx_done;
    uint32_t rx_partial_frames;
    uint32_t rx_partial_bytes;
    uint32_t fifo_len;
    uint32_t fifo_avail;
    uint32_t flags;
    int32_t in_hopid;
    int32_t out_hopid;
    uint32_t throttling;
    uint32_t last_error;
    uint32_t reserved[4];
    struct tbstream_zc_ring_stats tx;
    struct tbstream_zc_ring_stats rx;
};

#define TBSTREAM_ZC_STATS_VERSION 1u
#define TBSTREAM_ZC_STATS_F_FAILED 0x1u
#define TBSTREAM_ZC_STATS_F_CLOSED 0x2u
#define TBSTREAM_ZC_STATS_F_REMOVED 0x4u
#define TBSTREAM_ZC_STATS_F_TX_IMPORTED 0x8u
#define TBSTREAM_ZC_STATS_F_RX_IMPORTED 0x10u

/* One directional pool for TBSTREAM_ZC_IMPORT. fd of -1 keeps the kernel
 * page-backed pool for that direction (offset/length/flags must be 0). */
struct tbstream_zc_import_range {
    int32_t fd;
    uint32_t flags;
    uint64_t offset;
    uint64_t length;
};

#define TBSTREAM_ZC_IMPORT_VERSION 1u

/* Select DMA-BUF backed frame pools for the zero-copy session that a
 * following TBSTREAM_ZC_ENABLE starts. Must run on an exclusively opened,
 * never-activated stream; imported halves have no CPU mapping (mmap leaves
 * holes) and legacy read/write are rejected while an import is configured.
 * Requires CAP_SYS_RAWIO and thunderbolt_stream.zc_diagnostic_dmabuf=1. */
struct tbstream_zc_import {
    uint32_t version;
    uint32_t flags;
    struct tbstream_zc_import_range tx;
    struct tbstream_zc_import_range rx;
    uint64_t reserved[4];
};

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
#define TBSTREAM_ZC_GET_STATS \
    _IOR(TBSTREAM_ZC_MAGIC, 0x06, struct tbstream_zc_stats)
#define TBSTREAM_ZC_IMPORT \
    _IOW(TBSTREAM_ZC_MAGIC, 0x09, struct tbstream_zc_import)

#endif
