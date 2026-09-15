/* NHI transport core for ROCm tensor parallelism: uncached GPU frame pools
 * imported into the thunderbolt-stream zero-copy rings, slot arithmetic,
 * submission, event reaping and ordered teardown.  Linux-only; see
 * ds4_tp_nhi.c for the contract rules each entry follows. */

#ifndef DS4_TP_NHI_H
#define DS4_TP_NHI_H

#ifdef __linux__

#include <stddef.h>
#include <stdint.h>

typedef struct ds4_tp_nhi ds4_tp_nhi;

int ds4_tp_nhi_open(ds4_tp_nhi **out,
                    const char *device_path,
                    uint32_t ring_frames,
                    char *err,
                    size_t errlen);
uint32_t ds4_tp_nhi_msg_frames(const ds4_tp_nhi *t);
uint32_t ds4_tp_nhi_msgs(const ds4_tp_nhi *t);
void *ds4_tp_nhi_tx_slot(ds4_tp_nhi *t, uint64_t seq);
const void *ds4_tp_nhi_rx_slot(ds4_tp_nhi *t, uint64_t seq);
/* Reserve seq%msgs after seq-msgs TX_DONE, before any GPU overwrite. */
int ds4_tp_nhi_acquire_tx(ds4_tp_nhi *t, uint64_t seq,
                          char *err, size_t errlen);
int ds4_tp_nhi_submit(ds4_tp_nhi *t, uint64_t seq, char *err, size_t errlen);
int ds4_tp_nhi_reap(ds4_tp_nhi *t, char *err, size_t errlen);
/* Mark one receive message GPU-consumed, in strict sequence order.  Reap
 * batches POST_RX only after both the RX event and this mark exist. */
int ds4_tp_nhi_consumed(ds4_tp_nhi *t, uint64_t seq,
                        char *err, size_t errlen);
int ds4_tp_nhi_peer_closed(const ds4_tp_nhi *t);
/* After GPU producers/readers are quiesced, drain TX_DONE and all eligible
 * RX reposts before ordered device close. */
int ds4_tp_nhi_quiesce(ds4_tp_nhi *t, char *err, size_t errlen);
void ds4_tp_nhi_close(ds4_tp_nhi *t);

#endif /* __linux__ */

#endif /* DS4_TP_NHI_H */
