#ifndef DS4_GPU_TP_H
#define DS4_GPU_TP_H

#include <stdint.h>
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Metal queues each gate on its service thread and orders GPU arrival/release
 * with events or flags. CUDA waits for its local stream and calls the exchange
 * synchronously. ROCm/NHI copies fixed graph views into rotating UC slots and
 * eagerly waits on the peer's in-band stamp. Callbacks return nonzero on
 * success; both ranks must issue the same gate sequence. Shutdown precedes
 * transport/slab destruction. */
typedef int (*ds4_gpu_tp_exchange_fn)(void *ud, uint32_t layer, uint32_t gate, uint64_t seq);
typedef int (*ds4_gpu_tp_batch_exchange_fn)(void *ud, uint32_t layer,
                                          uint32_t rows, uint64_t seq);
typedef int (*ds4_gpu_tp_big_exchange_fn)(void *ud, uint32_t layer,
                                        uint64_t seq, const void *out,
                                        void *in, uint64_t bytes);

int ds4_gpu_tp_init(uint32_t rank, ds4_gpu_tensor *slab,
                    uint64_t gpu_flags_off, uint64_t out_off, uint64_t vec_bytes,
                    ds4_gpu_tp_exchange_fn fn, void *ud);
/* ROCm/NHI gate service.  Graph partials remain in fixed slab views while
 * each gate copies to/from the globally rotating transport slots.  The
 * service thread waits the TX-ready event before submit and calls consumed
 * only after the RX wait-copy's final-reader event. */
typedef void *(*ds4_gpu_tp_nhi_tx_slot_fn)(void *ud, uint64_t seq);
typedef const void *(*ds4_gpu_tp_nhi_rx_slot_fn)(void *ud, uint64_t seq);
typedef int (*ds4_gpu_tp_nhi_seq_fn)(void *ud, uint64_t seq);
typedef void (*ds4_gpu_tp_nhi_fail_fn)(void *ud);
int ds4_gpu_tp_nhi_init(uint32_t rank,
                        ds4_gpu_tensor *slab,
                        uint64_t out_offset,
                        uint64_t in_offset,
                        uint32_t n_slots,
                        uint32_t n_embd,
                        ds4_gpu_tp_nhi_tx_slot_fn tx_slot_fn,
                        ds4_gpu_tp_nhi_rx_slot_fn rx_slot_fn,
                        ds4_gpu_tp_nhi_seq_fn acquire_tx_fn,
                        ds4_gpu_tp_nhi_seq_fn submit_fn,
                        ds4_gpu_tp_nhi_seq_fn consumed_fn,
                        ds4_gpu_tp_nhi_fail_fn fail_fn,
                        void *ud);
/* Set the transport ring capacity in messages; bounds the big-gate
 * pipelining window so a ring slot is never refilled before its prior
 * message was consumed.  Call after ds4_gpu_tp_nhi_init. */
void ds4_gpu_tp_nhi_set_ring_msgs(uint32_t msgs);
void ds4_gpu_tp_shutdown(void);
int ds4_gpu_tp_failed(void);
int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate);
/* Stage-1 loopback tests for the ROCm TP combine kernels
 * (tests/test_tp_combine_rocm): bit-exact world-of-two combine, and
 * spin-combine gated by an in-band stamp written by a second-stream
 * kernel or the host. */
int ds4_gpu_tp_test_combine(uint32_t n, uint32_t iterations);
int ds4_gpu_tp_test_spin_exchange(int uncached_pool, int host_stamp,
                                  uint32_t n, uint32_t seqs);
/* Test-only ownership selector: rank 0/1 enables its contiguous expert half;
 * any other value restores the unsharded production default. */
void ds4_gpu_tp_test_set_expert_shard(int rank);
/* Stage-2 GPU helpers for the NHI TP transport (ds4_tp_nhi.c): uncached
 * dedicated pool allocation with DMA-BUF export, and stream-based
 * fill-and-release / spin-combine wrappers for the exchange loop. */
int ds4_gpu_tp_pool_alloc_export_uc(uint64_t bytes, void **dev_ptr,
                                    int *dmabuf_fd);
int ds4_gpu_tp_dev_buf_create(const float *init_host, uint32_t n,
                              void **dev_ptr);
int ds4_gpu_tp_dev_buf_read(const void *dev_ptr, float *out_host, uint32_t n);
void ds4_gpu_tp_dev_buf_free(void *dev_ptr);
int ds4_gpu_tp_fill_release(void *slot_dev, const float *src_host,
                            uint32_t n, uint32_t stamp);
int ds4_gpu_tp_spin_combine_start(void *acc_dev, const void *slot_dev,
                                  uint32_t n, uint32_t expect_stamp,
                                  unsigned long long max_spins);
int ds4_gpu_tp_spin_combine_wait(int *timed_out);
void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn);
int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows);
void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn);
int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                              const ds4_gpu_tensor *out_t,
                              ds4_gpu_tensor *in_t, uint64_t bytes);

/* Metal multi-session tapes reuse slab slots and therefore require event
 * arrival. Single-session flag gates may flush in layer order. */
void ds4_gpu_tp_set_session_batch_mode(int enabled);
int ds4_gpu_tp_decode_split_flush_safe(void);
/* Weight ranges consumed by the next Metal poll gate of the given kind. */
int ds4_gpu_tp_gate_prefetch_plan(uint32_t gate,
                                const void *model_map, uint64_t model_size,
                                const uint64_t *offsets, const uint64_t *bytes,
                                uint32_t count);
/* Coordinator-only drafting is replicated; verification remains sharded. */
void ds4_gpu_tp_suspend_expert_sharding(int suspend);
/* No-op when TP is not bound. */
void ds4_gpu_tp_keepalive_pause(int paused);
/* GLM attention head ownership; the caller zeros unowned heads and exchanges
 * the output-projection partials at the big gate. */
void ds4_gpu_tp_set_attn_head_split(int enabled);

#ifdef __cplusplus
}
#endif
#endif
