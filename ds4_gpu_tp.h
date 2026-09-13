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
 * synchronously. Callbacks return nonzero on success; both ranks must issue
 * the same gate sequence. Shutdown precedes transport/slab destruction. */
typedef int (*ds4_gpu_tp_exchange_fn)(void *ud, uint32_t layer, uint32_t gate, uint64_t seq);
typedef int (*ds4_gpu_tp_batch_exchange_fn)(void *ud, uint32_t layer,
                                          uint32_t rows, uint64_t seq);
typedef int (*ds4_gpu_tp_big_exchange_fn)(void *ud, uint32_t layer,
                                        uint64_t seq, const void *out,
                                        void *in, uint64_t bytes);

int ds4_gpu_tp_init(uint32_t rank, ds4_gpu_tensor *slab,
                    uint64_t gpu_flags_off, uint64_t out_off, uint64_t vec_bytes,
                    ds4_gpu_tp_exchange_fn fn, void *ud);
void ds4_gpu_tp_shutdown(void);
int ds4_gpu_tp_failed(void);
int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate);
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
