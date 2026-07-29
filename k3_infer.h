#ifndef K3_INFER_H
#define K3_INFER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_gpu.h"
#include "k3_model.h"

typedef struct k3_context k3_context;
typedef void (*k3_hidden_hook_fn)(void *ud,
                                  const char *stream,
                                  uint32_t layer,
                                  uint32_t pos0,
                                  ds4_gpu_tensor *tensor,
                                  uint64_t n_f32);

typedef struct {
    const char *model_directory;
    uint32_t context_capacity;
    uint32_t max_prefill_tokens;
    bool layer_slice;
    uint32_t layer_start;
    uint32_t layer_end;
    bool has_output;
} k3_context_config;

int k3_context_create(k3_context **out,
                      const k3_context_config *config,
                      char *err,
                      size_t errcap);
int k3_context_clone(k3_context **out,
                     k3_context *shared,
                     uint32_t context_capacity,
                     char *err,
                     size_t errcap);
void k3_context_destroy(k3_context *ctx);
int k3_context_reset(k3_context *ctx, char *err, size_t errcap);

/*
 * Prefill one contiguous chunk. Calling this repeatedly is supported; every
 * chunk is still evaluated layer-major, so larger chunks give better shard
 * streaming efficiency. logits receives K3_N_VOCAB floats for the final row.
 */
int k3_context_prefill(k3_context *ctx,
                       const int32_t *token_ids,
                       uint32_t n_tokens,
                       float *logits,
                       char *err,
                       size_t errcap);

/*
 * Append one generated/input token and return logits for its successor.
 * Generation uses the same layer-major shard stream as prefill.
 */
int k3_context_decode(k3_context *ctx,
                      int32_t token_id,
                      float *logits,
                      char *err,
                      size_t errcap);

/*
 * Evaluate the configured contiguous layer slice. The boundary layout is the
 * complete prefix tensor, complete current-delta tensor, then the row-major
 * AttnRes block tensor. Layer zero consumes token embeddings; later slices
 * require input_boundary. output_boundary may be NULL for KV-only chunks.
 */
int k3_context_eval_slice(k3_context *ctx,
                          const int32_t *token_ids,
                          uint32_t n_tokens,
                          uint32_t pos0,
                          uint32_t layer_start,
                          uint32_t layer_end,
                          const float *input_boundary,
                          float *output_boundary,
                          bool output_logits,
                          float *logits,
                          char *err,
                          size_t errcap);

void k3_context_set_hidden_hook(k3_context *ctx,
                                k3_hidden_hook_fn fn,
                                void *ud);
uint32_t k3_context_position(const k3_context *ctx);
uint32_t k3_context_capacity(const k3_context *ctx);
uint32_t k3_context_prefill_capacity(const k3_context *ctx);

#endif
