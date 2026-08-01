#ifndef DS4_LAYER_PACK_H
#define DS4_LAYER_PACK_H

#include <stddef.h>
#include <stdio.h>

#define DS4_LAYER_PACK_MAX_GPUS 16
#define DS4_LAYER_PACK_CPU      (-1)

typedef struct {
    /* Per-device budget in BYTES (already net of per-device overhead such as
     * cuBLAS workspace, global scratch, logits buffer, MTP, steering, and the
     * configured safety margin). The packer treats these as raw capacity. */
    size_t gpu_budget_bytes[DS4_LAYER_PACK_MAX_GPUS];
    int    n_gpus;
} ds4_layer_pack_config;

#ifdef __cplusplus
extern "C" {
#endif

/* Compute a monotonic-contiguous layer placement.
 *
 * Inputs:
 *   entry_bytes[]     — per-entry byte footprint, in FORWARD ORDER:
 *                         entry 0           = embedding pseudo-layer
 *                         entry 1..n_layers = transformer layers
 *                         entry n_layers+1  = output-head pseudo-layer
 *   n_entries         — total number of entries (typically n_layers + 2)
 *   cfg               — per-device budgets
 *
 * Output:
 *   device_for_entry[] — filled with target device for each entry; CPU is
 *                        DS4_LAYER_PACK_CPU. Caller-owned, must be
 *                        large enough for n_entries int slots.
 *
 * Returns:
 *   0 on success.
 *   Nonzero on configuration errors (null pointer, n_entries < 0,
 *   n_gpus out of range).
 *
 * Per the design doc, an entry that exceeds every GPU budget spills to CPU;
 * by the monotonicity rule every entry after it also goes to CPU. There is
 * no error path for "entry too large" — the CPU tier is always available. */
int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *cfg,
                                int *device_for_entry);

/* Print a human-readable layout summary matching the design doc's format:
 *
 *   multi-GPU layout:
 *     GPU0: layers 0-21 + embedding   (38.4 / 40.0 GB)
 *     GPU1: layers 22-31              (11.7 / 12.0 GB)
 *     CPU : layers 32-42 + output head
 *
 * Entry-naming convention:
 *   entry 0          -> "embedding"
 *   entry i in 1..n_layers -> transformer layer (numbered i-1)
 *   entry n_layers+1 -> "output head"
 *
 * gpu_used_bytes[] and gpu_budget_bytes[] may be NULL — in that case the
 * "(used / budget)" line is omitted for the GPU lines. */
void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus);

#ifdef __cplusplus
}
#endif

#endif /* DS4_LAYER_PACK_H */
