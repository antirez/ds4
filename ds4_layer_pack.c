/* Monotonic-contiguous multi-tier layer placement packer.
 *
 * Pure C99: no CUDA, no platform-specific code. Used by both the CUDA and
 * Metal/CPU builds. See ds4_layer_pack.h for the API contract and the
 * design doc docs/superpowers/specs/2026-05-26-multi-gpu-pp-v0-design.md
 * for the rationale. */

#include "ds4_layer_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *cfg,
                                int *device_for_entry) {
    if (!entry_bytes || !cfg || !device_for_entry) return 1;
    if (n_entries < 0) return 2;
    if (cfg->n_gpus < 0 || cfg->n_gpus > DS4_LAYER_PACK_MAX_GPUS) return 3;

    /* Work over a local copy of budgets so the caller's struct is untouched. */
    size_t budget[DS4_LAYER_PACK_MAX_GPUS];
    for (int d = 0; d < cfg->n_gpus; d++) {
        budget[d] = cfg->gpu_budget_bytes[d];
    }

    int d = 0;
    for (int e = 0; e < n_entries; e++) {
        /* Advance until the current device can hold this entry, or we run
         * out of devices. Strict greater-than: exact fits stay on d. */
        while (d < cfg->n_gpus && entry_bytes[e] > budget[d]) {
            d++;
        }
        if (d < cfg->n_gpus) {
            device_for_entry[e] = d;
            budget[d] -= entry_bytes[e];
        } else {
            device_for_entry[e] = DS4_LAYER_PACK_CPU;
            /* d stays at cfg->n_gpus so every subsequent entry also lands
             * on the CPU tier. */
        }
    }
    return 0;
}

/* Internal helper: append " + embedding" or " + output head" to a buffer if
 * the matching pseudo-layer lives on this tier. */
static void append_pseudo_layer_tags(char *buf, size_t buflen,
                                      const int *device_for_entry,
                                      int tier,
                                      int n_entries,
                                      int n_layers) {
    /* entry 0 -> embedding, entry n_layers+1 -> output head */
    if (n_entries > 0 && device_for_entry[0] == tier) {
        strncat(buf, " + embedding", buflen - strlen(buf) - 1);
    }
    if (n_entries > 1 && n_layers + 1 < n_entries &&
        device_for_entry[n_layers + 1] == tier) {
        strncat(buf, " + output head", buflen - strlen(buf) - 1);
    }
}

/* Emit "L-R" for a contiguous transformer-layer range assigned to a tier.
 * Returns 1 if anything was printed (i.e. tier owned at least one layer). */
static int print_layer_range_for_tier(FILE *out, const int *device_for_entry,
                                      int tier, int n_entries, int n_layers) {
    /* Transformer layers live at indices 1..n_layers in device_for_entry.
     * Their human-facing numbers are 0..n_layers-1 (i.e. entry index - 1). */
    int first = -1;
    int last  = -1;
    for (int i = 1; i <= n_layers && i < n_entries; i++) {
        if (device_for_entry[i] == tier) {
            if (first < 0) first = i - 1;  /* human-facing index */
            last = i - 1;
        }
    }
    if (first < 0) {
        fprintf(out, "(no transformer layers)");
        return 0;
    }
    if (first == last) {
        fprintf(out, "layer %d", first);
    } else {
        fprintf(out, "layers %d-%d", first, last);
    }
    return 1;
}

void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus) {
    if (!out || !device_for_entry) return;
    (void)entry_bytes; /* not needed for the textual layout */

    fprintf(out, "multi-GPU layout:\n");
    for (int d = 0; d < n_gpus; d++) {
        fprintf(out, "  GPU%d: ", d);
        int have_layers = print_layer_range_for_tier(out, device_for_entry,
                                                     d, n_entries, n_layers);
        /* Pseudo-layer tags. */
        char tags[64];
        tags[0] = '\0';
        append_pseudo_layer_tags(tags, sizeof(tags), device_for_entry, d,
                                  n_entries, n_layers);
        if (tags[0]) {
            if (!have_layers) {
                /* Erase the "(no transformer layers)" stub and just write
                 * the tag with leading "+ " stripped. */
                /* The stub was already written; appending " + embedding"
                 * is still informative. */
            }
            fprintf(out, "%s", tags);
        }
        /* Usage line. */
        if (gpu_used_bytes && gpu_budget_bytes) {
            const double used_gb   = (double)gpu_used_bytes[d]   / 1073741824.0;
            const double budget_gb = (double)gpu_budget_bytes[d] / 1073741824.0;
            /* Two-space pad before the parens matches the design doc. */
            fprintf(out, "  (%.1f / %.1f GB)", used_gb, budget_gb);
        }
        fputc('\n', out);
    }
    /* CPU tier: only print if at least one entry is on CPU. */
    int cpu_present = 0;
    for (int i = 0; i < n_entries; i++) {
        if (device_for_entry[i] == DS4_LAYER_PACK_CPU) { cpu_present = 1; break; }
    }
    if (cpu_present) {
        fprintf(out, "  CPU : ");
        int have_layers = print_layer_range_for_tier(out, device_for_entry,
                                                      DS4_LAYER_PACK_CPU,
                                                      n_entries, n_layers);
        char tags[64];
        tags[0] = '\0';
        append_pseudo_layer_tags(tags, sizeof(tags), device_for_entry,
                                  DS4_LAYER_PACK_CPU, n_entries, n_layers);
        if (tags[0]) {
            (void)have_layers;
            fprintf(out, "%s", tags);
        }
        fputc('\n', out);
    }
}
