#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_HC = 4,
    GUARD_FLOATS = 11,
};

static const uint32_t guard_bits = 0x7fc12345u;

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void fill_guard(uint32_t *dst, size_t count) {
    for (size_t i = 0; i < count; i++) dst[i] = guard_bits;
}

static int guard_is_intact(const char *name,
                           const uint32_t *storage,
                           size_t payload_floats) {
    for (size_t i = 0; i < GUARD_FLOATS; i++) {
        if (storage[i] != guard_bits) {
            fprintf(stderr, "%s prefix guard changed at %zu: 0x%08x\n",
                    name, i, storage[i]);
            return 0;
        }
    }
    for (size_t i = 0; i < GUARD_FLOATS; i++) {
        const size_t at = GUARD_FLOATS + payload_floats + i;
        if (storage[at] != guard_bits) {
            fprintf(stderr, "%s suffix guard changed at %zu: 0x%08x\n",
                    name, i, storage[at]);
            return 0;
        }
    }
    return 1;
}

static ds4_gpu_tensor *make_guarded_tensor(size_t payload_floats,
                                            uint32_t **host_storage) {
    const size_t total = payload_floats + 2u * GUARD_FLOATS;
    uint32_t *storage = malloc(total * sizeof(*storage));
    if (!storage) return NULL;
    fill_guard(storage, total);

    ds4_gpu_tensor *base = ds4_gpu_tensor_alloc(total * sizeof(float));
    if (!base || !ds4_gpu_tensor_write(base, 0, storage,
                                       total * sizeof(*storage))) {
        ds4_gpu_tensor_free(base);
        free(storage);
        return NULL;
    }
    *host_storage = storage;
    return base;
}

static int read_guarded_tensor(ds4_gpu_tensor *base,
                               uint32_t *storage,
                               size_t payload_floats) {
    const size_t total = payload_floats + 2u * GUARD_FLOATS;
    return ds4_gpu_tensor_read(base, 0, storage,
                               total * sizeof(*storage));
}

static int run_case(uint32_t rows, uint32_t n_embd) {
    const size_t x_floats = (size_t)rows * TEST_HC * n_embd;
    const size_t weights_floats = (size_t)rows * TEST_HC;
    const size_t out_floats = (size_t)rows * n_embd;
    const size_t last_floats = n_embd;
    uint32_t *x_storage = NULL;
    uint32_t *weights_storage = NULL;
    uint32_t *ref_storage = NULL;
    uint32_t *fused_storage = NULL;
    uint32_t *last_storage = NULL;
    ds4_gpu_tensor *x_base = make_guarded_tensor(x_floats, &x_storage);
    ds4_gpu_tensor *weights_base =
        make_guarded_tensor(weights_floats, &weights_storage);
    ds4_gpu_tensor *ref_base = make_guarded_tensor(out_floats, &ref_storage);
    ds4_gpu_tensor *fused_base =
        make_guarded_tensor(out_floats, &fused_storage);
    ds4_gpu_tensor *last_base =
        make_guarded_tensor(last_floats, &last_storage);
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *weights = NULL;
    ds4_gpu_tensor *ref = NULL;
    ds4_gpu_tensor *fused = NULL;
    ds4_gpu_tensor *last = NULL;
    int ok = x_base && weights_base && ref_base && fused_base && last_base;

    if (ok) {
        float *x_values = (float *)(x_storage + GUARD_FLOATS);
        float *weight_values = (float *)(weights_storage + GUARD_FLOATS);
        for (size_t i = 0; i < x_floats; i++) {
            const int32_t centered = (int32_t)((i * 37u + 13u) % 257u) - 128;
            x_values[i] = (float)centered / 32.0f;
        }
        for (size_t i = 0; i < weights_floats; i++) {
            weight_values[i] = 0.25f;
        }
        ok = ds4_gpu_tensor_write(x_base, 0, x_storage,
                                  (x_floats + 2u * GUARD_FLOATS) * sizeof(float)) &&
             ds4_gpu_tensor_write(weights_base, 0, weights_storage,
                                  (weights_floats + 2u * GUARD_FLOATS) * sizeof(float));
    }
    if (ok) {
        x = ds4_gpu_tensor_view(x_base,
                (uint64_t)GUARD_FLOATS * sizeof(float),
                (uint64_t)x_floats * sizeof(float));
        weights = ds4_gpu_tensor_view(weights_base,
                (uint64_t)GUARD_FLOATS * sizeof(float),
                (uint64_t)weights_floats * sizeof(float));
        ref = ds4_gpu_tensor_view(ref_base,
                (uint64_t)GUARD_FLOATS * sizeof(float),
                (uint64_t)out_floats * sizeof(float));
        fused = ds4_gpu_tensor_view(fused_base,
                (uint64_t)GUARD_FLOATS * sizeof(float),
                (uint64_t)out_floats * sizeof(float));
        last = ds4_gpu_tensor_view(last_base,
                (uint64_t)GUARD_FLOATS * sizeof(float),
                (uint64_t)last_floats * sizeof(float));
        ok = x && weights && ref && fused && last;
    }
    if (ok) {
        ok = ds4_gpu_hc_weighted_sum_tensor(ref, x, weights,
                                             n_embd, TEST_HC) &&
             ds4_gpu_hc_weighted_sum_capture_last_tensor(fused, last,
                                                          x, weights,
                                                          n_embd, TEST_HC) &&
             ds4_gpu_synchronize();
    }
    if (ok) {
        ok = read_guarded_tensor(ref_base, ref_storage, out_floats) &&
             read_guarded_tensor(fused_base, fused_storage, out_floats) &&
             read_guarded_tensor(last_base, last_storage, last_floats) &&
             read_guarded_tensor(x_base, x_storage, x_floats) &&
             read_guarded_tensor(weights_base, weights_storage, weights_floats);
    }
    if (ok && memcmp(ref_storage + GUARD_FLOATS,
                     fused_storage + GUARD_FLOATS,
                     out_floats * sizeof(float)) != 0) {
        fprintf(stderr, "DSpark capture batch mismatch rows=%u embd=%u\n",
                rows, n_embd);
        ok = 0;
    }
    if (ok && memcmp(ref_storage + GUARD_FLOATS +
                         (size_t)(rows - 1u) * n_embd,
                     last_storage + GUARD_FLOATS,
                     last_floats * sizeof(float)) != 0) {
        fprintf(stderr, "DSpark capture last-row mismatch rows=%u embd=%u\n",
                rows, n_embd);
        ok = 0;
    }
    if (ok) {
        ok = guard_is_intact("input", x_storage, x_floats) &&
             guard_is_intact("weights", weights_storage, weights_floats) &&
             guard_is_intact("reference", ref_storage, out_floats) &&
             guard_is_intact("fused batch", fused_storage, out_floats) &&
             guard_is_intact("fused last", last_storage, last_floats);
    }

    ds4_gpu_tensor_free(last);
    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(last_base);
    ds4_gpu_tensor_free(fused_base);
    ds4_gpu_tensor_free(ref_base);
    ds4_gpu_tensor_free(weights_base);
    ds4_gpu_tensor_free(x_base);
    free(last_storage);
    free(fused_storage);
    free(ref_storage);
    free(weights_storage);
    free(x_storage);

    if (ok) {
        fprintf(stderr, "DSpark Metal capture rows=%u embd=%u bitwise PASS\n",
                rows, n_embd);
    }
    return ok;
}

int main(void) {
    int ok = ds4_gpu_init();
    const uint32_t rows[] = {1u, 2u, 5u};
    const uint32_t embd[] = {17u, 4096u};
    for (size_t d = 0; ok && d < sizeof(embd) / sizeof(embd[0]); d++) {
        for (size_t r = 0; ok && r < sizeof(rows) / sizeof(rows[0]); r++) {
            ok = run_case(rows[r], embd[d]);
        }
    }
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
