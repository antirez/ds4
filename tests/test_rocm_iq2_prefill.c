/* Exercise the production resident IQ2/IQ2 hot-list dispatch. Zero gate/up
 * weights make every intermediate and output exactly zero. Poisoned F32 mid
 * storage exposes a hot WMMA producer writing only to the F16 scratch alias.
 * Small dimensions keep this independent of models and large allocations. */
#include "ds4_gpu.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { DIM = 256, EXPERTS = 8, USED = 6, IQ2 = 16, GUARD = 16 };
typedef struct { uint16_t d, qs[32]; } iq2_block;
typedef struct {
    ds4_gpu_tensor *storage, *view;
    float *host;
    uint64_t count;
} guarded_tensor;

static void release(guarded_tensor *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->storage);
    free(t->host);
}

static int allocate(guarded_tensor *t, uint64_t count, float value) {
    t->count = count;
    const uint64_t bytes = (count + 2u * GUARD) * sizeof(float);
    t->host = (float *)malloc((size_t)bytes);
    t->storage = ds4_gpu_tensor_alloc(bytes);
    if (!t->host || !t->storage) return 0;
    t->view = ds4_gpu_tensor_view(t->storage, GUARD * sizeof(float),
                                 count * sizeof(float));
    if (!t->view) return 0;
    for (uint64_t i = 0; i < count + 2u * GUARD; i++) {
        t->host[i] = i < GUARD || i >= count + GUARD ? 123456.0f : value;
    }
    return ds4_gpu_tensor_write(t->storage, 0, t->host, bytes);
}

static int check(guarded_tensor *t, bool zero, const char *name, uint32_t n) {
    if (!ds4_gpu_tensor_read(t->storage, 0, t->host,
                             (t->count + 2u * GUARD) * sizeof(float))) return 0;
    for (uint64_t i = 0; i < t->count + 2u * GUARD; i++) {
        const bool canary = i < GUARD || i >= t->count + GUARD;
        if (!canary && !zero) continue;
        const float expected = canary ? 123456.0f : 0.0f;
        if (t->host[i] != expected) {
            fprintf(stderr, "IQ2/IQ2 N=%u %s[%llu]=%g expected %g\n",
                    n, name, (unsigned long long)i, t->host[i], expected);
            return 0;
        }
    }
    return 1;
}

static int run(uint32_t n, bool mixed, const void *model, uint64_t model_size,
               uint64_t matrix_bytes) {
    const uint64_t pairs = (uint64_t)n * USED;
    guarded_tensor x = {0}, gate = {0}, up = {0}, mid = {0}, down = {0}, out = {0};
    ds4_gpu_tensor *ids_gpu = NULL, *weights_gpu = NULL;
    int32_t *ids = (int32_t *)malloc((size_t)pairs * sizeof(int32_t));
    float *weights = (float *)malloc((size_t)pairs * sizeof(float));
    int ok = ids && weights;
    if (ok) {
        for (uint32_t t = 0; t < n; t++) {
            for (uint32_t s = 0; s < USED; s++) {
                ids[t * USED + s] = (int32_t)(mixed && s == USED - 1u && t >= n / 2u ? 7u : s);
                weights[t * USED + s] = 0.25f;
            }
        }
        ids_gpu = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
        weights_gpu = ds4_gpu_tensor_alloc(pairs * sizeof(float));
        ok = ids_gpu && weights_gpu &&
             ds4_gpu_tensor_write(ids_gpu, 0, ids, pairs * sizeof(int32_t)) &&
             ds4_gpu_tensor_write(weights_gpu, 0, weights, pairs * sizeof(float));
    }
    ok = ok && allocate(&x, (uint64_t)n * DIM, 0.25f) &&
         allocate(&gate, pairs * DIM, 73.0f) && allocate(&up, pairs * DIM, 73.0f) &&
         allocate(&mid, pairs * DIM, 73.0f) && allocate(&down, pairs * DIM, 73.0f) &&
         allocate(&out, (uint64_t)n * DIM, 73.0f);
    bool mid_is_f16 = true;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(
            out.view, gate.view, up.view, mid.view, down.view,
            model, model_size, 0, matrix_bytes, 2u * matrix_bytes,
            IQ2, IQ2, DIM * sizeof(iq2_block), sizeof(iq2_block),
            DIM * sizeof(iq2_block), sizeof(iq2_block), DIM, DIM, DIM,
            ids_gpu, weights_gpu, EXPERTS, USED, 7.0f, x.view, 0, n,
            &mid_is_f16, true);
    }
    if (ok && mid_is_f16) {
        fprintf(stderr, "IQ2/IQ2 unexpectedly advertised F16 mid\n");
        ok = 0;
    }
    ok = ok && check(&mid, true, "mid", n) && check(&out, true, "out", n) &&
         check(&x, false, "x", n) && check(&gate, false, "gate", n) &&
         check(&up, false, "up", n) && check(&down, false, "down", n);
    release(&out); release(&down); release(&mid); release(&up); release(&gate); release(&x);
    ds4_gpu_tensor_free(ids_gpu);
    ds4_gpu_tensor_free(weights_gpu);
    free(ids); free(weights);
    fprintf(stderr, "IQ2/IQ2 N=%u routing=%s: %s\n", n,
            mixed ? "mixed" : "same", ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    if (sizeof(iq2_block) != 66u) return 1;
    const uint64_t matrix_bytes = EXPERTS * DIM * sizeof(iq2_block);
    const uint64_t model_size = 3u * matrix_bytes;
    FILE *file = tmpfile();
    void *model = MAP_FAILED;
    if (file && ftruncate(fileno(file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(file), 0);
    }
    if (model == MAP_FAILED) {
        if (file) fclose(file);
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    /* Grid0/sign0 has positive values. Nonzero down weights ensure stale
     * nonzero mid storage cannot be hidden behind a zero output matrix. */
    iq2_block *down = (iq2_block *)((char *)model + 2u * matrix_bytes);
    for (uint32_t i = 0; i < EXPERTS * DIM; i++) down[i].d = 0x1c00u;

    int ok = ds4_gpu_init();
    if (!ok) {
        munmap(model, (size_t)model_size);
        fclose(file);
        return 77;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    /* Exercise the normal hot-list route regardless of diagnostic shell
     * settings; quality mode and the existing rollback are checked below. */
    unsetenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED");
    unsetenv("DS4_ROCM_IQ2_MOE_WMMA_PROFILE");
    ok = ds4_gpu_set_model_map(model, model_size) &&
         ds4_gpu_set_model_fd(fileno(file)) &&
         ds4_gpu_cache_model_range(model, model_size, 0, model_size, "iq2_regression");
    const uint32_t sizes[] = {7, 8, 9, 15, 16, 128};
    for (uint32_t i = 0; ok && i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        ok = run(sizes[i], false, model, model_size, matrix_bytes) &&
             run(sizes[i], true, model, model_size, matrix_bytes);
    }
    if (ok) {
        setenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED", "1", 1);
        ok = run(16, true, model, model_size, matrix_bytes);
        unsetenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED");
    }
    if (ok) {
        ds4_gpu_set_quality(true);
        ok = run(16, true, model, model_size, matrix_bytes);
    }
    ds4_gpu_set_model_fd(-1);
    ds4_gpu_cleanup();
    munmap(model, (size_t)model_size);
    fclose(file);
    return ok ? 0 : 1;
}
