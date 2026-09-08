/* Q8 MMA must not read beyond an exactly-sized raw weight allocation. */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "Q8 bounds failure at line %d\n", __LINE__); goto cleanup; \
} } while (0)

static void ones(unsigned char *w, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        const uint16_t scale = 0x3c00; /* FP16 1 */
        memcpy(w + b * 34, &scale, 2);
        memset(w + b * 34 + 2, 1, 32);
    }
}

static int check_shape(uint64_t width, uint64_t rank, uint32_t groups) {
    const uint32_t tokens = 8;
    const uint64_t low_dim = rank * groups, out_dim = 3;
    const size_t a_bytes = low_dim * (width / 32) * 34;
    const size_t b_bytes = out_dim * ((low_dim + 31) / 32) * 34;
    const size_t nx = tokens * groups * width;
    const size_t nl = tokens * low_dim, ny = tokens * out_dim;
    unsigned char *model = malloc(a_bytes + b_bytes + 4);
    float *x = malloc(nx * sizeof(float)), *values = malloc((nl + ny) * sizeof(float));
    ds4_gpu_tensor *heads = NULL, *low = NULL, *out = NULL;
    int rc = 1, initialized = 0;
    CHECK(model && x && values);
    ones(model, (a_bytes + b_bytes) / 34);
    memset(model + a_bytes + b_bytes, 0, 4);
    for (size_t i = 0; i < nx; i++) x[i] = 127.0f;
    CHECK(ds4_gpu_init()); initialized = 1;
    CHECK(ds4_gpu_set_model_map(model, a_bytes + b_bytes + 4));
    /* Cache only the exact tensor spans, not the host buffer's end padding. */
    CHECK(ds4_gpu_cache_model_range(model, a_bytes + b_bytes + 4,
                                   0, a_bytes, "Q8 bounds A"));
    CHECK(ds4_gpu_cache_model_range(model, a_bytes + b_bytes + 4,
                                   a_bytes, b_bytes, "Q8 bounds B"));
    heads = ds4_gpu_tensor_alloc(nx * sizeof(float));
    low = ds4_gpu_tensor_alloc(nl * sizeof(float));
    out = ds4_gpu_tensor_alloc(ny * sizeof(float));
    CHECK(heads && low && out);
    CHECK(ds4_gpu_tensor_write(heads, 0, x, nx * sizeof(float)));
    CHECK(ds4_gpu_attention_output_q8_batch_tensor(out, low, NULL, NULL,
        model, a_bytes + b_bytes + 4, 0, a_bytes, width, rank, groups, out_dim,
        heads, tokens));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(low, 0, values, nl * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(out, 0, values + nl, ny * sizeof(float)));
    for (size_t i = 0; i < nl; i++) CHECK(values[i] == (float)(width * 127));
    for (size_t i = 0; i < ny; i++) CHECK(values[nl + i] == (float)(low_dim * width * 127));
    rc = 0;
cleanup:
    if (rc) fprintf(stderr, "width=%llu rank=%llu groups=%u\n",
                    (unsigned long long)width, (unsigned long long)rank, groups);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(heads);
    if (initialized) ds4_gpu_cleanup();
    free(values); free(x); free(model);
    return rc;
}

int main(void) {
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 8) {
        fprintf(stderr, "SKIP: exact Q8 MMA requires an sm_80+ CUDA device\n");
        return 0;
    }
    setenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A", "1", 1);
    unsetenv("DS4_CUDA_COPY_MODEL");
    if (check_shape(32, 1, 1) || check_shape(96, 65, 3) ||
        check_shape(4096, 128, 4)) return 1;
    fprintf(stderr, "PASS: Q8 MMA exact-size weights, including odd halfword tails\n");
    return 0;
}
