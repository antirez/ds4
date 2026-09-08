/* Model-free checks of GB10 Q8 attention-output prefill through the public API.
 * Compare each optimization independently and together against the old path,
 * with and without the startup artifact. No external GGUF is required. */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "Q8 prefill failure at line %d\n", __LINE__); \
    goto cleanup; \
} } while (0)

static void put32(unsigned char *p, size_t *off, uint32_t x) {
    memcpy(p + *off, &x, 4); *off += 4;
}

static void put64(unsigned char *p, size_t *off, uint64_t x) {
    memcpy(p + *off, &x, 8); *off += 8;
}

static void tensor_info(unsigned char *p, size_t *off, const char *name,
                        uint64_t cols, uint64_t rows, uint64_t offset) {
    const size_t len = strlen(name);
    put64(p, off, len);
    memcpy(p + *off, name, len); *off += len;
    put32(p, off, 2);
    put64(p, off, cols); put64(p, off, rows);
    put32(p, off, 8); /* GGML_TYPE_Q8_0 */
    put64(p, off, offset);
}

static void pack_weights(unsigned char *p, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        /* Finite, nonuniform FP16 scales and signed Q8 codes. */
        const uint16_t scale = (uint16_t)(0x2000u + (b * 19u % 4096u));
        memcpy(p + b * 34u, &scale, 2);
        for (unsigned j = 0; j < 32; j++)
            p[b * 34u + 2u + j] = (unsigned char)(b * 23u + j * 41u);
    }
}

static int check_shape(uint64_t width, uint64_t rank, uint32_t groups,
                       uint32_t tokens, int artifact) {
    const uint64_t low_dim = rank * groups, out_dim = 37;
    const uint64_t a_bytes = low_dim * ((width + 31) / 32) * 34;
    const uint64_t b_bytes = out_dim * ((low_dim + 31) / 32) * 34;
    const size_t nx = (size_t)tokens * groups * width;
    const size_t nl = (size_t)tokens * low_dim, ny = (size_t)tokens * out_dim;
    unsigned char header[512] = {0};
    size_t pos = 0;
    put32(header, &pos, 0x46554747); put32(header, &pos, 3);
    put64(header, &pos, 2); put64(header, &pos, 0);
    tensor_info(header, &pos, "blk.0.attn_output_a.weight", width, low_dim, 0);
    tensor_info(header, &pos, "blk.0.attn_output_b.weight", low_dim, out_dim, a_bytes);
    const size_t a_off = (pos + 31u) & ~(size_t)31u;
    const size_t size = a_off + a_bytes + b_bytes + 4; /* Raw unaligned-load padding. */
    unsigned char *model = calloc(1, size);
    float *x = malloc(nx * sizeof(float));
    float *got = malloc((nl + ny + 8) * sizeof(float));
    float *ref = malloc((nl + ny + 8) * sizeof(float));
    ds4_gpu_tensor *heads = NULL, *low = NULL, *out = NULL;
    char path[] = "/tmp/ds4-q8-prefill-XXXXXX";
    FILE *file = NULL;
    int fd = -1, path_created = 0, initialized = 0, rc = 1;
    CHECK(model && x && got && ref);
    memcpy(model, header, pos);
    pack_weights(model + a_off, a_bytes / 34);
    pack_weights(model + a_off + a_bytes, b_bytes / 34);
    for (size_t i = 0; i < nx; i++)
        x[i] = (float)((int)(i * 17u % 257u) - 128) * 0.015625f;
    /* A zero block, signed zero, and nonuniform final partial blocks. */
    for (size_t i = 0; i < nx && i < 32; i++) x[i] = i & 1 ? -0.0f : 0.0f;
    CHECK(ds4_gpu_init());
    initialized = 1;
    CHECK(ds4_gpu_set_model_map(model, size));
    if (artifact) {
        fd = mkstemp(path);
        CHECK(fd >= 0);
        path_created = 1;
        file = fdopen(fd, "wb");
        CHECK(file);
        fd = -1;
        CHECK(fwrite(model, 1, size, file) == size);
        const int closed = fclose(file);
        file = NULL;
        CHECK(closed == 0);
        CHECK(ds4_gpu_build_derived_artifacts(model, size, path) == 1);
    }
    heads = ds4_gpu_tensor_alloc(nx * sizeof(float));
    low = ds4_gpu_tensor_alloc((nl + 4) * sizeof(float));
    out = ds4_gpu_tensor_alloc((ny + 4) * sizeof(float));
    CHECK(heads && low && out);
    CHECK(ds4_gpu_tensor_write(heads, 0, x, nx * sizeof(float)));
    for (unsigned variant = 0; variant < 8; variant++) {
        fprintf(stderr, "Q8 fixture width=%llu rank=%llu groups=%u tokens=%u artifact=%d variant=%u\n",
                (unsigned long long)width, (unsigned long long)rank, groups, tokens, artifact, variant);
        if (variant & 1) unsetenv("DS4_CUDA_NO_Q8_0_QUANT_WARPS");
        else setenv("DS4_CUDA_NO_Q8_0_QUANT_WARPS", "1", 1);
        if (variant & 2) unsetenv("DS4_CUDA_NO_Q8_MMA_ALIGNED");
        else setenv("DS4_CUDA_NO_Q8_MMA_ALIGNED", "1", 1);
        if (variant & 4) unsetenv("DS4_CUDA_NO_Q8_MMA_SCALE_PADDING");
        else setenv("DS4_CUDA_NO_Q8_MMA_SCALE_PADDING", "1", 1);
        for (size_t i = 0; i < nl + ny + 8; i++) got[i] = 123456.0f;
        CHECK(ds4_gpu_tensor_write(low, 0, got, (nl + 4) * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(out, 0, got + nl + 4, (ny + 4) * sizeof(float)));
        CHECK(ds4_gpu_attention_output_q8_batch_tensor(out, low, NULL, NULL,
            model, size, a_off, a_off + a_bytes, width, rank, groups, out_dim,
            heads, tokens));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(low, 0, got, (nl + 4) * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(out, 0, got + nl + 4, (ny + 4) * sizeof(float)));
        for (size_t i = 0; i < nl + ny + 8; i++) CHECK(isfinite(got[i]));
        for (size_t i = 0; i < 4; i++) {
            CHECK(got[nl + i] == 123456.0f);
            CHECK(got[nl + 4 + ny + i] == 123456.0f);
        }
        if (!variant) memcpy(ref, got, (nl + ny + 8) * sizeof(float));
        else CHECK(memcmp(ref, got, (nl + ny + 8) * sizeof(float)) == 0);
    }
    rc = 0;
cleanup:
    if (rc) fprintf(stderr, "shape width=%llu rank=%llu groups=%u tokens=%u artifact=%d\n",
                    (unsigned long long)width, (unsigned long long)rank, groups, tokens, artifact);
    if (file) fclose(file);
    if (fd >= 0) close(fd);
    if (path_created) unlink(path);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(heads);
    if (initialized) ds4_gpu_cleanup();
    free(ref); free(got); free(x); free(model);
    return rc;
}

int main(void) {
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess ||
        prop.major != 12 || prop.minor != 1 || !prop.integrated) {
        fprintf(stderr, "SKIP: Q8 prefill fast paths require integrated sm_121\n");
        return 0;
    }
    /* Force the existing Q8 path, not the separate optional FP16 cuBLAS path. */
    setenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A", "1", 1);
    /* Keep the tiny fixture and its raw-load padding in one device allocation.
     * Exact-size raw tensor overreads are covered separately from this test. */
    setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    const uint32_t counts[] = {1, 7, 8, 9, 15, 16, 17, 32};
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        if (check_shape(33, 17, 3, counts[i], 0) ||
            check_shape(4096, 128, 4, counts[i], 0) ||
            check_shape(4096, 128, 4, counts[i], 1)) return 1;
    }
    /* A partial eight-warp CTA and ragged rows in the exact MMA tile. */
    if (check_shape(257, 65, 3, 17, 0) ||
        check_shape(4096, 65, 128, 17, 1) ||
        check_shape(8192, 128, 4, 8, 1) ||
        check_shape(4064, 128, 4, 8, 0)) return 1;
    fprintf(stderr, "PASS: Q8 prefill paths are byte-identical across 28 shapes\n");
    return 0;
}
