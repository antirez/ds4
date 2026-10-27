/* F16 embedding lookup and HC expansion must be bit-exact for short blocks. */
#include "ds4_gpu.h"
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    enum { VOCAB = 17, DIM = 4096, HC = 4, ROWS = 513 };
    const uint64_t offset = 64, size = offset + VOCAB * DIM * 2u, start = 0;
    unsigned char *model = calloc(1, size);
    float *actual = malloc(((size_t)ROWS * HC * DIM + 1) * sizeof(float));
    int32_t ids[ROWS];
    assert(model && actual && sizeof(_Float16) == sizeof(uint16_t));
    for (unsigned i = 0; i < VOCAB * DIM; i++) {
        uint16_t bits = (uint16_t)(i * 1777u + 31u);
        if ((bits & 0x7c00u) == 0x7c00u) bits ^= 0x0400u;
        memcpy(model + offset + i * 2u, &bits, sizeof(bits));
    }
    for (unsigned i = 0; i < ROWS; i++) ids[i] = i % VOCAB;
    char path[] = "/tmp/ds4-embed-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0 && unlink(path) == 0);
    for (uint64_t n = 0; n < size;) {
        const ssize_t wrote = write(fd, model + n, size - n);
        if (wrote < 0 && errno == EINTR) continue;
        assert(wrote > 0);
        n += (uint64_t)wrote;
    }
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model, size, &start, &size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model));
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(((uint64_t)ROWS * HC * DIM + 1) * sizeof(float));
    assert(tokens && out && ds4_gpu_tensor_write(tokens, 0, ids, sizeof(ids)));
    const unsigned counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 511, 512, 513};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        const unsigned rows = counts[c];
        const size_t values = (size_t)rows * HC * DIM;
        const float canary = 12345;
        assert(ds4_gpu_tensor_write(out, values * sizeof(float), &canary, sizeof(canary)));
        assert(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, size, offset,
                                              VOCAB, rows, DIM, HC));
        assert(ds4_gpu_tensor_read(out, 0, actual, (values + 1) * sizeof(float)));
        assert(actual[values] == canary);
        for (unsigned r = 0; r < rows; r++) {
            for (unsigned h = 0; h < HC; h++) {
                for (unsigned k = 0; k < DIM; k++) {
                    _Float16 half;
                    memcpy(&half, model + offset + ((size_t)ids[r] * DIM + k) * 2u, 2);
                    const float expected = (float)half;
                    const size_t index = ((size_t)r * HC + h) * DIM + k;
                    assert(memcmp(actual + index, &expected, sizeof(float)) == 0);
                }
            }
        }
        printf("CUDA F16 embedding rows=%u: exact\n", rows);
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup();
    assert(close(fd) == 0);
    free(actual);
    free(model);
    return 0;
}
