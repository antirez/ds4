/* GGUF-free integration test of the real Metal F16 compressor APIs.
 * Compare the fused quad projection/state append with two ordinary paired
 * projections followed by separate state stores. Every output and complete
 * ring state is compared bitwise, including tensor-view guards, in eager and
 * batched command modes. Link with ds4_metal.o and ds4_image.o on macOS. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { K = 4096, W0 = 1024, W1 = 256, GUARD = 16 };
typedef struct {
    ds4_gpu_tensor *base, *view;
    size_t n;
} tensor;

static tensor make_tensor(size_t n) {
    tensor t = { .n = n };
    t.base = ds4_gpu_tensor_alloc((n + 2 * GUARD) * sizeof(float));
    t.view = t.base ? ds4_gpu_tensor_view(t.base, GUARD * sizeof(float),
                                        n * sizeof(float)) : NULL;
    if (!t.view) {
        fprintf(stderr, "tensor allocation failed\n");
        exit(2);
    }
    return t;
}

static void reset_tensor(tensor *t) {
    if (!ds4_gpu_tensor_fill_f32(t->base, -123.125f, t->n + 2 * GUARD))
        exit(3);
}

static void drop_tensor(tensor *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->base);
}

static int compare_tensor(const tensor *a, const tensor *b,
                          const char *what, unsigned case_id) {
    const size_t n = a->n + 2 * GUARD, bytes = n * sizeof(float);
    uint32_t *av = malloc(bytes), *bv = malloc(bytes);
    int ok = a->n == b->n && av && bv &&
             ds4_gpu_tensor_read(a->base, 0, av, bytes) &&
             ds4_gpu_tensor_read(b->base, 0, bv, bytes);
    if (ok && memcmp(av, bv, bytes)) {
        for (size_t i = 0; i < n; i++) {
            if (av[i] == bv[i]) continue;
            fprintf(stderr, "case %u %s index %zu: got %08x, ref %08x\n",
                    case_id, what, i, av[i], bv[i]);
            break;
        }
        ok = 0;
    }
    const float sentinel = -123.125f;
    uint32_t sentinel_bits;
    memcpy(&sentinel_bits, &sentinel, sizeof(sentinel_bits));
    for (size_t i = 0; ok && i < GUARD; i++) {
        if (av[i] != sentinel_bits || av[n - 1 - i] != sentinel_bits) {
            fprintf(stderr, "case %u %s: tensor guard modified\n", case_id, what);
            ok = 0;
        }
    }
    free(av);
    free(bv);
    return ok;
}

int main(void) {
    const uint32_t widths[4] = {W0, W0, W1, W1};
    const size_t page = (size_t)getpagesize();
    uint64_t woff[4], ape[2][2];
    /* A nonzero page-relative offset also exercises model-range wrapping. */
    uint64_t cursor = page + 128;
    for (int j = 0; j < 4; j++) {
        woff[j] = cursor;
        cursor += (uint64_t)widths[j] * K * sizeof(_Float16);
    }
    for (int j = 0; j < 2; j++) {
        for (int type = 0; type < 2; type++) {
            ape[j][type] = cursor;
            cursor += (uint64_t)(j ? W1 : W0) * 4 * sizeof(float);
        }
    }
    const uint64_t model_bytes = (cursor + page - 1) / page * page;
    void *model = NULL;
    if (posix_memalign(&model, page, (size_t)model_bytes)) return 2;
    memset(model, 0, (size_t)model_bytes);
    for (int j = 0; j < 4; j++) {
        _Float16 *w = (_Float16 *)((char *)model + woff[j]);
        for (size_t i = 0; i < (size_t)widths[j] * K; i++)
            w[i] = (_Float16)(((int)((i * 13 + i / 23 + j * 37) % 255) -
                              127) / 1024.0f);
    }
    for (int j = 0; j < 2; j++) {
        for (int type = 0; type < 2; type++) {
            const size_t n = (size_t)(j ? W1 : W0) * 4;
            for (size_t i = 0; i < n; i++) {
                float v = ((int)((i * 7 + j * 19) % 63) - 31) / 32.0f;
                if (type)
                    ((_Float16 *)((char *)model + ape[j][type]))[i] = (_Float16)v;
                else
                    ((float *)((char *)model + ape[j][type]))[i] = v;
            }
        }
    }
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, model_bytes)) return 3;
    ds4_gpu_set_quality(false);
    tensor x = make_tensor(K), got[8], ref[8];
    for (int j = 0; j < 8; j++) {
        size_t n = (size_t)widths[j % 4] * (j >= 4 ? 8 : 1);
        got[j] = make_tensor(n);
        ref[j] = make_tensor(n);
    }
    const uint32_t positions[] = {0, 1, 3, 4, 7, 127, 128, UINT32_MAX};
    unsigned cases = 0, checks = 0;
    float xv[K];
    for (int batched = 0; batched < 2; batched++) {
        for (int t0 = 0; t0 < 2; t0++) {
            for (int t1 = 0; t1 < 2; t1++) {
                for (unsigned p = 0; p < sizeof(positions) / sizeof(*positions); p++) {
                    const uint32_t pos = positions[p];
                    for (int k = 0; k < K; k++)
                        xv[k] = ((int)((k * 17 + cases * 29 + k / 5) % 511) -
                                 255) / 512.0f;
                    reset_tensor(&x);
                    if (!ds4_gpu_tensor_write(x.view, 0, xv, sizeof(xv))) return 4;
                    for (int j = 0; j < 8; j++) {
                        reset_tensor(&got[j]);
                        reset_tensor(&ref[j]);
                    }
                    if (batched && !ds4_gpu_begin_commands()) return 5;
                    int ok = ds4_gpu_matmul_f16_pair_tensor(
                            ref[0].view, ref[1].view, model, model_bytes,
                            woff[0], woff[1], K, W0, x.view, 1) &&
                        ds4_gpu_matmul_f16_pair_tensor(
                            ref[2].view, ref[3].view, model, model_bytes,
                            woff[2], woff[3], K, W1, x.view, 1) &&
                        ds4_gpu_compressor_store_batch_tensor(
                            ref[0].view, ref[1].view, ref[4].view, ref[5].view,
                            model, model_bytes, ape[0][t0], t0, W0 / 2, 4, pos, 1) &&
                        ds4_gpu_compressor_store_batch_tensor(
                            ref[2].view, ref[3].view, ref[6].view, ref[7].view,
                            model, model_bytes, ape[1][t1], t1, W1 / 2, 4, pos, 1) &&
                        ds4_gpu_matmul_f16_quad_compressor_store_tensor(
                            got[0].view, got[1].view, got[2].view, got[3].view,
                            got[4].view, got[5].view, got[6].view, got[7].view,
                            model, model_bytes, woff[0], woff[1], woff[2], woff[3],
                            ape[0][t0], t0, ape[1][t1], t1,
                            K, W0, W1, x.view, 4, pos) == 1;
                    if (batched) ok = ds4_gpu_end_commands() && ok;
                    if (!ok) {
                        fprintf(stderr, "API failed in case %u\n", cases);
                        return 5;
                    }
                    for (int j = 0; j < 8; j++) {
                        if (!compare_tensor(&got[j], &ref[j],
                                            j < 4 ? "projection" : "state", cases))
                            return 6;
                        checks++;
                    }
                    cases++;
                }
            }
        }
    }
    printf("PASS native real-host F16 quad: %u cases, %u full guarded tensor "
           "comparisons; eager + batched, APE F32/F16 combinations, "
           "positions 0/1/3/4/7/127/128/UINT32_MAX; dimensions 4096x1024/256.\n",
           cases, checks);
    for (int j = 0; j < 8; j++) {
        drop_tensor(&got[j]);
        drop_tensor(&ref[j]);
    }
    drop_tensor(&x);
    ds4_gpu_cleanup();
    free(model);
    return 0;
}

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}
