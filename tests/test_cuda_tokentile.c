#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "%s failed\n", what); exit(1); }
}

static ds4_gpu_tensor *upload(const float *data, size_t count) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(count * sizeof(float));
    check(t && ds4_gpu_tensor_write(t, 0, data, count * sizeof(float)), "upload");
    return t;
}

static void run(float *model, uint32_t tokens, uint32_t pos, uint32_t ratio,
                uint32_t H, uint32_t raw_start, float query_scale) {
    enum { D = 512, W = 128 };
    const uint32_t past = pos < W - 1 ? pos : W - 1;
    const uint32_t raw_count = past + tokens;
    const uint32_t comp_count = ratio ? (pos + tokens) / ratio : 0;
    const size_t nq = (size_t)tokens * H * D;
    float *q = malloc(nq * sizeof(float));
    float *actual = malloc((nq + 8) * sizeof(float));
    float *raw = malloc((size_t)raw_count * D * sizeof(float));
    float *comp = malloc((size_t)(comp_count + 1) * D * sizeof(float));
    check(q && actual && raw && comp, "host allocation");
    /* Exactly representable inputs. Nonzero queries exercise the score MMA,
     * varying rows exercise the value MMA, and sinks exercise normalization. */
    for (size_t i = 0; i < nq; i++) q[i] = ((int)(i % 17) - 8) * query_scale;
    for (size_t i = 0; i < (size_t)raw_count * D; i++)
        raw[i] = ((int)((i * 7 + i / D) % 29) - 14) * 0.03125f;
    for (size_t i = 0; i < (size_t)(comp_count + 1) * D; i++)
        comp[i] = ((int)((i * 11 + i / D) % 31) - 15) * 0.03125f;
    float *ring = malloc((size_t)raw_count * D * sizeof(float));
    check(ring != NULL, "ring allocation");
    for (uint32_t r = 0; r < raw_count; r++) for (uint32_t d = 0; d < D; d++)
        ring[(size_t)((raw_start + r) % raw_count) * D + d] = raw[(size_t)r * D + d];
    ds4_gpu_tensor *qg = upload(q, nq), *rg = upload(ring, (size_t)raw_count * D);
    ds4_gpu_tensor *cg = upload(comp, (size_t)(comp_count + 1) * D);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((nq + 8) * sizeof(float));
    check(out && ds4_gpu_tensor_fill_f32(out, NAN, nq + 8), "output allocation");
    if (ratio)
        check(ds4_gpu_attention_decode_mixed_batch_heads_tensor(out, model, 65536, 0,
              qg, rg, cg, 0, NULL, 0, tokens, pos, raw_count, raw_count, raw_start,
              comp_count, W, ratio, H, D), "dense mixed attention");
    else
        check(ds4_gpu_attention_decode_raw_batch_heads_tensor(out, model, 65536, 0,
              qg, rg, tokens, pos, raw_count, raw_count, raw_start, W, H, D), "raw attention");
    check(ds4_gpu_tensor_read(out, 0, actual, (nq + 8) * sizeof(float)), "attention read");
    for (size_t i = nq; i < nq + 8; i++) check(isnan(actual[i]), "output tail");
    double max_error = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t begin = past + t + 1 > W ? past + t + 1 - W : 0;
        const uint32_t visible = ratio ? (pos + t + 1) / ratio : 0;
        for (uint32_t h = 0; h < H; h += 7) {
            const size_t base = ((size_t)t * H + h) * D;
            double maximum = model[h];
            for (uint32_t r = begin; r < past + t + 1 + visible; r++) {
                const float *kv = r < past + t + 1 ? raw + (size_t)r * D :
                                  comp + (size_t)(r - past - t - 1) * D;
                double dot = 0;
                for (uint32_t d = 0; d < D; d++) dot += (double)q[base + d] * kv[d];
                maximum = fmax(maximum, dot / sqrt(D));
            }
            double sum = exp(model[h] - maximum), value[D] = {0};
            for (uint32_t r = begin; r < past + t + 1 + visible; r++) {
                const float *kv = r < past + t + 1 ? raw + (size_t)r * D :
                                  comp + (size_t)(r - past - t - 1) * D;
                double dot = 0;
                for (uint32_t d = 0; d < D; d++) dot += (double)q[base + d] * kv[d];
                const double p = exp(dot / sqrt(D) - maximum);
                sum += p;
                for (uint32_t d = 0; d < D; d++) value[d] += p * kv[d];
            }
            for (uint32_t d = 0; d < D; d++) {
                const double error = fabs(actual[base + d] - value[d] / sum);
                check(isfinite(actual[base + d]) && error < 0.001, "double attention reference");
                if (error > max_error) max_error = error;
            }
        }
    }
    printf("attention tokens=%u pos=%u ratio=%u heads=%u ring=%u max_error=%.8g: PASS\n",
           tokens, pos, ratio, H, raw_start, max_error);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(cg);
    ds4_gpu_tensor_free(rg); ds4_gpu_tensor_free(qg);
    free(ring); free(comp); free(raw); free(actual); free(q);
}

/* With the window and single-token online paths forced, every batch goes
 * through the online decode kernel.  Its rows are independent, so each
 * batched row must equal the same token computed alone, bit for bit.  On
 * Spark, batches of up to 8 rows and single tokens use the staged variant
 * while larger batches keep the baseline, so this also checks that both
 * produce identical bits. */
static void run_online_rows(float *model, uint32_t tokens, uint32_t pos, uint32_t ratio,
                            uint32_t H, uint32_t raw_start, float query_scale) {
    enum { D = 512, W = 128 };
    const uint32_t past = pos < W - 1 ? pos : W - 1;
    const uint32_t raw_count = past + tokens;
    const uint32_t comp_count = ratio ? (pos + tokens) / ratio : 0;
    const size_t row = (size_t)H * D, nq = (size_t)tokens * row;
    float *q = malloc(nq * sizeof(float));
    float *batch = malloc((nq + 8) * sizeof(float));
    float *single = malloc((nq + 8) * sizeof(float));
    float *ring = malloc((size_t)raw_count * D * sizeof(float));
    float *comp = malloc((size_t)(comp_count + 1) * D * sizeof(float));
    check(q && batch && single && ring && comp, "host allocation");
    for (size_t i = 0; i < nq; i++) q[i] = ((int)(i % 23) - 11) * query_scale;
    for (size_t i = 0; i < (size_t)raw_count * D; i++)
        ring[i] = ((int)((i * 5 + i / D) % 37) - 18) * 0.03125f;
    for (size_t i = 0; i < (size_t)(comp_count + 1) * D; i++)
        comp[i] = ((int)((i * 13 + i / D) % 41) - 20) * 0.03125f;
    ds4_gpu_tensor *qg = upload(q, nq), *rg = upload(ring, (size_t)raw_count * D);
    ds4_gpu_tensor *cg = upload(comp, (size_t)(comp_count + 1) * D);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((nq + 8) * sizeof(float));
    check(out && ds4_gpu_tensor_fill_f32(out, NAN, nq + 8), "output allocation");
    /* The ring holds the raw rows from first_raw_pos = pos - past on; a
     * single token at pos + t sees the same ring with its later rows absent. */
    for (uint32_t t = 0; t <= tokens; t++) {
        const uint32_t n = t == tokens ? tokens : 1;
        const uint32_t p0 = t == tokens ? pos : pos + t;
        const uint32_t n_raw = t == tokens ? raw_count : past + t + 1;
        ds4_gpu_tensor *qv = t == tokens ? qg : ds4_gpu_tensor_view(qg, t * row * 4, row * 4);
        ds4_gpu_tensor *ov = t == tokens ? out : ds4_gpu_tensor_view(out, t * row * 4, row * 4);
        check(qv && ov, "row views");
        if (ratio)
            check(ds4_gpu_attention_decode_mixed_batch_heads_tensor(ov, model, 65536, 0,
                  qv, rg, cg, 0, NULL, 0, n, p0, n_raw, raw_count, raw_start,
                  comp_count, W, ratio, H, D), "online mixed attention");
        else
            check(ds4_gpu_attention_decode_raw_batch_heads_tensor(ov, model, 65536, 0,
                  qv, rg, n, p0, n_raw, raw_count, raw_start, W, H, D), "online raw attention");
        if (t == tokens) {
            check(ds4_gpu_tensor_read(out, 0, batch, (nq + 8) * sizeof(float)), "batch read");
        } else {
            ds4_gpu_tensor_free(ov);
            ds4_gpu_tensor_free(qv);
            if (t + 1 == tokens) {
                check(ds4_gpu_tensor_read(out, 0, single, (nq + 8) * sizeof(float)), "rows read");
                check(ds4_gpu_tensor_fill_f32(out, NAN, nq + 8), "output reset");
            }
        }
    }
    for (size_t i = 0; i < nq; i++) {
        if (!isfinite(batch[i]) || memcmp(&batch[i], &single[i], sizeof(float))) {
            fprintf(stderr, "online attention tokens=%u pos=%u ratio=%u heads=%u i=%zu: %.9g != %.9g\n",
                    tokens, pos, ratio, H, i, batch[i], single[i]);
            exit(1);
        }
    }
    for (size_t i = nq; i < nq + 8; i++) check(isnan(batch[i]) && isnan(single[i]), "online output tail");
    printf("online attention tokens=%u pos=%u ratio=%u heads=%u ring=%u: rows exact PASS\n",
           tokens, pos, ratio, H, raw_start);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(cg);
    ds4_gpu_tensor_free(rg); ds4_gpu_tensor_free(qg);
    free(comp); free(ring); free(single); free(batch); free(q);
}

int main(void) {
    /* Keep the tiny fake model on the GPU. Unsupported host-registration
     * probes otherwise obscure memcheck's kernel-access diagnostics. */
    check(setenv("DS4_CUDA_COPY_MODEL", "1", 1) == 0, "test model placement");
    float *model = mmap(NULL, 65536, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
    check(model != MAP_FAILED, "model allocation");
    for (int i = 0; i < 64; i++) model[i] = (i % 7 - 3) * 0.125f;
    check(ds4_gpu_init() && ds4_gpu_set_model_map(model, 65536), "GPU initialization");
    run(model, 128, 0, 0, 64, 0, 0.03125f);
    run(model, 131, 513, 0, 64, 0, 0.03125f);
    run(model, 129, 0, 2, 64, 0, 0.03125f);
    run(model, 257, 1025, 2, 64, 0, 0.03125f);
    run(model, 131, 129, 1, 64, 0, 0.03125f);
    for (uint32_t rows = 2; rows <= 7; rows++) {
        run(model, rows, 0, 0, 61, 0, 0.03125f);
        run(model, rows, 127, 0, 64, 37, 3.0f);
        run(model, rows, 127, 4, 64, 37, 0.03125f);
        run(model, rows, 2047, 4, 61, 37, 3.0f);
        run(model, rows, 32767, 128, 64, 37, 0.03125f);
    }
    ds4_gpu_cleanup();

    /* Online decode kernel for every batch size, checked against the double
     * reference and row by row against single tokens. */
    check(setenv("DS4_CUDA_WINDOW_ATTENTION", "1", 1) == 0 &&
          setenv("DS4_CUDA_DECODE_HEADS8_ONLINE", "1", 1) == 0, "online attention paths");
    check(ds4_gpu_init() && ds4_gpu_set_model_map(model, 65536), "GPU reinitialization");
    for (uint32_t rows = 2; rows <= 8; rows += 3) {
        run(model, rows, 127, 4, 32, 37, 0.03125f);
        run(model, rows, 2047, 4, 61, 11, 3.0f);
    }
    run(model, 128, 1000, 4, 32, 5, 0.03125f);
    run_online_rows(model, 2, 0, 0, 61, 0, 0.03125f);
    run_online_rows(model, 3, 5, 4, 32, 2, 3.0f);
    run_online_rows(model, 6, 105, 4, 32, 37, 0.03125f);
    run_online_rows(model, 5, 2047, 4, 61, 11, 3.0f);
    run_online_rows(model, 8, 4000, 128, 64, 37, 0.03125f);
    run_online_rows(model, 128, 1000, 4, 32, 5, 0.03125f);
    run_online_rows(model, 131, 513, 0, 61, 7, 0.03125f);
    ds4_gpu_cleanup();
    munmap(model, 65536);
    return 0;
}
