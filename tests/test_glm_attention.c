#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "ds4_linux_memory.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "%s failed\n", what); exit(1); }
}

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static bool benchmark_has_headroom(void) {
    uint64_t available;
    return ds4_linux_nonmovable_memory(&available) && available >= (32ull << 30);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    check(t && ds4_gpu_tensor_write(t, 0, data, bytes), "tensor upload");
    return t;
}

static ds4_gpu_tensor *upload_cache(const float *data, size_t count, bool f16) {
    if (!f16) return upload(data, count * sizeof(float));
    _Float16 *half = malloc(count * sizeof(*half));
    check(half != NULL, "half cache allocation");
    for (size_t i = 0; i < count; i++) half[i] = data[i];
    ds4_gpu_tensor *t = upload(half, count * sizeof(*half));
    free(half);
    return t;
}

static void attention_reference(float *out, const float *q, const float *low,
        const float *kv, const float *kr,
        const int32_t *ids, uint32_t tokens, uint32_t heads, uint32_t dim,
        uint32_t pos, uint32_t cap, uint32_t count, uint32_t nope,
        uint32_t rope, bool selected) {
    double *weights = calloc(count, sizeof(*weights));
    check(weights != NULL, "reference allocation");
    for (uint32_t t = 0; t < tokens; t++) {
        uint32_t visible = selected ? count : (pos + t + 1);
        if (visible > count) visible = count;
        for (uint32_t h = 0; h < heads; h++) {
            const size_t base = ((size_t)t * heads + h) * dim;
            double sum = 0;
            for (uint32_t i = 0; i < visible; i++) {
                const uint32_t row = selected ? (uint32_t)ids[t * count + i] : i;
                double dot = 0;
                weights[i] = 0;
                if (row >= cap) continue;
                for (uint32_t d = 0; d < dim; d++)
                    dot += (double)low[base + d] * kv[(size_t)row * dim + d];
                for (uint32_t d = 0; d < rope; d += 2) {
                    const double theta = row * pow(10000.0, -(double)d / rope);
                    const double x = kr[(size_t)row * rope + d];
                    const double y = kr[(size_t)row * rope + d + 1];
                    const size_t qi = ((size_t)t * heads + h) * (nope + rope) + nope + d;
                    dot += q[qi] * (x * cos(theta) - y * sin(theta)) +
                           q[qi + 1] * (x * sin(theta) + y * cos(theta));
                }
                sum += weights[i] = exp(dot / sqrt(nope + rope));
            }
            for (uint32_t d = 0; d < dim; d++) {
                double value = 0;
                for (uint32_t i = 0; i < visible; i++) {
                    const uint32_t row = selected ? (uint32_t)ids[t * count + i] : i;
                    if (row < cap) value += weights[i] * kv[(size_t)row * dim + d];
                }
                out[base + d] = sum > 0 ? value / sum : 0;
            }
        }
    }
    free(weights);
}

static void attention_case(uint32_t tokens, uint32_t heads, uint32_t dim,
                           uint32_t pos, uint32_t cap, bool selected,
                           bool zero_rope, bool bench, bool f16) {
    const uint32_t nope = zero_rope ? 256 : 32, rope = zero_rope ? 0 : 64;
    const uint32_t count = selected ? (cap > 2048 ? 2051 : 8) : cap;
    const bool known_answer = selected && !f16;
    const size_t out_n = (size_t)tokens * heads * dim;
    const size_t q_n = (size_t)tokens * heads * (nope + rope);
    const size_t kv_n = (size_t)cap * dim;
    const size_t rope_n = (size_t)cap * rope;
    float *q = calloc(q_n, sizeof(float));
    float *low = calloc(out_n, sizeof(float));
    float *kv = malloc(kv_n * sizeof(float));
    float *kr = calloc(rope_n ? rope_n : 1, sizeof(float));
    int32_t *ids = malloc((size_t)tokens * count * sizeof(int32_t));
    float *ref = malloc(out_n * sizeof(float));
    float *actual = malloc(out_n * sizeof(float));
    float *repeat = malloc(out_n * sizeof(float));
    check(q && low && kv && kr && ids && ref && actual && repeat,
          "host allocation");
    for (size_t i = 0; i < kv_n; i++)
        kv[i] = known_answer ? 1.0f : ((int)(i % 31) - 15) * 0.03125f;
    if (!known_answer) {
        for (size_t i = 0; i < out_n; i++)
            low[i] = ((int)(i % 23) - 11) * 0.015625f;
        for (size_t i = 0; i < q_n; i++)
            q[i] = ((int)(i % 17) - 8) * 0.03125f;
        for (size_t i = 0; i < rope_n; i++)
            kr[i] = ((int)(i % 13) - 6) * 0.0625f;
    }
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t i = 0; i < count; i++) {
            /* Include valid, padded, out-of-range and entirely empty rows. */
            ids[(size_t)t * count + i] = t % 3 == 0 ? -1 :
                (t % 3 == 1 && i % 2 == 0 ? (i == 0 ? -1 : (int32_t)cap) :
                 (int32_t)(i % cap));
        }
    }
    ds4_gpu_tensor *qg = upload(q, q_n * sizeof(float));
    ds4_gpu_tensor *lg = upload(low, out_n * sizeof(float));
    ds4_gpu_tensor *kg = upload_cache(kv, kv_n, f16);
    ds4_gpu_tensor *rg = upload_cache(kr, rope_n ? rope_n : 1, f16);
    ds4_gpu_tensor *ig = selected ? upload(ids, (size_t)tokens * count * sizeof(int32_t)) : NULL;
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    check(out != NULL, "output allocation");
    float *oracle = NULL;
    if (!bench) {
        oracle = malloc(out_n * sizeof(float));
        check(oracle != NULL, "attention oracle allocation");
        attention_reference(oracle, q, low, kv, kr, ids, tokens, heads, dim,
                            pos, cap, count, nope, rope, selected);
    }
    const char *flag = selected ? "DS4_ROCM_GLM_SELECTED_ATTN_GEMM" :
                                  "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM";
    double elapsed = 0;
    const int runs = bench ? 6 : 3;
    for (int run = 0; run < runs; run++) {
        if (!bench && run == 0) setenv(flag, "0", 1);
        else unsetenv(flag);
        check(ds4_gpu_synchronize(), "attention pre-sync");
        const double start = seconds();
        int ok = selected ? ds4_gpu_glm_attention_indexed_batch_lora_tensor(
            out, qg, lg, kg, rg, ig, tokens, count, cap, f16,
            heads, dim, nope, rope, 4096, 10000, 1, 0, 1, 32, 1) :
            ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                out, qg, lg, kg, rg, tokens, pos, count, cap, f16,
                heads, dim, nope, rope, 4096, 10000, 1, 0, 1, 32, 1);
        check(ok && ds4_gpu_synchronize(), "attention launch");
        if (run > 0) elapsed += seconds() - start;
        float *result = run == 0 ? ref : (run == 1 ? actual : repeat);
        check(ds4_gpu_tensor_read(out, 0, result, out_n * sizeof(float)),
              "attention read");
        for (size_t i = 0; i < out_n; i++) {
            const float expected = known_answer ?
                (i / (heads * dim) % 3 == 0 ? 0.0f : 1.0f) :
                (oracle ? oracle[i] : ref[i]);
            if (!isfinite(result[i]) || fabsf(result[i] - expected) > 0.002f) {
                fprintf(stderr, "attention selected=%d tokens=%u heads=%u pos=%u "
                        "run=%d index=%zu: %.9g != %.9g\n", selected, tokens,
                        heads, pos, run, i, result[i], expected);
                exit(1);
            }
        }
        if (run >= 2)
            check(memcmp(actual, repeat, out_n * sizeof(float)) == 0,
                  "attention repeat determinism");
    }
    unsetenv(flag);
#ifndef DS4_ROCM_BUILD
    if (!bench && !selected && zero_rope && f16) {
        for (int run = 0; run < 3; run++) {
            check(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
                out, lg, kg, pos, tokens, cap, cap, f16, heads, dim, nope),
                "dense compact GEMM attention");
            float *result = run == 0 ? actual : repeat;
            check(ds4_gpu_tensor_read(out, 0, result, out_n * sizeof(float)),
                  "dense compact attention read");
            for (size_t i = 0; i < out_n; i++)
                check(isfinite(result[i]) && fabsf(result[i] - oracle[i]) < 0.002f,
                      "dense compact attention reference");
            if (run > 0)
                check(memcmp(actual, repeat, out_n * sizeof(float)) == 0,
                      "dense compact attention repeat determinism");
        }
    }
#endif
    printf("attention selected=%d tokens=%u heads=%u dim=%u pos=%u cap=%u rope=%u f16=%d: "
           "PASS, %.3f ms\n", selected, tokens, heads, dim, pos, cap, rope,
           f16, elapsed * 1000 / (runs - 1));
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(ig);
    ds4_gpu_tensor_free(rg);
    ds4_gpu_tensor_free(kg);
    ds4_gpu_tensor_free(lg);
    ds4_gpu_tensor_free(qg);
    free(oracle);
    free(repeat); free(actual); free(ref); free(ids);
    free(kr); free(kv); free(low); free(q);
}

static void reduction_cases(void) {
    enum { D = 512, N = 128, T = 3, C = 521, H = 8, V = 4, R = 64, MODEL_BYTES = 65536 };
    unsigned char *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANON, -1, 0);
    check(model != MAP_FAILED, "test model allocation");
    float *weight = (float *)model, *bias = weight + N;
    float raw[T * N], result[T * N];
    for (int i = 0; i < N; i++) { weight[i] = 1; bias[i] = 0.25f; }
    for (int i = 0; i < T * N; i++) raw[i] = 128 + (i % 29) * 0.125f;
    /* Q8 value projections: every weight is exactly 1. */
    for (int i = 0; i < H * V * (D / 32); i++) {
        unsigned char *block = model + 4096 + i * 34;
        _Float16 scale = 1;
        memcpy(block, &scale, sizeof(scale));
        memset(block + 2, 1, 32);
    }
    check(ds4_gpu_set_model_map(model, MODEL_BYTES), "test model registration");
    ds4_gpu_tensor *x = upload(raw, sizeof(raw));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(raw));
    for (int run = 0; run < 20; run++) {
        check(ds4_gpu_glm_store_indexer_k_tensor(out, x, model, MODEL_BYTES,
              0, N * sizeof(float), 0, T, T, N, R, 4096,
              1e-5f, 10000, 1, 0, 1, 32, 1, false), "indexer normalization");
        check(ds4_gpu_tensor_read(out, 0, result, sizeof(result)), "indexer read");
        for (int t = 0; t < T; t++) {
            double mean = 0, variance = 0;
            for (int i = 0; i < N; i++) mean += raw[t * N + i];
            mean /= N;
            for (int i = 0; i < N; i++) {
                double d = raw[t * N + i] - mean;
                variance += d * d;
            }
            for (int i = 0; i < N; i++) {
                double expected = (raw[t * N + i] - mean) /
                    sqrt(variance / N + 1e-5) + 0.25;
                if (i < R) {
                    const int pair = i ^ 1;
                    const double other = (raw[t * N + pair] - mean) /
                        sqrt(variance / N + 1e-5) + 0.25;
                    const double theta = t * pow(10000, -(double)(i & ~1) / R);
                    expected = expected * cos(theta) +
                        (i & 1 ? other : -other) * sin(theta);
                }
                check(isfinite(result[t * N + i]) &&
                      fabs(result[t * N + i] - expected) < 1e-4,
                      "indexer normalization reference");
            }
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(x);

    float q[T * R], keys[C * R], values[C * V], full[T * V];
    for (int i = 0; i < T * R; i++) q[i] = (i % 7 - 3) * 0.125f;
    for (int i = 0; i < C * R; i++) keys[i] = (i % 11 - 5) * 0.125f;
    for (int i = 0; i < C * V; i++) values[i] = (i % 19 - 9) * 0.125f;
    ds4_gpu_tensor *qg = upload(q, sizeof(q));
    ds4_gpu_tensor *kg = upload(keys, sizeof(keys));
    ds4_gpu_tensor *vg = upload(values, sizeof(values));
    out = ds4_gpu_tensor_alloc(sizeof(full));
    for (int run = 0; run < 20; run++) {
        check(ds4_gpu_glm_attention_full_tensor(out, qg, kg, vg,
              C - T, T, C, C, 1, R, V, false), "full attention");
        check(ds4_gpu_tensor_read(out, 0, full, sizeof(full)), "full attention read");
        for (int t = 0; t < T; t++) {
            double sum = 0, acc[V] = {0};
            for (int r = 0; r <= C - T + t; r++) {
                double score = 0;
                for (int d = 0; d < R; d++) score += q[t * R + d] * keys[r * R + d];
                double p = exp(score / sqrt(R));
                sum += p;
                for (int d = 0; d < V; d++) acc[d] += p * values[r * V + d];
            }
            for (int d = 0; d < V; d++)
                check(isfinite(full[t * V + d]) &&
                      fabs(full[t * V + d] - acc[d] / sum) < 1e-5,
                      "full attention reference");
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(vg);
    ds4_gpu_tensor_free(kg); ds4_gpu_tensor_free(qg);

#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    float *kv = malloc(C * D * sizeof(float));
    float low[H * D] = {0}, sq[H * (32 + R)] = {0}, rope[C * R] = {0};
    int32_t selected[C];
    check(kv != NULL, "split cache allocation");
    for (int i = 0; i < C * D; i++) kv[i] = 1;
    kg = upload_cache(kv, C * D, true);
    ds4_gpu_tensor *rg = upload_cache(rope, C * R, true);
    qg = upload(sq, sizeof(sq));
    ds4_gpu_tensor *lg = upload(low, sizeof(low));
    ds4_gpu_tensor *ig = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *pl = ds4_gpu_tensor_alloc(4 * H * D * sizeof(float));
    ds4_gpu_tensor *pm = ds4_gpu_tensor_alloc(4 * H * 2 * sizeof(float));
    out = ds4_gpu_tensor_alloc(H * V * sizeof(float));
    float split[H * V];
    for (int mode = 0; mode < 3; mode++) {
        for (int i = 0; i < C; i++) selected[i] = mode == 2 ? -1 :
            (mode == 1 && i % 2 == 0 ? (i % 4 == 0 ? -1 : C) : i);
        check(ds4_gpu_tensor_write(ig, 0, selected, sizeof(selected)), "split ids");
        /* Every context-length remainder must satisfy Metal's 16-byte
         * threadgroup allocation rule, in both cache formats. */
        for (int f16 = 0; f16 < 2; f16++) {
            ds4_gpu_tensor *cache = upload_cache(kv, C * D, f16);
            for (int count = C - 3; count <= C; count++) {
                check(ds4_gpu_glm_attention_indexed_decode_tensor(
                      out, qg, lg, cache, rg, model, MODEL_BYTES, 4096,
                      ig, count, C, f16, H, D, 32, 0, V, 4096,
                      10000, 1, 0, 1, 32, 1), "indexed decode attention");
                check(ds4_gpu_tensor_read(out, 0, split, sizeof(split)), "indexed decode read");
                for (int i = 0; i < H * V; i++)
                    check(isfinite(split[i]) && fabsf(split[i] - (mode == 2 ? 0 : D)) < 0.01f,
                          "indexed decode reference");
            }
            ds4_gpu_tensor_free(cache);
        }
        for (int run = 0; run < 20; run++) {
            check(ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
                  out, pl, pm, qg, lg, kg, rg, model, MODEL_BYTES, 4096,
                  ig, C, mode == 0, C, true, H, D, 32, R, V, 4096,
                  192, 4, 10000, 1, 0, 1, 32, 1), "split attention");
            check(ds4_gpu_tensor_read(out, 0, split, sizeof(split)), "split read");
            for (int i = 0; i < H * V; i++)
                check(isfinite(split[i]) && fabsf(split[i] - (mode == 2 ? 0 : D)) < 0.01f,
                      "split attention reference");
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(pm); ds4_gpu_tensor_free(pl);
    ds4_gpu_tensor_free(ig); ds4_gpu_tensor_free(lg); ds4_gpu_tensor_free(qg);
    ds4_gpu_tensor_free(rg); ds4_gpu_tensor_free(kg); free(kv);
#endif
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("normalization and attention reductions: PASS");
}

int main(int argc, char **argv) {
    const bool bench = argc == 2 && strcmp(argv[1], "--bench") == 0;
    check(argc == 1 || bench, "arguments");
    if (bench && !benchmark_has_headroom()) {
        fprintf(stderr, "Unload the model before benchmarking; 32 GiB available memory required.\n");
        return 1;
    }
    check(ds4_gpu_init(), "GPU initialization");
    if (bench) {
        attention_case(256, 64, 512, 0, 256, false, false, true, false);
        attention_case(512, 64, 512, 0, 512, false, false, true, false);
        attention_case(1024, 64, 512, 0, 1024, false, false, true, false);
        attention_case(256, 64, 512, 3840, 4096, false, false, true, false);
        attention_case(256, 64, 512, 0, 256, false, true, true, true);
        attention_case(1024, 64, 512, 0, 1024, false, true, true, true);
        attention_case(2048, 64, 512, 0, 2048, false, true, true, true);
        attention_case(2048, 64, 512, 2048, 4096, false, true, true, true);
        attention_case(256, 64, 512, 3840, 4096, false, true, true, true);
        ds4_gpu_cleanup();
    } else {
#ifdef DS4_ROCM_BUILD
        for (int f16 = 0; f16 < 2; f16++) {
            attention_case(64, 17, 32, 0, 32, true, false, false, f16);
            attention_case(67, 1, 32, 0, 32, true, false, false, f16);
            attention_case(128, 17, 32, 0, 32, true, false, false, f16);
            attention_case(256, 17, 32, 0, 32, true, false, false, f16);
            attention_case(256, 17, 32, 0, 256, false, false, false, f16);
            attention_case(257, 17, 32, 127, 384, false, false, false, f16);
            attention_case(256, 1, 32, 128, 384, false, true, false, f16);
        }
#endif
        attention_case(67, 17, 512, 0, 32, true, true, false, false);
        attention_case(67, 17, 512, 0, 32, true, true, false, true);
        attention_case(64, 17, 512, 0, 2052, true, true, false, true);
        attention_case(257, 17, 512, 127, 384, false, true, false, true);
        for (int f16 = 0; f16 < 2; f16++) {
            attention_case(32, 8, 512, 0, 32, false, false, false, f16);
            attention_case(31, 17, 512, 97, 128, false, false, false, f16);
        }
        reduction_cases();
    }
    return 0;
}
