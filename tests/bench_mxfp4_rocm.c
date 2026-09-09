/* MXFP4 routed-MoE prefill benchmark (ROCm).
 *
 * Times ds4_gpu_routed_moe_batch_tensor at production prefill shapes:
 * n_tokens=4096, 256 experts, top-8, model_dim=7168, ffn_dim=2048.
 * Weights are synthetic (kernel cost is data-independent); selection is
 * uniform-random with a fixed seed, matching the production ~128
 * tokens/expert distribution.
 *
 * Not a correctness harness — tests/test_mxfp4_rocm.c owns exactness.
 * This exists so kernel retiling work has a seconds-scale iteration loop
 * instead of 75-second distributed probes.
 *
 * Usage: tests/bench_mxfp4_rocm [n_tokens] [iters]
 */

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ds4_gpu.h"

#define MXFP4_TYPE 39u
#define QK_MXFP4 32u

#ifndef MODEL_DIM
#define MODEL_DIM 7168u
#endif
#ifndef FFN_DIM
#define FFN_DIM 2048u
#endif
#ifndef N_TOTAL_EXPERT
#define N_TOTAL_EXPERT 256u
#endif
#ifndef N_EXPERT
#define N_EXPERT 8u
#endif
#define CLAMP 7.0f

typedef __attribute__((__may_alias__)) struct {
    uint8_t e;
    uint8_t qs[16];
} block_mxfp4;

static uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1u) / a * a;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_matrix(block_mxfp4 *m, uint32_t rows, uint32_t cols, uint32_t seed) {
    /* Cheap deterministic fill: any nibble pattern exercises the same
     * kernel cost; keep the e8m0 scale nonzero to avoid denormal edges. */
    uint64_t state = seed;
    const uint64_t blocks = (uint64_t)rows * (cols / QK_MXFP4);
    for (uint64_t i = 0; i < blocks; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        m[i].e = (uint8_t)(126u + (state >> 59u));
        for (uint32_t k = 0; k < 16u; k++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            m[i].qs[k] = (uint8_t)(state >> 56u);
        }
    }
}

int main(int argc, char **argv) {
    const uint32_t n_tokens = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096u;
    const int iters = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 6;

    const uint64_t gate_row_bytes = (MODEL_DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t gate_expert_bytes = FFN_DIM * gate_row_bytes;
    const uint64_t gate_tensor_bytes = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_row_bytes = (FFN_DIM / QK_MXFP4) * sizeof(block_mxfp4);
    const uint64_t down_expert_bytes = MODEL_DIM * down_row_bytes;
    const uint64_t down_tensor_bytes = N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = align_up_u64(gate_tensor_bytes, 4096u);
    const uint64_t down_offset = align_up_u64(up_offset + gate_tensor_bytes, 4096u);
    const uint64_t model_size = align_up_u64(down_offset + down_tensor_bytes, 4096u);

    fprintf(stderr,
            "bench: tokens=%u iters=%d model=%.2f GiB (gate/up %.2f GiB each, down %.2f GiB)\n",
            n_tokens, iters, (double)model_size / 1073741824.0,
            (double)gate_tensor_bytes / 1073741824.0,
            (double)down_tensor_bytes / 1073741824.0);

    FILE *model_file = tmpfile();
    void *model = MAP_FAILED;
    if (model_file && ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    if (!model_file || model == MAP_FAILED) {
        fprintf(stderr, "bench: model image allocation failed\n");
        return 1;
    }
    fill_matrix((block_mxfp4 *)((uint8_t *)model + gate_offset), FFN_DIM, MODEL_DIM, 0x12345678u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + up_offset), FFN_DIM, MODEL_DIM, 0x9abcdef0u);
    fill_matrix((block_mxfp4 *)((uint8_t *)model + down_offset), MODEL_DIM, FFN_DIM, 0x0f1e2d3cu);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "bench: ds4_gpu_init failed\n");
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t model_offsets[] = { gate_offset, up_offset, down_offset };
    const uint64_t model_sizes[] = { gate_tensor_bytes, gate_tensor_bytes, down_tensor_bytes };
    const uint64_t max_tensor_bytes = gate_tensor_bytes;
    if (!ds4_gpu_set_model_map(model, model_size) ||
        !ds4_gpu_set_model_fd(fileno(model_file)) ||
        !ds4_gpu_set_model_map_spans(model, model_size, model_offsets, model_sizes,
                                     3u, max_tensor_bytes)) {
        fprintf(stderr, "bench: model cache setup failed\n");
        return 1;
    }

    const uint64_t token_x_count = (uint64_t)n_tokens * MODEL_DIM;
    const uint64_t route_count = (uint64_t)n_tokens * N_EXPERT;
    const uint64_t mid_count = route_count * FFN_DIM;
    const uint64_t down_count = route_count * MODEL_DIM;

    float *x = (float *)calloc((size_t)token_x_count, sizeof(float));
    int32_t *selected = (int32_t *)calloc((size_t)route_count, sizeof(int32_t));
    float *weights = (float *)calloc((size_t)route_count, sizeof(float));
    if (!x || !selected || !weights) {
        fprintf(stderr, "bench: host allocation failed\n");
        return 1;
    }
    uint64_t rng = 0x243f6a8885a308d3ull;
    for (uint64_t i = 0; i < token_x_count; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        x[i] = ((double)(rng >> 40u) / (double)(1ull << 24u) - 0.5) * 0.03125;
    }
    /* Uniform top-8 without replacement, fixed seed: ~n_tokens*8/256
     * tokens per expert, matching production routing balance. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t pool[N_TOTAL_EXPERT];
        for (uint32_t e = 0; e < N_TOTAL_EXPERT; e++) pool[e] = e;
        for (uint32_t s = 0; s < N_EXPERT; s++) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            const uint32_t j = s + (uint32_t)((rng >> 33u) % (N_TOTAL_EXPERT - s));
            const uint32_t tmp = pool[s]; pool[s] = pool[j]; pool[j] = tmp;
            selected[(uint64_t)t * N_EXPERT + s] = (int32_t)pool[s];
            weights[(uint64_t)t * N_EXPERT + s] = 1.0f / (float)N_EXPERT;
        }
    }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(token_x_count * sizeof(float));
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(route_count * sizeof(int32_t));
    ds4_gpu_tensor *w_t = ds4_gpu_tensor_alloc(route_count * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up_t = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid_t = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down_t = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(token_x_count * sizeof(float));
    if (!x_t || !sel_t || !w_t || !gate_t || !up_t || !mid_t || !down_t || !out_t) {
        fprintf(stderr, "bench: tensor allocation failed\n");
        return 1;
    }
    if (!ds4_gpu_tensor_write(x_t, 0u, x, token_x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(sel_t, 0u, selected, route_count * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(w_t, 0u, weights, route_count * sizeof(float))) {
        fprintf(stderr, "bench: tensor upload failed\n");
        return 1;
    }

    double best = 1e30;
    double total = 0.0;
    int done = 0;
    for (int it = 0; it < iters; it++) {
        bool mid_is_f16 = false;
        const double t0 = now_sec();
        const int ok = ds4_gpu_routed_moe_batch_tensor(
            out_t, gate_t, up_t, mid_t, down_t,
            model, model_size,
            gate_offset, up_offset, down_offset,
            MXFP4_TYPE, MXFP4_TYPE,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            MODEL_DIM, FFN_DIM, MODEL_DIM,
            sel_t, w_t, N_TOTAL_EXPERT, N_EXPERT,
            CLAMP, x_t, 0u, n_tokens, &mid_is_f16, true);
        if (!ok) {
            fprintf(stderr, "bench: iteration %d launch failed\n", it);
            return 1;
        }
        if (!ds4_gpu_synchronize()) {
            fprintf(stderr, "bench: iteration %d sync failed\n", it);
            return 1;
        }
        const double dt = now_sec() - t0;
        if (it > 0) { /* first iteration absorbs plan/scratch warmup */
            if (dt < best) best = dt;
            total += dt;
            done++;
        }
        fprintf(stderr, "bench: iter %d %.1f ms\n", it, dt * 1000.0);
    }
    const double unique_bytes =
        (double)gate_tensor_bytes * 2.0 + (double)down_tensor_bytes;
    const double avg = total / (double)(done > 0 ? done : 1);
    fprintf(stderr,
            "bench: best %.1f ms avg %.1f ms | unique weights %.2f GiB -> %.1f GB/s (best)\n",
            best * 1000.0, avg * 1000.0,
            unique_bytes / 1073741824.0,
            unique_bytes / best / 1e9);

    ds4_gpu_cleanup();
    return 0;
}
