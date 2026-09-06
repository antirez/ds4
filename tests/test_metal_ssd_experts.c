#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { E = 256, N = 8, STEPS = 48 };
static int D = 256, H = 512;
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int check_mapping_lifetime(void) {
    const uint64_t page = getpagesize(), bytes = 4100 * page;
    float *target = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    float *aux = mmap(NULL, page, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
    if (target == MAP_FAILED || aux == MAP_FAILED) {
        if (target != MAP_FAILED) munmap(target, bytes);
        if (aux != MAP_FAILED) munmap(aux, page);
        return 0;
    }
    aux[0] = 3.25f;
    target[0] = 2.0f;
    target[(bytes - page) / sizeof(float)] = 4.25f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float));
    const float input = 2.0f;
    int ok = x && out && ds4_gpu_tensor_write(x, 0, &input, sizeof(input)) &&
             ds4_gpu_set_model_map_range(aux, page, 0, page, sizeof(float));
    /* More layer switches than the view table can retain. The auxiliary
     * model must survive both single-span and disjoint target replacements. */
    for (uint64_t i = 0; i < 4100 && ok; i++) {
        const uint64_t offset = i * page;
        ok = ds4_gpu_set_model_map_spans(target, bytes, &offset, &page, 1, sizeof(float));
    }
    float actual = NAN;
    ok = ok && ds4_gpu_matmul_f32_tensor(out, target, bytes, bytes - page, 1, 1, x, 1) &&
         ds4_gpu_tensor_read(out, 0, &actual, sizeof(actual)) && actual == 8.5f;
    const uint64_t offsets[] = {0, 2 * page}, sizes[] = {page, page};
    ok = ok && ds4_gpu_set_model_map_spans(target, bytes, offsets, sizes, 2, sizeof(float));
    ok = ok && ds4_gpu_matmul_f32_tensor(out, target, bytes, 0, 1, 1, x, 1) &&
         ds4_gpu_tensor_read(out, 0, &actual, sizeof(actual)) && actual == 4.0f;
    ok = ok && ds4_gpu_matmul_f32_tensor(out, aux, page, 0, 1, 1, x, 1) &&
         ds4_gpu_tensor_read(out, 0, &actual, sizeof(actual)) && actual == 6.5f;
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    munmap(target, bytes);
    munmap(aux, page);
    fprintf(stderr, "Metal SSD bounded target views and auxiliary mapping: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int check_batch_cache(void *model, uint64_t bytes, uint64_t expert) {
    enum { T = 257 };
    const uint64_t tensor = E * expert;
    const uint64_t row = D / 256 * sizeof(iq2_block);
    const uint64_t down_row = H / 256 * sizeof(iq2_block);
    const size_t xb = T * D * sizeof(float), mb = T * N * H * sizeof(float);
    const size_t ob = T * D * sizeof(float), ib = T * N * sizeof(int32_t);
    float *x = malloc(xb), *weights = malloc(ib), *ref = malloc(ob), *got = malloc(ob);
    float *ref_mid = malloc(mb), *got_mid = malloc(mb);
    int32_t *ids = malloc(ib);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xb), *it = ds4_gpu_tensor_alloc(ib);
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(ib), *gate = ds4_gpu_tensor_alloc(mb);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mb), *mid = ds4_gpu_tensor_alloc(mb);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(T * N * D * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ob);
    int ok = x && weights && ref && got && ref_mid && got_mid && ids &&
             xt && it && wt && gate && up && mid && down && out;
    if (!ok) goto done;
    for (int i = 0; i < T * D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 32.0f;
    for (int t = 0; t < T; t++) for (int k = 0; k < N; k++) {
        ids[t * N + k] = (t * 13 + k * 17) % E;
        weights[t * N + k] = (k + 1) / 36.0f;
    }
    ok = ds4_gpu_tensor_write(xt, 0, x, xb) &&
         ds4_gpu_tensor_write(it, 0, ids, ib) &&
         ds4_gpu_tensor_write(wt, 0, weights, ib);
    const uint32_t counts[] = {1, 2, 5, 6, 7, 16, 31, 32, 43, 114, 255, 256, 257, 16, 43};
    for (int quality = 0; quality < 2 && ok; quality++) {
        ds4_gpu_set_quality(quality);
        ds4_gpu_set_streaming_expert_cache_budget(E);
        for (size_t c = 0; c < sizeof(counts) / sizeof(*counts) && ok; c++) {
            const uint32_t n = counts[c];
            if (c + 1 == sizeof(counts) / sizeof(*counts))
                ds4_gpu_set_streaming_expert_cache_budget(E - 1);
            bool half[2] = {false, false};
            for (int streamed = 0; streamed < 2 && ok; streamed++) {
                ok = ds4_gpu_tensor_fill_f32(mid, NAN, T * N * H) &&
                     ds4_gpu_tensor_fill_f32(out, NAN, T * D) &&
                     ds4_gpu_begin_commands() && ds4_gpu_routed_moe_batch_tensor(
                        out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                        16, 16, expert, row, expert, down_row, D, H, D,
                        it, wt, E, N, 7.0f, xt, 3 + c % 2, n, &half[streamed],
                        !streamed) && ds4_gpu_end_commands() &&
                     ds4_gpu_tensor_read(out, 0, streamed ? got : ref, n * D * sizeof(float)) &&
                     ds4_gpu_tensor_read(mid, 0, streamed ? got_mid : ref_mid,
                        n * N * H * (half[streamed] ? sizeof(uint16_t) : sizeof(float)));
            }
            if (!ok || half[0] != half[1] || memcmp(ref, got, n * D * sizeof(float)) ||
                memcmp(ref_mid, got_mid, n * N * H *
                    (half[0] ? sizeof(uint16_t) : sizeof(float)))) {
                fprintf(stderr, "SSD batch mismatch tokens=%u quality=%d\n", n, quality);
                ok = 0;
            }
            if (c == 0 && ds4_gpu_stream_expert_cache_current_count() == 0) {
                fprintf(stderr, "SSD batch did not populate the expert cache\n");
                ok = 0;
            }
            for (size_t i = 0; i < n * D && ok; i++) if (!isfinite(got[i])) ok = 0;
        }
    }
    ds4_gpu_set_quality(false);
done:
    free(x); free(weights); free(ref); free(got); free(ref_mid); free(got_mid); free(ids);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    fprintf(stderr, "Metal SSD IQ2 batch cache exact outputs: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--full-glm-shape")) {
        D = 6144;
        H = 2048;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--full-glm-shape]\n", argv[0]);
        return 1;
    }
    const uint64_t row = D / 256 * sizeof(iq2_block);
    const uint64_t down_row = H / 256 * sizeof(iq2_block);
    const uint64_t expert = H * row;
    const uint64_t tensor = E * expert;
    const size_t bytes = 3 * tensor;
    FILE *file = tmpfile();
    if (!file || ftruncate(fileno(file), bytes)) return 1;
    void *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fileno(file), 0);
    if (model == MAP_FAILED) return 1;
    iq2_block *block = model;
    for (size_t i = 0; i < bytes / sizeof(*block); i++) {
        block[i].d = 0x1400;
        for (size_t j = 0; j < sizeof(block[i].qs); j++)
            block[i].qs[j] = random_u32();
    }
    int ok = msync(model, bytes, MS_SYNC) == 0 && ds4_gpu_init();
    ds4_gpu_set_quality(false);
    ds4_gpu_set_glm_model(true);
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(16);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(expert * 3);
    ok = ok && ds4_gpu_set_model_map(model, bytes) &&
         ds4_gpu_set_model_fd(fileno(file));
    float x[D], weights[N], reference[D], actual[D], first_pass[STEPS][D];
    int32_t ids[N];
    for (int i = 0; i < D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int i = 0; i < N; i++) weights[i] = (i + 1) / 36.0f;
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(N * H * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(N * H * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(N * H * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(N * D * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    ok = ok && xt && it && wt && gate && up && mid && down && out &&
         ds4_gpu_tensor_write(xt, 0, x, sizeof(x)) &&
         ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights));
    /* Move the missing slots through all eight positions. Every fourth
     * route is all-hit; the cache is too small to retain the whole sequence. */
    for (int step = 0; step < STEPS * 2 && ok; step++) {
        const int turn = step % STEPS;
        const uint32_t budget = step < STEPS ? 16u : 24u;
        if (step == STEPS) {
            ds4_gpu_set_streaming_expert_cache_budget(budget);
        }
        if (turn % 4 != 3) {
            for (int i = 0; i < N; i++) {
                const int slot = (i + turn) % N;
                ids[slot] = i < 4 ? i : 4 + (turn * 4 + i) % (E - 4);
            }
        }
        ok = ds4_gpu_tensor_write(it, 0, ids, sizeof(ids));
        for (int streamed = 0; streamed < 2 && ok; streamed++) {
            ok = ds4_gpu_tensor_fill_f32(mid, NAN, N * H) &&
                 ds4_gpu_tensor_fill_f32(out, NAN, D) &&
                 ds4_gpu_begin_commands() &&
                 ds4_gpu_routed_moe_one_tensor(
                    out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                    16, 16, expert, row, expert, down_row, D, H, D,
                    it, wt, E, N, 7.0f, xt, NULL, 3, !streamed) &&
                 ds4_gpu_end_commands() &&
                 ds4_gpu_tensor_read(out, 0, streamed ? actual : reference,
                                      sizeof(actual));
        }
        for (int i = 0; i < D && ok; i++) {
            if (!isfinite(actual[i]) || !isfinite(reference[i]) ||
                fabsf(actual[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i]))) {
                fprintf(stderr, "SSD expert mismatch step=%d element=%d ref=%g actual=%g\n",
                        step, i, reference[i], actual[i]);
                ok = 0;
            }
        }
        if (step < STEPS) memcpy(first_pass[turn], actual, sizeof(actual));
        else if (memcmp(first_pass[turn], actual, sizeof(actual))) {
            fprintf(stderr, "SSD cache capacity changed output at step %d\n", turn);
            ok = 0;
        }
        const uint32_t cached = ds4_gpu_stream_expert_cache_current_count();
        if (cached == 0 || cached > budget) ok = 0;
        if (turn == STEPS / 2) ds4_gpu_stream_expert_cache_reset_route_hotness();
    }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    if (ok) ok = check_batch_cache(model, bytes, expert);
    ds4_gpu_print_memory_report("SSD expert test");
    if (ok) ok = check_mapping_lifetime();
    ds4_gpu_cleanup();
    munmap(model, bytes);
    fclose(file);
    fprintf(stderr, "Metal SSD IQ2 eight-expert eviction: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
