#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { STEPS = 48 };
static int D = 256, H = 512, E = 256, N = 8;
static uint32_t quant_type = 16;
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint8_t e, qs[16]; } mxfp4_block;
static uint64_t block_bytes = sizeof(iq2_block);
static uint32_t block_values = 256;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int check_streaming_table_admission(void) {
    const char *flags[] = {"DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO",
        "DS4_METAL_ENABLE_Q4_EXPERT_TABLE", "DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO"};
    int ok = 1;
    ds4_gpu_set_ssd_streaming(true);
    for (size_t i = 0; i < sizeof(flags) / sizeof(*flags); i++) {
        if (getenv(flags[i])) {
            fprintf(stderr, "Run the SSD admission test without Q4 table overrides\n");
            return 0;
        }
    }
    ok = ds4_gpu_pro_q4_expert_table_auto_available() == 0;
    for (size_t i = 0; i < sizeof(flags) / sizeof(*flags); i++) {
        if (setenv(flags[i], "1", 1)) return 0;
        ok = ok && ds4_gpu_pro_q4_expert_table_auto_available() == 0;
        if (unsetenv(flags[i])) return 0;
    }
    fprintf(stderr, "Metal SSD excludes persistent full-expert tables: %s\n", ok ? "PASS" : "FAIL");
    return ok;
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
    const uint64_t row = D / block_values * block_bytes;
    const uint64_t down_row = H / block_values * block_bytes;
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
                        quant_type, quant_type, expert, row, expert, down_row, D, H, D,
                        it, wt, E, N, 7.0f, xt, 3 + c % 2, n, &half[streamed],
                        !streamed) && ds4_gpu_end_commands() &&
                     ds4_gpu_tensor_read(out, 0, streamed ? got : ref, n * D * sizeof(float)) &&
                     ds4_gpu_tensor_read(mid, 0, streamed ? got_mid : ref_mid,
                        n * N * H * (half[streamed] ? sizeof(uint16_t) : sizeof(float)));
                if (c == 0 && !streamed && quant_type != 16 && !quality &&
                    ds4_gpu_stream_expert_cache_current_count() != 0) {
                    fprintf(stderr, "Resident type=%u single-row batch populated the SSD cache\n", quant_type);
                    ok = 0;
                }
            }
            if (!ok || half[0] != half[1] || memcmp(ref, got, n * D * sizeof(float)) ||
                memcmp(ref_mid, got_mid, n * N * H *
                    (half[0] ? sizeof(uint16_t) : sizeof(float)))) {
                fprintf(stderr, "SSD batch mismatch tokens=%u quality=%d\n", n, quality);
                ok = 0;
            }
            /* The unfused six-expert quality path reads the layer directly. */
            if (c == 0 && (quant_type == 16 || !quality) &&
                ds4_gpu_stream_expert_cache_current_count() == 0) {
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
    fprintf(stderr, "Metal SSD type=%u batch cache exact outputs: %s\n", quant_type, ok ? "PASS" : "FAIL");
    return ok;
}

static uint64_t memory_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return UINT64_MAX;
    return info.phys_footprint;
}

static int check_seed_release(void *model, uint64_t bytes, uint64_t expert) {
    int32_t ids[E];
    for (int i = 0; i < E; i++) ids[i] = i;
    const ds4_gpu_stream_expert_table table = {
        .model_map = model, .model_size = bytes, .layer = 3, .n_total_expert = E,
        .gate_offset = 0, .up_offset = E * expert, .down_offset = 2 * E * expert,
        .gate_expert_bytes = expert, .down_expert_bytes = expert,
    };
    /* Repeatedly replace a small seeded cache, without a caller-owned ObjC
     * pool. Logical entry counts alone cannot detect retained Metal buffers. */
    uint64_t baseline = 0;
    for (unsigned pass = 0; pass < 10; pass++) {
        ds4_gpu_set_streaming_expert_cache_budget(E);
        if (!ds4_gpu_begin_commands() ||
            !ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(&table, ids, NULL, E) ||
            !ds4_gpu_end_commands() || ds4_gpu_stream_expert_cache_current_count() != (uint32_t)E)
            return 0;
        ds4_gpu_set_streaming_expert_cache_budget(0);
        if (ds4_gpu_stream_expert_cache_current_count() != 0) return 0;
        const uint64_t footprint = memory_footprint();
        if (footprint == UINT64_MAX) return 0;
        if (pass == 1) baseline = footprint;
        fprintf(stderr, "Metal SSD seed/release %u: %.2f MiB footprint\n",
                pass, footprint / 1048576.0);
        if (pass > 1 && footprint > baseline + 3 * bytes + (UINT64_C(32) << 20)) {
            fprintf(stderr, "Metal SSD seed/release retains discarded buffers\n");
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--table-admission")) {
        const int ok = ds4_gpu_init() && check_streaming_table_admission();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    } else if (argc == 2 && !strcmp(argv[1], "--full-glm-shape")) {
        D = 6144;
        H = 2048;
    } else if (argc == 2 && !strcmp(argv[1], "--q4")) {
        E = 384;
        N = 6;
        quant_type = 12;
        block_bytes = sizeof(q4_block);
    } else if (argc == 2 && !strcmp(argv[1], "--mxfp4")) {
        N = 6;
        quant_type = 39;
        block_bytes = sizeof(mxfp4_block);
        block_values = 32;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--full-glm-shape | --q4 | --mxfp4 | --table-admission]\n", argv[0]);
        return 1;
    }
    const uint64_t row = D / block_values * block_bytes;
    const uint64_t down_row = H / block_values * block_bytes;
    const uint64_t expert = H * row;
    const uint64_t tensor = E * expert;
    const size_t bytes = 3 * tensor;
    FILE *file = tmpfile();
    if (!file || ftruncate(fileno(file), bytes)) return 1;
    void *model = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fileno(file), 0);
    if (model == MAP_FAILED) return 1;
    for (size_t i = 0; i < bytes / block_bytes; i++) {
        if (quant_type == 12) {
            q4_block *block = (q4_block *)model + i;
            block->d = 0x0400;
            block->dmin = 0x1000;
            for (size_t j = 0; j < sizeof(block->scales); j++)
                block->scales[j] = random_u32();
            for (size_t j = 0; j < sizeof(block->qs); j++)
                block->qs[j] = random_u32();
        } else if (quant_type == 39) {
            mxfp4_block *block = (mxfp4_block *)model + i;
            block->e = 118 + random_u32() % 5;
            for (size_t j = 0; j < sizeof(block->qs); j++)
                block->qs[j] = random_u32();
        } else {
            iq2_block *block = (iq2_block *)model + i;
            block->d = 0x1400;
            for (size_t j = 0; j < sizeof(block->qs); j++)
                block->qs[j] = random_u32();
        }
    }
    int ok = msync(model, bytes, MS_SYNC) == 0 && ds4_gpu_init();
    if (ok) ok = check_streaming_table_admission();
    ds4_gpu_set_quality(false);
    ds4_gpu_set_glm_model(quant_type == 16);
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
                    quant_type, quant_type, expert, row, expert, down_row, D, H, D,
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
    if (ok && D == 256) ok = check_seed_release(model, bytes, expert);
    ds4_gpu_print_memory_report("SSD expert test");
    if (ok) ok = check_mapping_lifetime();
    ds4_gpu_cleanup();
    munmap(model, bytes);
    fclose(file);
    fprintf(stderr, "Metal SSD type=%u %d-expert eviction: %s\n", quant_type, N, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
