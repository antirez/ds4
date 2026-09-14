// SPDX-License-Identifier: MIT
// Model-free runtime oracle for Q-b token pairing, including the host opt-out.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
enum { K = 1024, M = 32768, MAX_TOKENS = 9, GUARD = 64 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
_Static_assert(sizeof(q4_block) == 144, "Q4_K ABI");
static const uint32_t poison = 0x7fc12345u;
static void require(bool ok, const char *msg) {
    if (!ok) { fprintf(stderr, "Q4 Q-b runtime FAIL: %s\n", msg); exit(1); }
}
static uint32_t random_u32(uint32_t *seed) { return *seed = *seed * 1664525u + 1013904223u; }

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        require(dev != nil, "no Metal device; run with GPU access");
        if (![dev.name isEqualToString:@"Apple M1"] && ![dev.name hasPrefix:@"Apple M1 "]) {
            fprintf(stderr, "SKIP: Q-b token-pair production default is M1-only; use the kernel oracle on other GPUs.\n");
            return 0;
        }
        require(unsetenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC") == 0, "classic setup");
        require(ds4_gpu_init(), "Metal backend initialization");
        const size_t page = (size_t)getpagesize();
        const size_t weight_bytes = (size_t)M * (K/256) * sizeof(q4_block);
        const size_t model_bytes = weight_bytes + 2*page;
        void *model = NULL;
        require(posix_memalign(&model, page, model_bytes) == 0, "model fixture allocation");
        memset(model, 0xa5, model_bytes);
        q4_block *w = (q4_block *)((char *)model + page);
        uint32_t seed = 731;
        for (size_t i = 0; i < weight_bytes/sizeof(*w); ++i) {
            w[i].d = 0x2000u + (random_u32(&seed) & 0x3ffu);
            w[i].dmin = i % 7 ? 0x1800u + (random_u32(&seed) & 0x3ffu) : 0;
            for (unsigned j = 0; j < 12; ++j) w[i].scales[j] = random_u32(&seed) >> 24;
            for (unsigned j = 0; j < 128; ++j) w[i].qs[j] = random_u32(&seed) >> 24;
        }
        const uint64_t span_offset = page, span_bytes = weight_bytes;
        require(ds4_gpu_set_model_map_spans(model, model_bytes, &span_offset, &span_bytes, 1, weight_bytes),
                "model range registration");
        const size_t x_count = (size_t)K * MAX_TOKENS;
        const size_t y_count = (size_t)M * MAX_TOKENS;
        const size_t total = y_count + 2*GUARD;
        float *input = malloc(x_count*sizeof(float));
        uint32_t *expected = malloc(total*sizeof(uint32_t)), *got = malloc(total*sizeof(uint32_t));
        require(input && expected && got, "host buffer allocation");
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count*sizeof(float));
        ds4_gpu_tensor *base = ds4_gpu_tensor_alloc(total*sizeof(float));
        ds4_gpu_tensor *out = base ? ds4_gpu_tensor_view(base, GUARD*sizeof(float), y_count*sizeof(float)) : NULL;
        require(x && base && out, "device buffer allocation");
        const char *values[] = {"1", NULL, "0", ""};
        unsigned cases = 0;
        for (int ssd = 0; ssd < 2; ++ssd) for (int quality = 0; quality < 2; ++quality) {
            ds4_gpu_set_ssd_streaming(ssd);
            ds4_gpu_set_quality(quality);
            for (int tokens = 1; tokens <= MAX_TOKENS; ++tokens) {
                for (size_t i = 0; i < x_count; ++i)
                    input[i] = i % 13 ? ((int)(random_u32(&seed) % 1025u) - 512) / 1024.f : -0.f;
                require(ds4_gpu_tensor_write(x, 0, input, x_count*sizeof(float)), "activation write");
                for (unsigned arm = 0; arm < 4; ++arm) {
                    require((values[arm] ? setenv("DS4_METAL_DISABLE_Q4_QB_TOKEN_PAIR", values[arm], 1) :
                                          unsetenv("DS4_METAL_DISABLE_Q4_QB_TOKEN_PAIR")) == 0, "rollback setup");
                    for (size_t i = 0; i < total; ++i) got[i] = poison;
                    require(ds4_gpu_tensor_write(base, 0, got, total*sizeof(uint32_t)), "output poison");
                    require(ds4_gpu_begin_commands(), "begin commands");
                    const int ok = ds4_gpu_matmul_quant_tensor(out, model, model_bytes, page, 12, K, M, x, tokens);
                    const int ended = ds4_gpu_end_commands();
                    require(ok && ended, "matmul dispatch");
                    require(ds4_gpu_tensor_read(base, 0, got, total*sizeof(uint32_t)), "output read");
                    for (size_t i = 0; i < total; ++i) {
                        if (i < GUARD || i >= GUARD + (size_t)tokens*M)
                            require(got[i] == poison, "output guard/inactive token overwritten");
                        else require((got[i] & 0x7f800000u) != 0x7f800000u, "nonfinite/unwritten output");
                    }
                    if (arm == 0) memcpy(expected, got, total*sizeof(uint32_t));
                    else require(memcmp(expected, got, total*sizeof(uint32_t)) == 0, "output not bitwise equal");
                    ++cases;
                }
            }
        }
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(base); ds4_gpu_tensor_free(x);
        ds4_gpu_cleanup();
        free(model); free(input); free(expected); free(got);
        printf("PASS: %u Q4 Q-b runtime cases (default/1/0/empty, 1..9 tokens, quality, SSD spans, guards).\n", cases);
    }
    return 0;
}
