#define _DARWIN_C_SOURCE
/* Model-free bitwise oracle for Q4 output/HC fusion at its automatic shapes. */
#include "ds4_gpu.h"
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static void check(int ok, const char *label) {
    if (!ok) { fprintf(stderr, "Q4 HC FAIL: %s\n", label); exit(1); }
}
static uint32_t random_word(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u; return *state;
}
static ds4_gpu_tensor *values(uint64_t count, uint32_t seed) {
    float *host = malloc(count * sizeof(float)); check(host != NULL, "host allocation");
    for (uint64_t i = 0; i < count; ++i)
        host[i] = ((int32_t)(random_word(&seed) % 1025u) - 512) / 2048.f;
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(count * sizeof(float));
    check(tensor != NULL && ds4_gpu_tensor_write(tensor, 0, host, count * sizeof(float)), "tensor upload");
    free(host); return tensor;
}
static void same(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t count) {
    uint32_t *x = malloc(count * 4u), *y = malloc(count * 4u);
    check(x != NULL && y != NULL, "read allocation");
    check(ds4_gpu_tensor_read(a, 0, x, count * 4u) &&
          ds4_gpu_tensor_read(b, 0, y, count * 4u), "readback");
    for (uint64_t i = 0; i < count; ++i) {
        if (x[i] != y[i] || (x[i] & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "Q4 HC mismatch word=%llu baseline=%08x candidate=%08x\n",
                    (unsigned long long)i, x[i], y[i]); exit(1);
        }
    }
    free(y); free(x);
}
static uint32_t parse_tokens(int argc, char **argv) {
    if (argc == 1) return 512u;
    check(argc == 3 && strcmp(argv[1], "--tokens") == 0,
          "usage: test_metal_q4_hc [--tokens 512..8192 (multiple of 32)]");
    errno = 0;
    char *end = NULL;
    const unsigned long n = strtoul(argv[2], &end, 10);
    check(errno == 0 && end != argv[2] && *end == '\0' &&
          n >= 512u && n <= 8192u && (n % 32u) == 0u,
          "invalid token count");
    return (uint32_t)n;
}
int main(int argc, char **argv) {
    enum { K = 8192, M = 4096, G = 8, R = 1024, HC = 4, GUARD = 64 };
    const uint32_t N = parse_tokens(argc, argv);
    const uint64_t a_bytes = (uint64_t)K * M / 256u * 144u;
    const uint64_t b_bytes = a_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t model_bytes = (a_bytes + b_bytes + page - 1u) & ~(page - 1u);
    void *model = NULL;
    check(posix_memalign(&model, page, model_bytes) == 0, "model allocation");
    uint32_t rng = 19u;
    for (uint64_t off = 0; off < a_bytes + b_bytes; off += 144u) {
        uint8_t *block = (uint8_t *)model + off;
        const uint16_t d = 0x1800u, dmin = 0x1400u;
        memcpy(block, &d, 2u); memcpy(block + 2u, &dmin, 2u);
        for (uint32_t i = 4u; i < 144u; ++i) block[i] = (uint8_t)random_word(&rng);
    }
    check(ds4_gpu_init(), "Metal initialization");
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        puts("SKIP: Q4 HC automatic prefill shape is pre-M5 only");
        ds4_gpu_cleanup(); free(model); return 0;
    }
    ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_test_set_flags(0u);
    check(ds4_gpu_set_model_map(model, model_bytes), "model map");
    ds4_gpu_tensor *heads = values((uint64_t)N * G * M, 3u);
    ds4_gpu_tensor *residual = values((uint64_t)N * HC * M, 5u);
    ds4_gpu_tensor *split = values((uint64_t)N * 24u, 7u);
    ds4_gpu_tensor *low = values((uint64_t)N * K, 9u);
    ds4_gpu_tensor *group = values((uint64_t)N * K, 11u);
    ds4_gpu_tensor *tmp = values((uint64_t)N * K, 13u);
    ds4_gpu_tensor *out = values((uint64_t)N * M, 17u);
    ds4_gpu_tensor *baseline = values((uint64_t)N * HC * M, 21u);
    ds4_gpu_tensor *guarded = values((uint64_t)N * HC * M + 2u * GUARD, 23u);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_view(guarded, GUARD * 4u, (uint64_t)N * HC * M * 4u);
    check(candidate != NULL, "guarded output view");
    uint32_t guards_before[2 * GUARD], guards_after[2 * GUARD];
    check(ds4_gpu_tensor_read(guarded, 0, guards_before, GUARD * 4u) &&
          ds4_gpu_tensor_read(guarded, (GUARD + (uint64_t)N * HC * M) * 4u,
                              guards_before + GUARD, GUARD * 4u), "guard snapshot");
    check(setenv("DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS", "1", 1) == 0, "baseline flags");
    check(ds4_gpu_attention_output_q4_K_batch_tensor(out, low, group, tmp,
          model, model_bytes, 0, a_bytes, 12, M, R, G, M, heads, N) == 1, "baseline output projection");
    check(ds4_gpu_hc_expand_split_tensor(baseline, out, residual, split, M, HC), "baseline HC");
    check(unsetenv("DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS") == 0, "automatic flags");
    check(ds4_gpu_attention_output_q4_K_batch_hc_tensor(out, candidate, residual, split,
          low, group, tmp, model, model_bytes, 0, a_bytes, 12, M, R, G, M,
          heads, N, HC) == 1, "automatic prefill HC fusion");
    same(baseline, candidate, (uint64_t)N * HC * M);
    check(ds4_gpu_tensor_read(guarded, 0, guards_after, GUARD * 4u) &&
          ds4_gpu_tensor_read(guarded, (GUARD + (uint64_t)N * HC * M) * 4u,
                              guards_after + GUARD, GUARD * 4u), "guard readback");
    check(memcmp(guards_before, guards_after, sizeof(guards_before)) == 0, "output guards");
    check(ds4_gpu_attention_output_q4_K_batch_hc_tensor(out, candidate, residual, split,
          low, group, tmp, model, model_bytes, 0, a_bytes, 12, M, R, G, M,
          heads, 33u, HC) == 0, "partial-token fallback");
    check(ds4_gpu_attention_output_q4_K_batch_hc_tensor(out, candidate, residual, split,
          low, group, tmp, model, model_bytes, 0, a_bytes, 12, M, R, G, M,
          heads, 8193u, HC) == 0, "above-limit partial-token fallback");
    check(setenv("DS4_METAL_REQUIRE_Q4_BATCH_ATTN_OUT_HC_FUSION", "1", 1) == 0,
          "require fusion for upper-bound check");
    check(ds4_gpu_attention_output_q4_K_batch_hc_tensor(out, candidate, residual, split,
          low, group, tmp, model, model_bytes, 0, a_bytes, 12, M, R, G, M,
          heads, 8224u, HC) == -1, "above-limit aligned-token refusal");
    check(unsetenv("DS4_METAL_REQUIRE_Q4_BATCH_ATTN_OUT_HC_FUSION") == 0,
          "clear fusion requirement");
    same(baseline, candidate, (uint64_t)N * HC * M);
    ds4_gpu_tensor *decode_x = ds4_gpu_tensor_view(low, 0, K * 4u);
    ds4_gpu_tensor *decode_out = ds4_gpu_tensor_view(out, 0, M * 4u);
    ds4_gpu_tensor *decode_base = ds4_gpu_tensor_view(baseline, 0, HC * M * 4u);
    ds4_gpu_tensor *decode_cand = ds4_gpu_tensor_view(candidate, 0, HC * M * 4u);
    ds4_gpu_tensor *decode_res = ds4_gpu_tensor_view(residual, 0, HC * M * 4u);
    ds4_gpu_tensor *decode_split = ds4_gpu_tensor_view(split, 0, 24u * 4u);
    check(ds4_gpu_matmul_quant_tensor(decode_out, model, model_bytes, a_bytes, 12,
                                      K, M, decode_x, 1), "decode baseline projection");
    check(ds4_gpu_hc_expand_split_tensor(decode_base, decode_out, decode_res, decode_split, M, HC), "decode baseline HC");
    check(ds4_gpu_matmul_q4_K_hc_expand_available() &&
          ds4_gpu_matmul_q4_K_hc_expand_tensor(decode_cand, decode_out, model, model_bytes,
              a_bytes, K, M, decode_x, decode_res, decode_split, M, HC), "decode fusion");
    same(decode_base, decode_cand, HC * M);
    ds4_gpu_tensor *all[] = {decode_x,decode_out,decode_base,decode_cand,decode_res,decode_split,
        candidate,guarded,baseline,out,tmp,group,low,split,residual,heads};
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); ++i) ds4_gpu_tensor_free(all[i]);
    ds4_gpu_cleanup(); free(model);
    printf("PASS: Q4 HC automatic prefill N%u and decode, bitwise outputs, guards, "
           "partial-token fallback, upper-bound refusal\n", N);
    return 0;
}
