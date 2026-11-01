/* Budget math for DeepSeek V4.1 images: default matches the checkpoint (1024), larger
 * budgets and native resolution work, decoded-image limits clamp instead of failing,
 * and prompt layouts accept grids far beyond 1024 tokens. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ds4_image.h"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    /* 1722x1008: native grid = 41x24 -> 24*42+2 = 1010 tokens, inside the default. */
    CHECK(ds4_image_deepseek_token_cost(1722, 1008, 0, 1) == 1010, "1722x1008 default != 1010");
    CHECK(ds4_image_deepseek_token_cost(1722, 1008, DS4_IMAGE_BUDGET_NATIVE, 1) == 1010,
          "1722x1008 native != 1010");
    /* 1080p is shrunk at the default budget, not at native. */
    const uint32_t d1080 = ds4_image_deepseek_token_cost(1920, 1080, 0, 1);
    const uint32_t n1080 = ds4_image_deepseek_token_cost(1920, 1080, DS4_IMAGE_BUDGET_NATIVE, 1);
    CHECK(d1080 <= 1024 && d1080 > 900, "1080p default %u", d1080);
    CHECK(n1080 > 1024, "1080p native %u should exceed 1024", n1080);
    /* 4K: default <= 1024, 4096 budget <= 4096 and > 3000, native much larger. */
    const uint32_t k4d = ds4_image_deepseek_token_cost(3840, 2160, 0, 1);
    const uint32_t k4b = ds4_image_deepseek_token_cost(3840, 2160, 4096, 1);
    const uint32_t k4n = ds4_image_deepseek_token_cost(3840, 2160, DS4_IMAGE_BUDGET_NATIVE, 1);
    CHECK(k4d <= 1024, "4K default %u", k4d);
    CHECK(k4b <= 4096 && k4b > 3000, "4K@4096 %u", k4b);
    CHECK(k4n > k4b, "4K native %u <= 4K@4096 %u", k4n, k4b);
    /* Huge image at native clamps to the decoded-image limits instead of failing. */
    const uint32_t huge = ds4_image_deepseek_token_cost(16000, 16000, DS4_IMAGE_BUDGET_NATIVE, 1);
    CHECK(huge > 0 && huge < 50000, "16000^2 native clamp %u", huge);
    /* V4 (non-4.1) keeps its 381 default. */
    CHECK(ds4_image_deepseek_token_cost(1920, 1080, 0, 0) <= 384, "V4 default exceeds 384");

    /* Real preprocessing at native: synthetic 3840x2160 gray image. */
    ds4_image img = {0};
    img.width = 3840; img.height = 2160;
    img.rgb = calloc((size_t)img.width * img.height * 3, 1);
    ds4_deepseek4_image_patches p = {0};
    char err[256] = {0};
    int ok = ds4_image_preprocess_deepseek41_budget(&p, &img, DS4_IMAGE_BUDGET_NATIVE, err, sizeof err);
    CHECK(ok, "preprocess native failed: %s", err);
    if (ok) {
        const uint32_t tokens = p.llm_grid_height * (p.llm_grid_width + 1) + 2;
        CHECK(tokens == k4n, "preprocess tokens %u != cost %u", tokens, k4n);
        ds4_deepseek4_image_layout lay = {0};
        CHECK(ds4_deepseek41_image_layout_build(&lay, p.llm_grid_height, p.llm_grid_width, err, sizeof err),
              "layout build failed for %u tokens: %s", tokens, err);
        ds4_deepseek4_image_layout_free(&lay);
        printf("4K native: grid %ux%u patches, %ux%u llm, %u tokens\n",
               p.grid_width, p.grid_height, p.llm_grid_width, p.llm_grid_height, tokens);
        ds4_deepseek4_image_patches_free(&p);
    }
    /* Default-budget preprocessing is unchanged by the refactor. */
    ok = ds4_image_preprocess_deepseek41(&p, &img, err, sizeof err);
    CHECK(ok && p.llm_grid_height * (p.llm_grid_width + 1) + 2 == k4d, "default preprocess mismatch");
    if (ok) ds4_deepseek4_image_patches_free(&p);
    free(img.rgb);

    /* Fingerprint re-keying differs per grid and is deterministic. */
    uint8_t a[32] = {1}, b[32] = {1}, c[32] = {1};
    ds4_image_fingerprint_with_grid(a, 72, 123);
    ds4_image_fingerprint_with_grid(b, 72, 123);
    ds4_image_fingerprint_with_grid(c, 72, 122);
    CHECK(!memcmp(a, b, 32) && memcmp(a, c, 32), "fingerprint re-key not deterministic/distinct");

    printf("1080p default=%u native=%u | 4K default=%u @4096=%u native=%u | 16000^2 native=%u\n",
           d1080, n1080, k4d, k4b, k4n, huge);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails != 0;
}
