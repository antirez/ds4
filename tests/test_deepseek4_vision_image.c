#include "ds4_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_layout(
        uint32_t height,
        uint32_t width,
        uint32_t start,
        const uint8_t *types,
        uint32_t type_count,
        const uint32_t *perm,
        uint32_t perm_count) {
    char error[160] = {0};
    ds4_deepseek4_image_layout layout = {0};
    if (!ds4_deepseek4_image_layout_build(
            &layout, height, width, start, error, sizeof(error))) {
        fprintf(stderr, "layout failed: %s\n", error);
        return 0;
    }
    int ok = layout.token_count == type_count &&
             layout.image_count == perm_count &&
             memcmp(layout.types, types, type_count) == 0 &&
             memcmp(layout.perm, perm,
                    (size_t)perm_count * sizeof(*perm)) == 0;
    if (!ok) fprintf(stderr, "layout differs for %ux%u at %u\n",
                     height, width, start);
    ds4_deepseek4_image_layout_free(&layout);
    return ok;
}

static int check_span_parser(void) {
    const int vocab = 100;
    const int tokens[] = {
        7,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE_NEWLINE,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        8,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        9,
    };
    uint32_t cursor = 0, block = 0, start = 0, end = 0;
    int found = ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end);
    if (found != 1 || block != 1 || start != 3 || end != 6 || cursor != 7)
        return 0;
    found = ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end);
    if (found != 1 || block != 8 || start != 8 || end != 10 || cursor != 11)
        return 0;
    if (ds4_deepseek4_next_image_span(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            &cursor, &block, &start, &end) != 0) return 0;

    static const int malformed[][4] = {
        {100 + DS4_DEEPSEEK4_IMAGE_PAD, 4, 5, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START, 4,
         100 + DS4_DEEPSEEK4_IMAGE_END, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_END, 6},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE, 6, 7},
        {100 + DS4_DEEPSEEK4_IMAGE_START,
         100 + DS4_DEEPSEEK4_IMAGE_END + 1, 6, 7},
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++) {
        cursor = 0;
        if (ds4_deepseek4_next_image_span(
                malformed[i], 4, vocab, &cursor,
                &block, &start, &end) != -1) return 0;
    }

    uint32_t chunk = 0;
    if (!ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            0, 5, &chunk) || chunk != 1) return 0;
    if (!ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            1, 6, &chunk) || chunk != 6) return 0;
    if (ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            1, 5, &chunk)) return 0;
    if (ds4_deepseek4_prefill_chunk(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            4, 8, &chunk)) return 0;
    return 1;
}

static int check_attention_bounds(void) {
    const int vocab = 100;
    const int tokens[] = {
        7,
        vocab + DS4_DEEPSEEK4_IMAGE_PAD,
        vocab + DS4_DEEPSEEK4_IMAGE_START,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE,
        vocab + DS4_DEEPSEEK4_IMAGE_END,
        8,
    };
    uint32_t bounds[sizeof(tokens) / sizeof(tokens[0]) * 2u];
    if (!ds4_deepseek4_attention_bounds(
            tokens, sizeof(tokens) / sizeof(tokens[0]), vocab,
            10u, 20u, 4u, bounds)) return 0;
    static const uint32_t expected[][2] = {
        {7, 10}, {8, 11}, {9, 18}, {10, 18}, {11, 18},
        {12, 18}, {12, 18}, {12, 18}, {12, 18}, {16, 19},
    };
    return memcmp(bounds, expected, sizeof(expected)) == 0;
}

int main(void) {
    static const uint8_t types_a[] = {
        1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4,
    };
    static const uint32_t perm_a[] = {0, 3, 1, 4, 2, 5};
    static const uint8_t types_b[] = {
        1, 1, 0, 2, 2, 2, 2, 3, 3, 2, 1, 2, 1, 3, 1, 4,
    };
    static const uint32_t perm_b[] = {0, 2, 1, 3, 4, 5};
    static const uint8_t types_c[] = {0, 2, 1, 3, 1, 4};
    static const uint32_t perm_c[] = {0};
    if (!check_layout(2, 3, 0, types_a, sizeof(types_a),
                      perm_a, sizeof(perm_a) / sizeof(perm_a[0])) ||
        !check_layout(3, 2, 5, types_b, sizeof(types_b),
                      perm_b, sizeof(perm_b) / sizeof(perm_b[0])) ||
        !check_layout(1, 1, 3, types_c, sizeof(types_c),
                      perm_c, sizeof(perm_c) / sizeof(perm_c[0])) ||
        !check_span_parser() ||
        !check_attention_bounds()) {
        return 1;
    }

    ds4_image image = {
        .width = 17,
        .height = 9,
    };
    image.rgb = malloc((size_t)image.width * image.height * 3u);
    if (!image.rgb) return 1;
    for (uint32_t y = 0; y < image.height; y++) {
        for (uint32_t x = 0; x < image.width; x++) {
            uint8_t *pixel = image.rgb + ((size_t)y * image.width + x) * 3u;
            pixel[0] = (uint8_t)(x * 13u + y * 3u);
            pixel[1] = (uint8_t)(x * 5u + y * 17u);
            pixel[2] = (uint8_t)(x * 7u + y * 11u);
        }
    }
    char error[160] = {0};
    ds4_deepseek4_image_patches patches = {0};
    if (!ds4_image_preprocess_deepseek4(
            &patches, &image, error, sizeof(error))) {
        fprintf(stderr, "preprocess failed: %s\n", error);
        free(image.rgb);
        return 1;
    }
    int ok = patches.padded_width == 532u &&
             patches.padded_height == 280u &&
             patches.content_width == 529u &&
             patches.content_height == 280u &&
             patches.grid_width == 38u &&
             patches.grid_height == 20u &&
             patches.llm_grid_width == 13u &&
             patches.llm_grid_height == 7u &&
             patches.patch_count == 760u;
    const size_t values = (size_t)patches.patch_count * 588u;
    for (size_t i = 0; ok && i < values; i++) {
        ok = isfinite(patches.patches[i]) &&
             patches.patches[i] >= -1.00001f &&
             patches.patches[i] <= 1.00001f;
    }
    if (!ok) fprintf(stderr, "DeepSeek preprocessing dimensions or values differ\n");
    ds4_deepseek4_image_patches_free(&patches);
    free(image.rgb);
    return ok ? 0 : 1;
}
