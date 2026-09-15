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

/* ------------------------------------------------------------------------
 * WebP decoding
 *
 * The vectors below were produced with cwebp from images whose pixels follow
 * the formulas repeated here, so a lossless decode has to reproduce them
 * exactly and a lossy decode has to land close to them.
 * ------------------------------------------------------------------------ */

static const unsigned char webp_lossless_gradient[] = {
    0x52, 0x49, 0x46, 0x46, 0x2c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x4c, 0x1f, 0x00, 0x00, 0x00, 0x2f, 0x0f, 0xc0, 0x02,
    0x00, 0xb9, 0x8c, 0xe8, 0x7f, 0xec, 0x22, 0x2a, 0xd0, 0xff, 0x80, 0x90,
    0x80, 0x30, 0xc2, 0xff, 0xbb, 0x9a, 0x3c, 0x10, 0x83, 0x10, 0x13, 0x00,
    0x5c, 0x75, 0x17, 0x00,
};

static const unsigned char webp_lossy_gradient[] = {
    0x52, 0x49, 0x46, 0x46, 0x62, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x20, 0x56, 0x00, 0x00, 0x00, 0x90, 0x02, 0x00, 0x9d,
    0x01, 0x2a, 0x10, 0x00, 0x0c, 0x00, 0x00, 0xc0, 0x12, 0x25, 0xb0, 0x02,
    0x74, 0xb7, 0x00, 0xa3, 0x00, 0x8c, 0x00, 0x13, 0x71, 0x0a, 0xc0, 0xed,
    0xa0, 0x00, 0xfe, 0xff, 0xd4, 0xeb, 0x7e, 0x48, 0x24, 0xc0, 0x04, 0x85,
    0x2d, 0x2a, 0xf8, 0xac, 0xb8, 0x32, 0x9c, 0xbd, 0x97, 0xb9, 0xb8, 0xc6,
    0xb0, 0xfb, 0xcf, 0x78, 0x29, 0x10, 0x9f, 0x84, 0x2a, 0xba, 0x63, 0x77,
    0xfe, 0xfb, 0x05, 0xff, 0x8d, 0x5b, 0xbe, 0x6b, 0x99, 0x0f, 0xfc, 0xdb,
    0xfa, 0x27, 0x06, 0xbb, 0x73, 0x17, 0x13, 0x00, 0x00, 0x00,
};

static const unsigned char webp_lossless_palette[] = {
    0x52, 0x49, 0x46, 0x46, 0x44, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x4c, 0x37, 0x00, 0x00, 0x00, 0x2f, 0x0f, 0xc0, 0x02,
    0x00, 0x1f, 0x20, 0x10, 0x20, 0x8e, 0xe4, 0x88, 0x54, 0x27, 0x47, 0x48,
    0x40, 0x74, 0x03, 0x6d, 0x37, 0xc3, 0x04, 0x81, 0x6c, 0x12, 0xc4, 0xfd,
    0x17, 0xea, 0xfc, 0xc7, 0x4b, 0x64, 0x64, 0x20, 0xc8, 0xb6, 0x41, 0x98,
    0xda, 0x24, 0x2f, 0x79, 0x80, 0x31, 0x44, 0xf4, 0xbf, 0x04, 0xa3, 0xde,
    0xca, 0xaa, 0xaf, 0x00,
};

static const unsigned char webp_animated[] = {
    0x52, 0x49, 0x46, 0x46, 0x02, 0x01, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x58, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46,
    0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x00, 0x0f, 0x00, 0x00, 0x50, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x20,
    0x2c, 0x00, 0x00, 0x00, 0x50, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x10, 0x00,
    0x10, 0x00, 0x01, 0x40, 0x26, 0x25, 0xa0, 0x00, 0x04, 0x61, 0x80, 0x00,
    0xfe, 0xfb, 0x7b, 0x17, 0xff, 0xfe, 0x37, 0x8f, 0xee, 0x53, 0xfc, 0x53,
    0xe5, 0x7f, 0xff, 0xe3, 0x7f, 0x7d, 0x5e, 0x55, 0x14, 0xc0, 0x00, 0x00,
    0x41, 0x4e, 0x4d, 0x46, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0f, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
    0x56, 0x50, 0x38, 0x20, 0x28, 0x00, 0x00, 0x00, 0x54, 0x01, 0x00, 0x9d,
    0x01, 0x2a, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x26, 0x25, 0x88, 0x00,
    0x04, 0x61, 0x80, 0x00, 0xfe, 0xf9, 0x36, 0xcf, 0xff, 0xb5, 0x3f, 0x7b,
    0x3f, 0x09, 0xfa, 0x0f, 0xfc, 0x8e, 0xc2, 0x1b, 0xb0, 0x40, 0x00, 0x00,
    0x41, 0x4e, 0x4d, 0x46, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0f, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
    0x56, 0x50, 0x38, 0x20, 0x2a, 0x00, 0x00, 0x00, 0x34, 0x01, 0x00, 0x9d,
    0x01, 0x2a, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x26, 0x25, 0xa0, 0x00,
    0x03, 0x70, 0x00, 0xfe, 0xf5, 0x27, 0x5f, 0xf7, 0x49, 0xfa, 0x9f, 0xe8,
    0xbf, 0xff, 0xe7, 0xa6, 0x7f, 0xfe, 0x09, 0x3f, 0xff, 0x82, 0x4f, 0xc6,
    0x40, 0x00,
};

#define WEBP_VEC_W 16u
#define WEBP_VEC_H 12u

static void webp_expect_gradient(uint32_t x, uint32_t y, uint8_t rgb[3]) {
    rgb[0] = (uint8_t)(x * 16u + y);
    rgb[1] = (uint8_t)(y * 20u);
    rgb[2] = (uint8_t)((x + y) * 8u);
}

static void webp_expect_palette(uint32_t x, uint32_t y, uint8_t rgb[3]) {
    static const uint8_t pal[4][3] = {
        {200, 30, 40}, {30, 200, 40}, {40, 30, 200}, {240, 240, 20},
    };
    const uint8_t *p = pal[(x / 2u + y / 3u) % 4u];
    rgb[0] = p[0];
    rgb[1] = p[1];
    rgb[2] = p[2];
}

static int check_webp_exact(const char *what, const unsigned char *data,
                            size_t len,
                            void (*expect)(uint32_t, uint32_t, uint8_t *)) {
    ds4_image image = {0};
    char error[160] = {0};
    int ok = 1;
    if (!ds4_image_decode_memory(&image, data, len, error, sizeof(error))) {
        fprintf(stderr, "%s failed to decode: %s\n", what, error);
        return 0;
    }
    if (image.width != WEBP_VEC_W || image.height != WEBP_VEC_H) {
        fprintf(stderr, "%s decoded to %ux%u\n", what, image.width, image.height);
        ok = 0;
    }
    for (uint32_t y = 0; ok && y < image.height; y++) {
        for (uint32_t x = 0; ok && x < image.width; x++) {
            const uint8_t *got = image.rgb + ((size_t)y * image.width + x) * 3u;
            uint8_t want[3];
            expect(x, y, want);
            if (memcmp(got, want, 3) != 0) {
                fprintf(stderr,
                        "%s differs at %u,%u: got %u,%u,%u want %u,%u,%u\n",
                        what, x, y, got[0], got[1], got[2], want[0], want[1],
                        want[2]);
                ok = 0;
            }
        }
    }
    ds4_image_free(&image);
    return ok;
}

static int check_webp_rejected(const char *what, const unsigned char *data,
                               size_t len, const char *needle) {
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, data, len, error, sizeof(error))) {
        fprintf(stderr, "%s decoded but should have been refused\n", what);
        ds4_image_free(&image);
        return 0;
    }
    if (needle && !strstr(error, needle)) {
        fprintf(stderr, "%s reported \"%s\", expected to mention \"%s\"\n",
                what, error, needle);
        return 0;
    }
    return 1;
}

static int check_webp(void) {
    if (!check_webp_exact("lossless WebP", webp_lossless_gradient,
                          sizeof(webp_lossless_gradient),
                          webp_expect_gradient))
        return 0;
    /* A palette small enough that cwebp packs several pixels per byte. */
    if (!check_webp_exact("lossless palette WebP", webp_lossless_palette,
                          sizeof(webp_lossless_palette), webp_expect_palette))
        return 0;

    /* Lossy decoding cannot be exact, so only require it to stay close. */
    {
        ds4_image image = {0};
        char error[160] = {0};
        long total = 0;
        if (!ds4_image_decode_memory(&image, webp_lossy_gradient,
                                     sizeof(webp_lossy_gradient), error,
                                     sizeof(error))) {
            fprintf(stderr, "lossy WebP failed to decode: %s\n", error);
            return 0;
        }
        if (image.width != WEBP_VEC_W || image.height != WEBP_VEC_H) {
            fprintf(stderr, "lossy WebP decoded to %ux%u\n", image.width,
                    image.height);
            ds4_image_free(&image);
            return 0;
        }
        for (uint32_t y = 0; y < image.height; y++) {
            for (uint32_t x = 0; x < image.width; x++) {
                const uint8_t *got = image.rgb + ((size_t)y * image.width + x) * 3u;
                uint8_t want[3];
                webp_expect_gradient(x, y, want);
                for (int c = 0; c < 3; c++) total += abs((int)got[c] - (int)want[c]);
            }
        }
        ds4_image_free(&image);
        {
            const double mean = (double)total / (WEBP_VEC_W * WEBP_VEC_H * 3);
            if (mean > 12.0) {
                fprintf(stderr, "lossy WebP mean error %.2f is too high\n", mean);
                return 0;
            }
        }
    }

    /* Animation has no single image to show, and the caller needs to know
     * that rather than receiving the first frame silently. */
    if (!check_webp_rejected("animated WebP", webp_animated,
                             sizeof(webp_animated), "animated"))
        return 0;
    /*
     * Truncation at any length must either be refused with a message or, if
     * the missing bytes happened to be padding, decode at the right size.
     * What it must never do is succeed with a short buffer.
     */
    for (size_t n = 0; n < sizeof(webp_lossless_gradient); n++) {
        ds4_image image = {0};
        char error[160] = {0};
        if (!ds4_image_decode_memory(&image, webp_lossless_gradient, n, error,
                                     sizeof(error))) {
            if (!error[0]) {
                fprintf(stderr, "truncated WebP of %zu bytes gave no reason\n", n);
                return 0;
            }
            continue;
        }
        if (image.width != WEBP_VEC_W || image.height != WEBP_VEC_H) {
            fprintf(stderr, "truncated WebP of %zu bytes decoded to %ux%u\n", n,
                    image.width, image.height);
            ds4_image_free(&image);
            return 0;
        }
        ds4_image_free(&image);
    }
    {
        /* A RIFF/WEBP wrapper with no image chunk. */
        static const unsigned char empty[] = {
            'R', 'I', 'F', 'F', 4, 0, 0, 0, 'W', 'E', 'B', 'P',
        };
        if (!check_webp_rejected("chunkless WebP", empty, sizeof(empty), NULL))
            return 0;
    }
    return 1;
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
        !check_attention_bounds() ||
        !check_webp()) {
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
