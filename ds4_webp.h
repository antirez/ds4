/*
 * WebP Decoder - Single-header implementation
 *
 * A dependency-free C implementation for reading still WebP images, written
 * for ds4 in the style of the Iris PNG and JPEG decoders.
 *
 * Usage:
 *   webp_image *img = webp_load("image.webp");
 *   if (!img) { handle error }
 *
 *   // Access pixel data (RGB or RGBA)
 *   uint8_t *pixel = img->data + (y * img->width + x) * img->channels;
 *
 *   webp_free(img);
 *
 * To use as header-only, define WEBP_IMPLEMENTATION before including:
 *   #define WEBP_IMPLEMENTATION
 *   #include "ds4_webp.h"
 *
 * Features:
 *   - Lossy 'VP8 ' key frames (RFC 6386)
 *   - Lossless 'VP8L' with all four transforms (RFC 9649 Section 3)
 *   - Extended 'VP8X' container with an 'ALPH' chunk, raw or lossless,
 *     with any of the four alpha filters
 *   - Metadata chunks (ICCP, EXIF, XMP) and unknown chunks are skipped
 *   - Animated files (VP8X animation flag set) are rejected;
 *     webp_is_animated() tells them apart
 *
 * Pixels match libwebp's default decoding (dwebp) bit for bit. VP8
 * reconstruction is normative; the YUV->RGB step reproduces libwebp's
 * "fancy" chroma upsampler and fixed-point BT.601 conversion.
 *
 * License: portions of this file (the VP8 boolean decoder and coefficient
 * parsing, filter strength setup, the chroma upsampler and YUV->RGB
 * conversion) are derived from libwebp, Copyright (c) 2010 Google Inc.,
 * distributed under the BSD-3-Clause license reproduced in
 * licenses/BSD-3-Clause-libwebp.txt. The VP8 probability and quantizer
 * tables are taken from RFC 6386 (Bankoski et al., 2011).
 */

#ifndef DS4_WEBP_H
#define DS4_WEBP_H

#include <stddef.h>
#include <stdint.h>

/* Limits for untrusted server input, the same as the Iris decoders. */
#ifndef WEBP_MAX_INPUT_BYTES
#define WEBP_MAX_INPUT_BYTES (64u * 1024u * 1024u)
#endif
#ifndef WEBP_MAX_DIMENSION
#define WEBP_MAX_DIMENSION 16384
#endif
#ifndef WEBP_MAX_PIXELS
#define WEBP_MAX_PIXELS (64u * 1024u * 1024u)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Image Structure
 * ======================================================================== */

typedef struct {
    int width;
    int height;
    int channels;       /* 3=RGB, 4=RGBA */
    uint8_t *data;      /* Row-major, channel-interleaved */
} webp_image;

/* ========================================================================
 * Public API
 * ======================================================================== */

/*
 * Load WebP image from file.
 * Returns NULL on error.
 */
webp_image *webp_load(const char *path);

/*
 * Load WebP image from memory buffer.
 * Returns NULL on error.
 */
webp_image *webp_load_mem(const uint8_t *data, size_t len);

/*
 * Returns 1 when the buffer is an animated WebP file, which
 * webp_load_mem() does not decode.
 */
int webp_is_animated(const uint8_t *data, size_t len);

/*
 * Free image and pixel data.
 */
void webp_free(webp_image *img);

#ifdef __cplusplus
}
#endif

/* ========================================================================
 * Implementation
 * ======================================================================== */

#ifdef WEBP_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Utilities
 * ======================================================================== */

static uint32_t webp_le24(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
}

static uint32_t webp_le32(const uint8_t *p) {
    return webp_le24(p) | ((uint32_t)p[3] << 24);
}

static int webp_dimensions_ok(uint32_t w, uint32_t h) {
    return w > 0 && h > 0 &&
           w <= WEBP_MAX_DIMENSION && h <= WEBP_MAX_DIMENSION &&
           (uint64_t)w * h <= WEBP_MAX_PIXELS;
}

static int webp_clip(int v, int max) {
    return v < 0 ? 0 : v > max ? max : v;
}

static uint8_t webp_clip8(int v) {
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/* ========================================================================
 * VP8 Tables (RFC 6386)
 * ======================================================================== */

/* Tables below are extracted verbatim from RFC 6386 (Sections 11.5, 13.4,
 * 13.5 and 14.1). */
static const uint8_t webp_vp8_coeff_update_probs[4][8][3][11] =
{
    {
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255},
            {250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        }
    },
    {
        {
            {217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255},
            {234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255}
        },
        {
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        }
    },
    {
        {
            {186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255},
            {234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255},
            {251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255}
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        }
    },
    {
        {
            {248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255},
            {248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
        }
    }
};

static const uint8_t webp_vp8_default_coeff_probs[4][8][3][11] =
{
    {
        {
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
        },
        {
            {253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128},
            {189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128},
            {106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128}
        },
        {
            {1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128},
            {181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128},
            {78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128}
        },
        {
            {1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128},
            {184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128},
            {77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128}
        },
        {
            {1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128},
            {170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128},
            {37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128}
        },
        {
            {1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128},
            {207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128},
            {102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128}
        },
        {
            {1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128},
            {177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128},
            {80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128}
        },
        {
            {1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
        }
    },
    {
        {
            {198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62},
            {131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1},
            {68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128}
        },
        {
            {1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128},
            {184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128},
            {81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128}
        },
        {
            {1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128},
            {99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128},
            {23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128}
        },
        {
            {1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128},
            {109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128},
            {44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128}
        },
        {
            {1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128},
            {94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128},
            {22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128}
        },
        {
            {1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128},
            {124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128},
            {35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128}
        },
        {
            {1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128},
            {121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128},
            {45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128}
        },
        {
            {1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128},
            {203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128},
            {137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128}
        }
    },
    {
        {
            {253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128},
            {175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128},
            {73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128}
        },
        {
            {1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128},
            {239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128},
            {155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128}
        },
        {
            {1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128},
            {201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128},
            {69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128}
        },
        {
            {1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128},
            {223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128},
            {141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128}
        },
        {
            {1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128},
            {190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128},
            {149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128}
        },
        {
            {1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128}
        },
        {
            {1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128},
            {213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128},
            {55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128}
        },
        {
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
        }
    },
    {
        {
            {202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255},
            {126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128},
            {61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128}
        },
        {
            {1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128},
            {166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128},
            {39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128}
        },
        {
            {1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128},
            {124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128},
            {24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128}
        },
        {
            {1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128},
            {149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128},
            {28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128}
        },
        {
            {1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128},
            {123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128},
            {20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128}
        },
        {
            {1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128},
            {168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128},
            {47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128}
        },
        {
            {1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128},
            {141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128},
            {42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128}
        },
        {
            {1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128}
        }
    }
};

static const uint8_t webp_vp8_kf_bmode_probs[10][10][9] =
{
    {
        {231, 120, 48, 89, 115, 113, 120, 152, 112},
        {152, 179, 64, 126, 170, 118, 46, 70, 95},
        {175, 69, 143, 80, 85, 82, 72, 155, 103},
        {56, 58, 10, 171, 218, 189, 17, 13, 152},
        {144, 71, 10, 38, 171, 213, 144, 34, 26},
        {114, 26, 17, 163, 44, 195, 21, 10, 173},
        {121, 24, 80, 195, 26, 62, 44, 64, 85},
        {170, 46, 55, 19, 136, 160, 33, 206, 71},
        {63, 20, 8, 114, 114, 208, 12, 9, 226},
        {81, 40, 11, 96, 182, 84, 29, 16, 36}
    },
    {
        {134, 183, 89, 137, 98, 101, 106, 165, 148},
        {72, 187, 100, 130, 157, 111, 32, 75, 80},
        {66, 102, 167, 99, 74, 62, 40, 234, 128},
        {41, 53, 9, 178, 241, 141, 26, 8, 107},
        {104, 79, 12, 27, 217, 255, 87, 17, 7},
        {74, 43, 26, 146, 73, 166, 49, 23, 157},
        {65, 38, 105, 160, 51, 52, 31, 115, 128},
        {87, 68, 71, 44, 114, 51, 15, 186, 23},
        {47, 41, 14, 110, 182, 183, 21, 17, 194},
        {66, 45, 25, 102, 197, 189, 23, 18, 22}
    },
    {
        {88, 88, 147, 150, 42, 46, 45, 196, 205},
        {43, 97, 183, 117, 85, 38, 35, 179, 61},
        {39, 53, 200, 87, 26, 21, 43, 232, 171},
        {56, 34, 51, 104, 114, 102, 29, 93, 77},
        {107, 54, 32, 26, 51, 1, 81, 43, 31},
        {39, 28, 85, 171, 58, 165, 90, 98, 64},
        {34, 22, 116, 206, 23, 34, 43, 166, 73},
        {68, 25, 106, 22, 64, 171, 36, 225, 114},
        {34, 19, 21, 102, 132, 188, 16, 76, 124},
        {62, 18, 78, 95, 85, 57, 50, 48, 51}
    },
    {
        {193, 101, 35, 159, 215, 111, 89, 46, 111},
        {60, 148, 31, 172, 219, 228, 21, 18, 111},
        {112, 113, 77, 85, 179, 255, 38, 120, 114},
        {40, 42, 1, 196, 245, 209, 10, 25, 109},
        {100, 80, 8, 43, 154, 1, 51, 26, 71},
        {88, 43, 29, 140, 166, 213, 37, 43, 154},
        {61, 63, 30, 155, 67, 45, 68, 1, 209},
        {142, 78, 78, 16, 255, 128, 34, 197, 171},
        {41, 40, 5, 102, 211, 183, 4, 1, 221},
        {51, 50, 17, 168, 209, 192, 23, 25, 82}
    },
    {
        {125, 98, 42, 88, 104, 85, 117, 175, 82},
        {95, 84, 53, 89, 128, 100, 113, 101, 45},
        {75, 79, 123, 47, 51, 128, 81, 171, 1},
        {57, 17, 5, 71, 102, 57, 53, 41, 49},
        {115, 21, 2, 10, 102, 255, 166, 23, 6},
        {38, 33, 13, 121, 57, 73, 26, 1, 85},
        {41, 10, 67, 138, 77, 110, 90, 47, 114},
        {101, 29, 16, 10, 85, 128, 101, 196, 26},
        {57, 18, 10, 102, 102, 213, 34, 20, 43},
        {117, 20, 15, 36, 163, 128, 68, 1, 26}
    },
    {
        {138, 31, 36, 171, 27, 166, 38, 44, 229},
        {67, 87, 58, 169, 82, 115, 26, 59, 179},
        {63, 59, 90, 180, 59, 166, 93, 73, 154},
        {40, 40, 21, 116, 143, 209, 34, 39, 175},
        {57, 46, 22, 24, 128, 1, 54, 17, 37},
        {47, 15, 16, 183, 34, 223, 49, 45, 183},
        {46, 17, 33, 183, 6, 98, 15, 32, 183},
        {65, 32, 73, 115, 28, 128, 23, 128, 205},
        {40, 3, 9, 115, 51, 192, 18, 6, 223},
        {87, 37, 9, 115, 59, 77, 64, 21, 47}
    },
    {
        {104, 55, 44, 218, 9, 54, 53, 130, 226},
        {64, 90, 70, 205, 40, 41, 23, 26, 57},
        {54, 57, 112, 184, 5, 41, 38, 166, 213},
        {30, 34, 26, 133, 152, 116, 10, 32, 134},
        {75, 32, 12, 51, 192, 255, 160, 43, 51},
        {39, 19, 53, 221, 26, 114, 32, 73, 255},
        {31, 9, 65, 234, 2, 15, 1, 118, 73},
        {88, 31, 35, 67, 102, 85, 55, 186, 85},
        {56, 21, 23, 111, 59, 205, 45, 37, 192},
        {55, 38, 70, 124, 73, 102, 1, 34, 98}
    },
    {
        {102, 61, 71, 37, 34, 53, 31, 243, 192},
        {69, 60, 71, 38, 73, 119, 28, 222, 37},
        {68, 45, 128, 34, 1, 47, 11, 245, 171},
        {62, 17, 19, 70, 146, 85, 55, 62, 70},
        {75, 15, 9, 9, 64, 255, 184, 119, 16},
        {37, 43, 37, 154, 100, 163, 85, 160, 1},
        {63, 9, 92, 136, 28, 64, 32, 201, 85},
        {86, 6, 28, 5, 64, 255, 25, 248, 1},
        {56, 8, 17, 132, 137, 255, 55, 116, 128},
        {58, 15, 20, 82, 135, 57, 26, 121, 40}
    },
    {
        {164, 50, 31, 137, 154, 133, 25, 35, 218},
        {51, 103, 44, 131, 131, 123, 31, 6, 158},
        {86, 40, 64, 135, 148, 224, 45, 183, 128},
        {22, 26, 17, 131, 240, 154, 14, 1, 209},
        {83, 12, 13, 54, 192, 255, 68, 47, 28},
        {45, 16, 21, 91, 64, 222, 7, 1, 197},
        {56, 21, 39, 155, 60, 138, 23, 102, 213},
        {85, 26, 85, 85, 128, 128, 32, 146, 171},
        {18, 11, 7, 63, 144, 171, 4, 4, 246},
        {35, 27, 10, 146, 174, 171, 12, 26, 128}
    },
    {
        {190, 80, 35, 99, 180, 80, 126, 54, 45},
        {85, 126, 47, 87, 176, 51, 41, 20, 32},
        {101, 75, 128, 139, 118, 146, 116, 128, 85},
        {56, 41, 15, 176, 236, 85, 37, 9, 62},
        {146, 36, 19, 30, 171, 255, 97, 27, 20},
        {71, 30, 17, 119, 118, 255, 17, 18, 138},
        {101, 38, 60, 138, 55, 70, 43, 26, 142},
        {138, 45, 61, 62, 219, 1, 81, 188, 64},
        {32, 41, 20, 117, 151, 142, 20, 21, 163},
        {112, 19, 12, 61, 195, 128, 48, 4, 24}
    }
};

static const uint16_t webp_vp8_dc_qlookup[128] = {
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
    18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
};

static const uint16_t webp_vp8_ac_qlookup[128] = {
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
};

/* Subblock modes, in RFC 6386 order (Section 11.2). */
enum {
    WEBP_B_DC, WEBP_B_TM, WEBP_B_VE, WEBP_B_HE, WEBP_B_LD,
    WEBP_B_RD, WEBP_B_VR, WEBP_B_VL, WEBP_B_HD, WEBP_B_HU
};

/* Macroblock luma modes and chroma modes. */
enum { WEBP_DC_PRED, WEBP_V_PRED, WEBP_H_PRED, WEBP_TM_PRED };

static const int8_t webp_vp8_bmode_tree[18] = {
    -WEBP_B_DC, 2,
    -WEBP_B_TM, 4,
    -WEBP_B_VE, 6,
    8, 12,
    -WEBP_B_HE, 10,
    -WEBP_B_RD, -WEBP_B_VR,
    -WEBP_B_LD, 14,
    -WEBP_B_VL, 16,
    -WEBP_B_HD, -WEBP_B_HU
};

static const uint8_t webp_vp8_zigzag[16] = {
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
};

/* Coefficient position -> band. The extra entry is read (never used) when
 * the last coefficient is decoded. */
static const uint8_t webp_vp8_bands[17] = {
    0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0
};

static const uint8_t webp_vp8_pcat3[] = { 173, 148, 140, 0 };
static const uint8_t webp_vp8_pcat4[] = { 176, 155, 140, 135, 0 };
static const uint8_t webp_vp8_pcat5[] = { 180, 157, 141, 134, 130, 0 };
static const uint8_t webp_vp8_pcat6[] = {
    254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0
};
static const uint8_t *const webp_vp8_pcat3456[4] = {
    webp_vp8_pcat3, webp_vp8_pcat4, webp_vp8_pcat5, webp_vp8_pcat6
};

/* ========================================================================
 * VP8 Boolean Decoder (RFC 6386 Section 7)
 * ======================================================================== */

/* 'range' holds range - 1. Bytes past the end of a partition read as zero
 * and set 'eof', which the caller turns into an error at the next row or
 * macroblock boundary.
 *
 * The state layout (a 64-bit window refilled 56 bits at a time) follows
 * libwebp on 64-bit hosts. For valid streams any layout gives the same
 * bits; this one also makes damaged streams that libwebp still accepts
 * decode to the same pixels. */
typedef struct {
    const uint8_t *buf;
    const uint8_t *end;
    const uint8_t *buf_max;     /* last position where 8 bytes can be read */
    uint64_t value;
    uint32_t range;
    int bits;
    int eof;
} webp_vp8_br;

static int webp_log2_floor(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return 31 ^ __builtin_clz(v);
#else
    int n = 0;
    while (v >>= 1) n++;
    return n;
#endif
}

static void webp_vp8_br_load(webp_vp8_br *br) {
    if (br->buf < br->buf_max) {
        uint64_t bits = 0;
        for (int i = 0; i < 7; i++) bits = (bits << 8) | br->buf[i];
        br->buf += 7;
        br->value = bits | (br->value << 56);
        br->bits += 56;
    } else if (br->buf < br->end) {
        br->bits += 8;
        br->value = (br->value << 8) | *br->buf++;
    } else if (!br->eof) {
        br->value <<= 8;
        br->bits += 8;
        br->eof = 1;
    } else {
        br->bits = 0;
    }
}

static void webp_vp8_br_init(webp_vp8_br *br, const uint8_t *buf, size_t size) {
    br->buf = buf;
    br->end = buf + size;
    br->buf_max = size >= 8 ? buf + size - 8 + 1 : buf;
    br->value = 0;
    br->range = 255 - 1;
    br->bits = -8;
    br->eof = 0;
    webp_vp8_br_load(br);
}

static int webp_vp8_bit(webp_vp8_br *br, int prob) {
    uint32_t range = br->range;
    if (br->bits < 0) webp_vp8_br_load(br);
    const uint32_t split = (range * (uint32_t)prob) >> 8;
    const uint32_t value = (uint32_t)(br->value >> br->bits);
    int bit;
    if (value > split) {
        range -= split;
        br->value -= (uint64_t)(split + 1) << br->bits;
        bit = 1;
    } else {
        range = split + 1;
        bit = 0;
    }
    const int shift = 7 ^ webp_log2_floor(range);
    range <<= shift;
    br->bits -= shift;
    br->range = range - 1;
    return bit;
}

/* Coefficient sign: an equiprobable bit read with a single shift. This is
 * read_bool(128) in every state except the initial one (range 255), which a
 * sign read never sees because each block starts with an end-of-block bit. */
static int webp_vp8_sign(webp_vp8_br *br) {
    if (br->bits < 0) webp_vp8_br_load(br);
    const uint32_t split = br->range >> 1;
    const uint32_t value = (uint32_t)(br->value >> br->bits);
    const int32_t mask = (int32_t)(split - value) >> 31;   /* -1 when the bit is 1 */
    br->value -= (uint64_t)((split + 1) & (uint32_t)mask) << br->bits;
    br->bits -= 1;
    br->range += (uint32_t)mask;
    br->range |= 1;
    return mask != 0;
}

static uint32_t webp_vp8_value(webp_vp8_br *br, int bits) {
    uint32_t v = 0;
    while (bits-- > 0) v |= (uint32_t)webp_vp8_bit(br, 0x80) << bits;
    return v;
}

static int webp_vp8_signed_value(webp_vp8_br *br, int bits) {
    const int v = (int)webp_vp8_value(br, bits);
    return webp_vp8_bit(br, 0x80) ? -v : v;
}

/* ========================================================================
 * VP8 Decoder State
 * ======================================================================== */

typedef struct {
    uint8_t limit;      /* 0: no filtering */
    uint8_t ilevel;
    uint8_t hev_thresh;
    uint8_t inner;
} webp_vp8_finfo;

typedef struct {
    int y1[2], y2[2], uv[2];   /* dc, ac */
} webp_vp8_quant;

typedef struct {
    uint8_t segment;
    uint8_t skip;
    uint8_t is_i4x4;
    uint8_t ymode;
    uint8_t uvmode;
    uint8_t imodes[16];
} webp_vp8_mbinfo;

typedef struct {
    int width, height;
    int mb_w, mb_h;

    webp_vp8_br br;             /* first partition: modes */
    webp_vp8_br parts[8];       /* token partitions */
    int num_parts_minus_one;

    int use_segment, update_map, absolute_delta;
    int quantizer[4];
    int filter_strength[4];
    uint8_t segment_probs[3];

    int filter_simple, filter_level, sharpness, use_lf_delta;
    int ref_lf_delta[4];
    int mode_lf_delta[4];
    int filter_type;            /* 0: off, 1: simple, 2: normal */
    webp_vp8_finfo fstrengths[4][2];   /* [segment][is_i4x4] */

    webp_vp8_quant dqm[4];
    uint8_t coeff_probs[4][8][3][11];
    int use_skip_proba;
    int skip_p;

    /* Reconstructed planes, padded to whole macroblocks. */
    uint8_t *y, *u, *v;
    int y_stride, uv_stride;

    webp_vp8_mbinfo *mbs;       /* current macroblock row */
    uint8_t *intra_t;           /* 4 subblock modes per column above */
    uint8_t intra_l[4];
    uint8_t *nz_top;            /* 9 non-zero flags per column above */
    uint8_t nz_left[9];         /* y[0..3], u[4..5], v[6..7], y2[8] */
    webp_vp8_finfo *finfo;      /* mb_w * mb_h */
} webp_vp8_dec;

/* ========================================================================
 * VP8 Frame Header
 * ======================================================================== */

static int webp_vp8_parse_segment_header(webp_vp8_dec *d) {
    webp_vp8_br *br = &d->br;
    d->use_segment = (int)webp_vp8_value(br, 1);
    if (d->use_segment) {
        d->update_map = (int)webp_vp8_value(br, 1);
        if (webp_vp8_value(br, 1)) {
            d->absolute_delta = (int)webp_vp8_value(br, 1);
            for (int s = 0; s < 4; s++)
                d->quantizer[s] = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 7) : 0;
            for (int s = 0; s < 4; s++)
                d->filter_strength[s] = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 6) : 0;
        }
        if (d->update_map) {
            for (int s = 0; s < 3; s++)
                d->segment_probs[s] = webp_vp8_value(br, 1) ? (uint8_t)webp_vp8_value(br, 8) : 255;
        }
    } else {
        d->update_map = 0;
    }
    return !br->eof;
}

static int webp_vp8_parse_filter_header(webp_vp8_dec *d) {
    webp_vp8_br *br = &d->br;
    d->filter_simple = (int)webp_vp8_value(br, 1);
    d->filter_level = (int)webp_vp8_value(br, 6);
    d->sharpness = (int)webp_vp8_value(br, 3);
    d->use_lf_delta = (int)webp_vp8_value(br, 1);
    if (d->use_lf_delta && webp_vp8_value(br, 1)) {
        for (int i = 0; i < 4; i++)
            if (webp_vp8_value(br, 1)) d->ref_lf_delta[i] = webp_vp8_signed_value(br, 6);
        for (int i = 0; i < 4; i++)
            if (webp_vp8_value(br, 1)) d->mode_lf_delta[i] = webp_vp8_signed_value(br, 6);
    }
    d->filter_type = d->filter_level == 0 ? 0 : d->filter_simple ? 1 : 2;
    return !br->eof;
}

static int webp_vp8_parse_partitions(webp_vp8_dec *d, const uint8_t *buf, size_t size) {
    const uint8_t *sizes = buf;
    const int last = d->num_parts_minus_one = (1 << webp_vp8_value(&d->br, 2)) - 1;
    if (size < 3u * (size_t)last) return 0;
    const uint8_t *part_start = buf + 3 * last;
    size_t size_left = size - 3u * (size_t)last;
    for (int p = 0; p < last; p++) {
        size_t psize = webp_le24(sizes);
        if (psize > size_left) psize = size_left;
        webp_vp8_br_init(&d->parts[p], part_start, psize);
        part_start += psize;
        size_left -= psize;
        sizes += 3;
    }
    webp_vp8_br_init(&d->parts[last], part_start, size_left);
    return part_start < buf + size;
}

static void webp_vp8_parse_quant(webp_vp8_dec *d) {
    webp_vp8_br *br = &d->br;
    const int base_q0 = (int)webp_vp8_value(br, 7);
    const int dqy1_dc = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 4) : 0;
    const int dqy2_dc = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 4) : 0;
    const int dqy2_ac = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 4) : 0;
    const int dquv_dc = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 4) : 0;
    const int dquv_ac = webp_vp8_value(br, 1) ? webp_vp8_signed_value(br, 4) : 0;
    for (int i = 0; i < 4; i++) {
        webp_vp8_quant *m = &d->dqm[i];
        int q;
        if (d->use_segment) {
            q = d->quantizer[i];
            if (!d->absolute_delta) q += base_q0;
        } else if (i > 0) {
            d->dqm[i] = d->dqm[0];
            continue;
        } else {
            q = base_q0;
        }
        m->y1[0] = webp_vp8_dc_qlookup[webp_clip(q + dqy1_dc, 127)];
        m->y1[1] = webp_vp8_ac_qlookup[webp_clip(q, 127)];
        m->y2[0] = webp_vp8_dc_qlookup[webp_clip(q + dqy2_dc, 127)] * 2;
        m->y2[1] = webp_vp8_ac_qlookup[webp_clip(q + dqy2_ac, 127)] * 155 / 100;
        if (m->y2[1] < 8) m->y2[1] = 8;
        m->uv[0] = webp_vp8_dc_qlookup[webp_clip(q + dquv_dc, 117)];
        m->uv[1] = webp_vp8_ac_qlookup[webp_clip(q + dquv_ac, 127)];
    }
}

static void webp_vp8_parse_proba(webp_vp8_dec *d) {
    webp_vp8_br *br = &d->br;
    for (int t = 0; t < 4; t++)
        for (int b = 0; b < 8; b++)
            for (int c = 0; c < 3; c++)
                for (int p = 0; p < 11; p++)
                    d->coeff_probs[t][b][c][p] =
                        webp_vp8_bit(br, webp_vp8_coeff_update_probs[t][b][c][p])
                            ? (uint8_t)webp_vp8_value(br, 8)
                            : webp_vp8_default_coeff_probs[t][b][c][p];
    d->use_skip_proba = (int)webp_vp8_value(br, 1);
    if (d->use_skip_proba) d->skip_p = (int)webp_vp8_value(br, 8);
}

static void webp_vp8_precompute_filter_strengths(webp_vp8_dec *d) {
    if (d->filter_type == 0) return;
    for (int s = 0; s < 4; s++) {
        int base_level;
        if (d->use_segment) {
            base_level = d->filter_strength[s];
            if (!d->absolute_delta) base_level += d->filter_level;
        } else {
            base_level = d->filter_level;
        }
        for (int i4x4 = 0; i4x4 <= 1; i4x4++) {
            webp_vp8_finfo *info = &d->fstrengths[s][i4x4];
            int level = base_level;
            if (d->use_lf_delta) {
                level += d->ref_lf_delta[0];
                if (i4x4) level += d->mode_lf_delta[0];
            }
            level = webp_clip(level, 63);
            if (level > 0) {
                int ilevel = level;
                if (d->sharpness > 0) {
                    ilevel >>= d->sharpness > 4 ? 2 : 1;
                    if (ilevel > 9 - d->sharpness) ilevel = 9 - d->sharpness;
                }
                if (ilevel < 1) ilevel = 1;
                info->ilevel = (uint8_t)ilevel;
                info->limit = (uint8_t)(2 * level + ilevel);
                info->hev_thresh = level >= 40 ? 2 : level >= 15 ? 1 : 0;
            } else {
                info->limit = 0;
            }
            info->inner = (uint8_t)i4x4;
        }
    }
}

/* ========================================================================
 * VP8 Modes (First Partition)
 * ======================================================================== */

static void webp_vp8_parse_intra_mode(webp_vp8_dec *d, webp_vp8_mbinfo *mb, int mb_x) {
    webp_vp8_br *br = &d->br;
    uint8_t *top = d->intra_t + 4 * mb_x;
    uint8_t *left = d->intra_l;
    if (d->update_map) {
        mb->segment = !webp_vp8_bit(br, d->segment_probs[0])
                          ? (uint8_t)webp_vp8_bit(br, d->segment_probs[1])
                          : (uint8_t)(webp_vp8_bit(br, d->segment_probs[2]) + 2);
    } else {
        mb->segment = 0;
    }
    mb->skip = d->use_skip_proba ? (uint8_t)webp_vp8_bit(br, d->skip_p) : 0;
    mb->is_i4x4 = !webp_vp8_bit(br, 145);
    if (!mb->is_i4x4) {
        /* kf_ymode_tree with kf_ymode_prob = { 145, 156, 163, 128 }. */
        const int ymode = webp_vp8_bit(br, 156)
                              ? (webp_vp8_bit(br, 128) ? WEBP_TM_PRED : WEBP_H_PRED)
                              : (webp_vp8_bit(br, 163) ? WEBP_V_PRED : WEBP_DC_PRED);
        /* Implied subblock modes used as context by B_PRED neighbors. */
        static const uint8_t implied[4] = { WEBP_B_DC, WEBP_B_VE, WEBP_B_HE, WEBP_B_TM };
        mb->ymode = (uint8_t)ymode;
        memset(top, implied[ymode], 4);
        memset(left, implied[ymode], 4);
    } else {
        for (int y = 0; y < 4; y++) {
            int mode = left[y];
            for (int x = 0; x < 4; x++) {
                const uint8_t *prob = webp_vp8_kf_bmode_probs[top[x]][mode];
                int i = webp_vp8_bmode_tree[webp_vp8_bit(br, prob[0])];
                while (i > 0) i = webp_vp8_bmode_tree[i + webp_vp8_bit(br, prob[i >> 1])];
                mode = -i;
                top[x] = (uint8_t)mode;
                mb->imodes[4 * y + x] = (uint8_t)mode;
            }
            left[y] = (uint8_t)mode;
        }
    }
    /* uv_mode_tree with kf_uv_mode_prob = { 142, 114, 183 }. */
    mb->uvmode = !webp_vp8_bit(br, 142) ? WEBP_DC_PRED
               : !webp_vp8_bit(br, 114) ? WEBP_V_PRED
               : webp_vp8_bit(br, 183) ? WEBP_TM_PRED : WEBP_H_PRED;
}

/* ========================================================================
 * VP8 Residuals (Token Partitions)
 * ======================================================================== */

static int webp_vp8_large_value(webp_vp8_br *br, const uint8_t *p) {
    int v;
    if (!webp_vp8_bit(br, p[3])) {
        if (!webp_vp8_bit(br, p[4])) {
            v = 2;
        } else {
            v = 3 + webp_vp8_bit(br, p[5]);
        }
    } else if (!webp_vp8_bit(br, p[6])) {
        if (!webp_vp8_bit(br, p[7])) {
            v = 5 + webp_vp8_bit(br, 159);                   /* cat1 */
        } else {
            v = 7 + 2 * webp_vp8_bit(br, 165);               /* cat2 */
            v += webp_vp8_bit(br, 145);
        }
    } else {
        const int bit1 = webp_vp8_bit(br, p[8]);
        const int bit0 = webp_vp8_bit(br, p[9 + bit1]);
        const int cat = 2 * bit1 + bit0;                     /* cat3..cat6 */
        v = 0;
        for (const uint8_t *tab = webp_vp8_pcat3456[cat]; *tab; tab++)
            v += v + webp_vp8_bit(br, *tab);
        v += 3 + (8 << cat);
    }
    return v;
}

/* Decodes one block's tokens starting at position n. Returns the position
 * after the last decoded token (the end-of-block position). */
static int webp_vp8_get_coeffs(webp_vp8_br *br, const uint8_t (*probs)[3][11], int ctx,
                               const int dq[2], int n, int16_t *out) {
    const uint8_t *p = probs[webp_vp8_bands[n]][ctx];
    for (; n < 16; n++) {
        if (!webp_vp8_bit(br, p[0])) return n;           /* end of block */
        while (!webp_vp8_bit(br, p[1])) {                /* DCT_0 */
            p = probs[webp_vp8_bands[++n]][0];
            if (n == 16) return 16;
        }
        const uint8_t (*p_ctx)[11] = probs[webp_vp8_bands[n + 1]];
        int v;
        if (!webp_vp8_bit(br, p[2])) {
            v = 1;
            p = p_ctx[1];
        } else {
            v = webp_vp8_large_value(br, p);
            p = p_ctx[2];
        }
        if (webp_vp8_sign(br)) v = -v;
        out[webp_vp8_zigzag[n]] = (int16_t)(v * dq[n > 0]);
    }
    return 16;
}

static void webp_vp8_transform_wht(const int16_t *in, int16_t *out) {
    int tmp[16];
    for (int i = 0; i < 4; i++) {
        const int a0 = in[0 + i] + in[12 + i];
        const int a1 = in[4 + i] + in[8 + i];
        const int a2 = in[4 + i] - in[8 + i];
        const int a3 = in[0 + i] - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for (int i = 0; i < 4; i++) {
        const int dc = tmp[0 + i * 4] + 3;
        const int a0 = dc + tmp[3 + i * 4];
        const int a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        const int a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        const int a3 = dc - tmp[3 + i * 4];
        out[0] = (int16_t)((a0 + a1) >> 3);
        out[16] = (int16_t)((a3 + a2) >> 3);
        out[32] = (int16_t)((a0 - a1) >> 3);
        out[48] = (int16_t)((a3 - a2) >> 3);
        out += 64;
    }
}

/* Parses the residuals of one macroblock into coeffs[384] (16 Y blocks,
 * then 4 U and 4 V blocks). Returns 1 when any block carries data, which
 * enables the inner-edge loop filter. */
static int webp_vp8_parse_residuals(webp_vp8_dec *d, const webp_vp8_mbinfo *mb, int mb_x,
                                    webp_vp8_br *br, int16_t *coeffs) {
    const webp_vp8_quant *q = &d->dqm[mb->segment];
    uint8_t *tnz = d->nz_top + 9 * mb_x;
    uint8_t *lnz = d->nz_left;
    const uint8_t (*ac_probs)[3][11];
    int16_t *dst = coeffs;
    int any = 0;
    int first;

    memset(coeffs, 0, 384 * sizeof(*coeffs));
    if (!mb->is_i4x4) {
        int16_t dc[16] = {0};
        const int ctx = tnz[8] + lnz[8];
        const int nz = webp_vp8_get_coeffs(br, d->coeff_probs[1], ctx, q->y2, 0, dc);
        tnz[8] = lnz[8] = (uint8_t)(nz > 0);
        if (nz > 1) {
            webp_vp8_transform_wht(dc, dst);
        } else {
            const int dc0 = (dc[0] + 3) >> 3;
            for (int i = 0; i < 16 * 16; i += 16) dst[i] = (int16_t)dc0;
        }
        first = 1;
        ac_probs = d->coeff_probs[0];
    } else {
        first = 0;
        ac_probs = d->coeff_probs[3];
    }

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            const int ctx = lnz[y] + tnz[x];
            const int nz = webp_vp8_get_coeffs(br, ac_probs, ctx, q->y1, first, dst);
            lnz[y] = tnz[x] = (uint8_t)(nz > first);
            if (nz > 1 || dst[0] != 0) any = 1;
            dst += 16;
        }
    }
    for (int ch = 0; ch < 2; ch++) {
        const int base = 4 + 2 * ch;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                const int ctx = lnz[base + y] + tnz[base + x];
                const int nz = webp_vp8_get_coeffs(br, d->coeff_probs[2], ctx, q->uv, 0, dst);
                lnz[base + y] = tnz[base + x] = (uint8_t)(nz > 0);
                if (nz > 1 || dst[0] != 0) any = 1;
                dst += 16;
            }
        }
    }
    return any;
}

/* ========================================================================
 * VP8 Reconstruction (Prediction and Inverse Transform)
 * ======================================================================== */

#define WEBP_BPS 32

static int webp_vp8_mul1(int a) { return (int)(((int64_t)a * 20091) >> 16) + a; }
static int webp_vp8_mul2(int a) { return (int)(((int64_t)a * 35468) >> 16); }

/* Inverse DCT of one 4x4 block, added to the prediction in dst. */
static void webp_vp8_idct_add(const int16_t *in, uint8_t *dst) {
    int tmp[16];
    for (int i = 0; i < 4; i++) {          /* vertical pass */
        const int a = in[i] + in[8 + i];
        const int b = in[i] - in[8 + i];
        const int c = webp_vp8_mul2(in[4 + i]) - webp_vp8_mul1(in[12 + i]);
        const int dd = webp_vp8_mul1(in[4 + i]) + webp_vp8_mul2(in[12 + i]);
        tmp[0 + i] = a + dd;
        tmp[4 + i] = b + c;
        tmp[8 + i] = b - c;
        tmp[12 + i] = a - dd;
    }
    for (int i = 0; i < 4; i++) {          /* horizontal pass */
        const int *r = tmp + 4 * i;
        const int dc = r[0] + 4;
        const int a = dc + r[2];
        const int b = dc - r[2];
        const int c = webp_vp8_mul2(r[1]) - webp_vp8_mul1(r[3]);
        const int dd = webp_vp8_mul1(r[1]) + webp_vp8_mul2(r[3]);
        uint8_t *o = dst + i * WEBP_BPS;
        o[0] = webp_clip8(o[0] + ((a + dd) >> 3));
        o[1] = webp_clip8(o[1] + ((b + c) >> 3));
        o[2] = webp_clip8(o[2] + ((b - c) >> 3));
        o[3] = webp_clip8(o[3] + ((a - dd) >> 3));
    }
}

static int webp_vp8_block_is_zero(const int16_t *c) {
    for (int i = 0; i < 16; i++)
        if (c[i]) return 0;
    return 1;
}

static void webp_vp8_true_motion(uint8_t *dst, int size) {
    const uint8_t *top = dst - WEBP_BPS;
    const int corner = top[-1];
    for (int y = 0; y < size; y++) {
        const int left = dst[-1];
        for (int x = 0; x < size; x++) dst[x] = webp_clip8(top[x] + left - corner);
        dst += WEBP_BPS;
    }
}

/* 16x16 luma and 8x8 chroma prediction. DC mode depends on which edges
 * exist; the other modes use the 127/129 frame borders as they are. */
static void webp_vp8_predict_block(uint8_t *dst, int size, int mode, int mb_x, int mb_y) {
    const int shift = size == 16 ? 4 : 3;
    switch (mode) {
    case WEBP_DC_PRED: {
        int dc = 0;
        if (mb_y > 0)
            for (int x = 0; x < size; x++) dc += dst[x - WEBP_BPS];
        if (mb_x > 0)
            for (int y = 0; y < size; y++) dc += dst[y * WEBP_BPS - 1];
        if (mb_x > 0 && mb_y > 0) {
            dc = (dc + size) >> (shift + 1);
        } else if (mb_x > 0 || mb_y > 0) {
            dc = (dc + (size >> 1)) >> shift;
        } else {
            dc = 0x80;
        }
        for (int y = 0; y < size; y++) memset(dst + y * WEBP_BPS, dc, (size_t)size);
        break;
    }
    case WEBP_V_PRED:
        for (int y = 0; y < size; y++) memcpy(dst + y * WEBP_BPS, dst - WEBP_BPS, (size_t)size);
        break;
    case WEBP_H_PRED:
        for (int y = 0; y < size; y++)
            memset(dst + y * WEBP_BPS, dst[y * WEBP_BPS - 1], (size_t)size);
        break;
    default:
        webp_vp8_true_motion(dst, size);
        break;
    }
}

static uint8_t webp_avg3(int x, int y, int z) { return (uint8_t)((x + y + y + z + 2) >> 2); }
static uint8_t webp_avg2(int x, int y) { return (uint8_t)((x + y + 1) >> 1); }

/* 4x4 subblock prediction, transcribed from RFC 6386 Section 12.3.
 * B[r][c] is row r, column c. A[-1] = L[-1] = P (top-left corner). */
static void webp_vp8_predict4(uint8_t *dst, int mode) {
    int AP[9], LP[5], E[9];
    const int *A = AP + 1;
    const int *L = LP + 1;
    uint8_t B[4][4];
    AP[0] = LP[0] = dst[-WEBP_BPS - 1];
    for (int i = 0; i < 8; i++) AP[i + 1] = dst[i - WEBP_BPS];
    for (int i = 0; i < 4; i++) LP[i + 1] = dst[i * WEBP_BPS - 1];
    E[0] = L[3]; E[1] = L[2]; E[2] = L[1]; E[3] = L[0];
    E[4] = A[-1];
    E[5] = A[0]; E[6] = A[1]; E[7] = A[2]; E[8] = A[3];
#define WEBP_AVG3P(p) webp_avg3((p)[-1], (p)[0], (p)[1])
#define WEBP_AVG2P(p) webp_avg2((p)[0], (p)[1])
    switch (mode) {
    case WEBP_B_DC: {
        int v = 4;
        for (int i = 0; i < 4; i++) v += A[i] + L[i];
        memset(B, v >> 3, sizeof(B));
        break;
    }
    case WEBP_B_TM:
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) B[r][c] = webp_clip8(L[r] + A[c] - A[-1]);
        break;
    case WEBP_B_VE:
        for (int c = 0; c < 4; c++)
            B[0][c] = B[1][c] = B[2][c] = B[3][c] = WEBP_AVG3P(A + c);
        break;
    case WEBP_B_HE:
        for (int c = 0; c < 4; c++) {
            B[3][c] = webp_avg3(L[2], L[3], L[3]);
            B[2][c] = WEBP_AVG3P(L + 2);
            B[1][c] = WEBP_AVG3P(L + 1);
            B[0][c] = WEBP_AVG3P(L + 0);
        }
        break;
    case WEBP_B_LD:
        B[0][0] = WEBP_AVG3P(A + 1);
        B[0][1] = B[1][0] = WEBP_AVG3P(A + 2);
        B[0][2] = B[1][1] = B[2][0] = WEBP_AVG3P(A + 3);
        B[0][3] = B[1][2] = B[2][1] = B[3][0] = WEBP_AVG3P(A + 4);
        B[1][3] = B[2][2] = B[3][1] = WEBP_AVG3P(A + 5);
        B[2][3] = B[3][2] = WEBP_AVG3P(A + 6);
        B[3][3] = webp_avg3(A[6], A[7], A[7]);
        break;
    case WEBP_B_RD:
        B[3][0] = WEBP_AVG3P(E + 1);
        B[3][1] = B[2][0] = WEBP_AVG3P(E + 2);
        B[3][2] = B[2][1] = B[1][0] = WEBP_AVG3P(E + 3);
        B[3][3] = B[2][2] = B[1][1] = B[0][0] = WEBP_AVG3P(E + 4);
        B[2][3] = B[1][2] = B[0][1] = WEBP_AVG3P(E + 5);
        B[1][3] = B[0][2] = WEBP_AVG3P(E + 6);
        B[0][3] = WEBP_AVG3P(E + 7);
        break;
    case WEBP_B_VR:
        B[3][0] = WEBP_AVG3P(E + 2);
        B[2][0] = WEBP_AVG3P(E + 3);
        B[3][1] = B[1][0] = WEBP_AVG3P(E + 4);
        B[2][1] = B[0][0] = WEBP_AVG2P(E + 4);
        B[3][2] = B[1][1] = WEBP_AVG3P(E + 5);
        B[2][2] = B[0][1] = WEBP_AVG2P(E + 5);
        B[3][3] = B[1][2] = WEBP_AVG3P(E + 6);
        B[2][3] = B[0][2] = WEBP_AVG2P(E + 6);
        B[1][3] = WEBP_AVG3P(E + 7);
        B[0][3] = WEBP_AVG2P(E + 7);
        break;
    case WEBP_B_VL:
        B[0][0] = WEBP_AVG2P(A);
        B[1][0] = WEBP_AVG3P(A + 1);
        B[2][0] = B[0][1] = WEBP_AVG2P(A + 1);
        B[1][1] = B[3][0] = WEBP_AVG3P(A + 2);
        B[2][1] = B[0][2] = WEBP_AVG2P(A + 2);
        B[3][1] = B[1][2] = WEBP_AVG3P(A + 3);
        B[2][2] = B[0][3] = WEBP_AVG2P(A + 3);
        B[3][2] = B[1][3] = WEBP_AVG3P(A + 4);
        B[2][3] = WEBP_AVG3P(A + 5);
        B[3][3] = WEBP_AVG3P(A + 6);
        break;
    case WEBP_B_HD:
        B[3][0] = WEBP_AVG2P(E);
        B[3][1] = WEBP_AVG3P(E + 1);
        B[2][0] = B[3][2] = WEBP_AVG2P(E + 1);
        B[2][1] = B[3][3] = WEBP_AVG3P(E + 2);
        B[2][2] = B[1][0] = WEBP_AVG2P(E + 2);
        B[2][3] = B[1][1] = WEBP_AVG3P(E + 3);
        B[1][2] = B[0][0] = WEBP_AVG2P(E + 3);
        B[1][3] = B[0][1] = WEBP_AVG3P(E + 4);
        B[0][2] = WEBP_AVG3P(E + 5);
        B[0][3] = WEBP_AVG3P(E + 6);
        break;
    default: /* WEBP_B_HU */
        B[0][0] = WEBP_AVG2P(L);
        B[0][1] = WEBP_AVG3P(L + 1);
        B[0][2] = B[1][0] = WEBP_AVG2P(L + 1);
        B[0][3] = B[1][1] = WEBP_AVG3P(L + 2);
        B[1][2] = B[2][0] = WEBP_AVG2P(L + 2);
        B[1][3] = B[2][1] = webp_avg3(L[2], L[3], L[3]);
        B[2][2] = B[2][3] = B[3][0] = B[3][1] = B[3][2] = B[3][3] = (uint8_t)L[3];
        break;
    }
#undef WEBP_AVG3P
#undef WEBP_AVG2P
    for (int r = 0; r < 4; r++) memcpy(dst + r * WEBP_BPS, B[r], 4);
}

/* Builds the prediction context of a macroblock from the (still unfiltered)
 * frame, predicts, adds the residuals and writes the result back.
 * Pixels outside the frame are 127 above and 129 on the left; the top-left
 * corner is 127 on the first row and 129 on the first column below it. */
static void webp_vp8_reconstruct_mb(webp_vp8_dec *d, const webp_vp8_mbinfo *mb,
                                    int mb_x, int mb_y, const int16_t *coeffs) {
    uint8_t ybuf[WEBP_BPS * 17];
    uint8_t ubuf[WEBP_BPS * 9];
    uint8_t vbuf[WEBP_BPS * 9];
    uint8_t *const y_dst = ybuf + WEBP_BPS + 8;
    uint8_t *const u_dst = ubuf + WEBP_BPS + 8;
    uint8_t *const v_dst = vbuf + WEBP_BPS + 8;
    const int ys = d->y_stride, uvs = d->uv_stride;
    uint8_t *const y_out = d->y + (size_t)mb_y * 16 * ys + (size_t)mb_x * 16;
    uint8_t *const u_out = d->u + (size_t)mb_y * 8 * uvs + (size_t)mb_x * 8;
    uint8_t *const v_out = d->v + (size_t)mb_y * 8 * uvs + (size_t)mb_x * 8;

    for (int j = 0; j < 16; j++)
        y_dst[j * WEBP_BPS - 1] = mb_x > 0 ? y_out[(ptrdiff_t)j * ys - 1] : 129;
    for (int j = 0; j < 8; j++) {
        u_dst[j * WEBP_BPS - 1] = mb_x > 0 ? u_out[(ptrdiff_t)j * uvs - 1] : 129;
        v_dst[j * WEBP_BPS - 1] = mb_x > 0 ? v_out[(ptrdiff_t)j * uvs - 1] : 129;
    }
    if (mb_y > 0) {
        y_dst[-WEBP_BPS - 1] = mb_x > 0 ? y_out[-(ptrdiff_t)ys - 1] : 129;
        u_dst[-WEBP_BPS - 1] = mb_x > 0 ? u_out[-(ptrdiff_t)uvs - 1] : 129;
        v_dst[-WEBP_BPS - 1] = mb_x > 0 ? v_out[-(ptrdiff_t)uvs - 1] : 129;
        memcpy(y_dst - WEBP_BPS, y_out - ys, 16);
        memcpy(u_dst - WEBP_BPS, u_out - uvs, 8);
        memcpy(v_dst - WEBP_BPS, v_out - uvs, 8);
        if (mb_x < d->mb_w - 1) {
            memcpy(y_dst - WEBP_BPS + 16, y_out - ys + 16, 4);
        } else {
            /* Rightmost macroblock: repeat the last pixel above. */
            memset(y_dst - WEBP_BPS + 16, y_out[-(ptrdiff_t)ys + 15], 4);
        }
    } else {
        memset(y_dst - WEBP_BPS - 1, 127, 16 + 4 + 1);
        memset(u_dst - WEBP_BPS - 1, 127, 8 + 1);
        memset(v_dst - WEBP_BPS - 1, 127, 8 + 1);
    }

    if (mb->is_i4x4) {
        /* Subblocks on the right column use the macroblock's above-right
         * pixels for every row, not pixels of the macroblock to the right. */
        for (int j = 1; j < 4; j++)
            memcpy(y_dst + (4 * j - 1) * WEBP_BPS + 16, y_dst - WEBP_BPS + 16, 4);
        for (int n = 0; n < 16; n++) {
            uint8_t *dst = y_dst + (n >> 2) * 4 * WEBP_BPS + (n & 3) * 4;
            webp_vp8_predict4(dst, mb->imodes[n]);
            if (!webp_vp8_block_is_zero(coeffs + 16 * n)) webp_vp8_idct_add(coeffs + 16 * n, dst);
        }
    } else {
        webp_vp8_predict_block(y_dst, 16, mb->ymode, mb_x, mb_y);
        for (int n = 0; n < 16; n++) {
            if (!webp_vp8_block_is_zero(coeffs + 16 * n))
                webp_vp8_idct_add(coeffs + 16 * n, y_dst + (n >> 2) * 4 * WEBP_BPS + (n & 3) * 4);
        }
    }
    webp_vp8_predict_block(u_dst, 8, mb->uvmode, mb_x, mb_y);
    webp_vp8_predict_block(v_dst, 8, mb->uvmode, mb_x, mb_y);
    for (int n = 0; n < 4; n++) {
        const int off = (n >> 1) * 4 * WEBP_BPS + (n & 1) * 4;
        if (!webp_vp8_block_is_zero(coeffs + 256 + 16 * n))
            webp_vp8_idct_add(coeffs + 256 + 16 * n, u_dst + off);
        if (!webp_vp8_block_is_zero(coeffs + 320 + 16 * n))
            webp_vp8_idct_add(coeffs + 320 + 16 * n, v_dst + off);
    }

    for (int j = 0; j < 16; j++) memcpy(y_out + (size_t)j * ys, y_dst + j * WEBP_BPS, 16);
    for (int j = 0; j < 8; j++) {
        memcpy(u_out + (size_t)j * uvs, u_dst + j * WEBP_BPS, 8);
        memcpy(v_out + (size_t)j * uvs, v_dst + j * WEBP_BPS, 8);
    }
}

/* ========================================================================
 * VP8 Loop Filter (RFC 6386 Section 15)
 * ======================================================================== */

static int webp_abs(int v) { return v < 0 ? -v : v; }
static int webp_sclip1(int v) { return v < -128 ? -128 : v > 127 ? 127 : v; }
static int webp_sclip2(int v) { return v < -16 ? -16 : v > 15 ? 15 : v; }

static void webp_vp8_do_filter2(uint8_t *p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0) + webp_sclip1(p1 - q1);
    const int a1 = webp_sclip2((a + 4) >> 3);
    const int a2 = webp_sclip2((a + 3) >> 3);
    p[-step] = webp_clip8(p0 + a2);
    p[0] = webp_clip8(q0 - a1);
}

static void webp_vp8_do_filter4(uint8_t *p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0);
    const int a1 = webp_sclip2((a + 4) >> 3);
    const int a2 = webp_sclip2((a + 3) >> 3);
    const int a3 = (a1 + 1) >> 1;
    p[-2 * step] = webp_clip8(p1 + a3);
    p[-step] = webp_clip8(p0 + a2);
    p[0] = webp_clip8(q0 - a1);
    p[step] = webp_clip8(q1 - a3);
}

static void webp_vp8_do_filter6(uint8_t *p, int step) {
    const int p2 = p[-3 * step], p1 = p[-2 * step], p0 = p[-step];
    const int q0 = p[0], q1 = p[step], q2 = p[2 * step];
    const int a = webp_sclip1(3 * (q0 - p0) + webp_sclip1(p1 - q1));
    const int a1 = (27 * a + 63) >> 7;
    const int a2 = (18 * a + 63) >> 7;
    const int a3 = (9 * a + 63) >> 7;
    p[-3 * step] = webp_clip8(p2 + a3);
    p[-2 * step] = webp_clip8(p1 + a2);
    p[-step] = webp_clip8(p0 + a1);
    p[0] = webp_clip8(q0 - a1);
    p[step] = webp_clip8(q1 - a2);
    p[2 * step] = webp_clip8(q2 - a3);
}

static int webp_vp8_hev(const uint8_t *p, int step, int thresh) {
    return webp_abs(p[-2 * step] - p[-step]) > thresh ||
           webp_abs(p[step] - p[0]) > thresh;
}

static int webp_vp8_needs_filter(const uint8_t *p, int step, int t) {
    return 4 * webp_abs(p[-step] - p[0]) + webp_abs(p[-2 * step] - p[step]) <= t;
}

static int webp_vp8_needs_filter2(const uint8_t *p, int step, int t, int it) {
    const int p3 = p[-4 * step], p2 = p[-3 * step], p1 = p[-2 * step];
    const int p0 = p[-step], q0 = p[0];
    const int q1 = p[step], q2 = p[2 * step], q3 = p[3 * step];
    if (4 * webp_abs(p0 - q0) + webp_abs(p1 - q1) > t) return 0;
    return webp_abs(p3 - p2) <= it && webp_abs(p2 - p1) <= it &&
           webp_abs(p1 - p0) <= it && webp_abs(q3 - q2) <= it &&
           webp_abs(q2 - q1) <= it && webp_abs(q1 - q0) <= it;
}

/* hstride crosses the edge, vstride walks along it. Macroblock edges use
 * the 6-tap filter, inner edges the 4-tap one. */
static void webp_vp8_filter_loop(uint8_t *p, int hstride, int vstride, int size,
                                 int thresh, int ithresh, int hev_t, int mb_edge) {
    const int thresh2 = 2 * thresh + 1;
    while (size-- > 0) {
        if (webp_vp8_needs_filter2(p, hstride, thresh2, ithresh)) {
            if (webp_vp8_hev(p, hstride, hev_t)) {
                webp_vp8_do_filter2(p, hstride);
            } else if (mb_edge) {
                webp_vp8_do_filter6(p, hstride);
            } else {
                webp_vp8_do_filter4(p, hstride);
            }
        }
        p += vstride;
    }
}

static void webp_vp8_simple_filter(uint8_t *p, int hstride, int vstride, int thresh) {
    const int thresh2 = 2 * thresh + 1;
    for (int i = 0; i < 16; i++) {
        if (webp_vp8_needs_filter(p, hstride, thresh2)) webp_vp8_do_filter2(p, hstride);
        p += vstride;
    }
}

static void webp_vp8_filter_mb(webp_vp8_dec *d, int mb_x, int mb_y) {
    const webp_vp8_finfo *f = &d->finfo[(size_t)mb_y * d->mb_w + mb_x];
    const int ys = d->y_stride, uvs = d->uv_stride;
    uint8_t *y = d->y + (size_t)mb_y * 16 * ys + (size_t)mb_x * 16;
    const int limit = f->limit;
    const int ilevel = f->ilevel;
    const int hev_t = f->hev_thresh;
    if (limit == 0) return;
    if (d->filter_type == 1) {
        if (mb_x > 0) webp_vp8_simple_filter(y, 1, ys, limit + 4);
        if (f->inner)
            for (int k = 1; k < 4; k++) webp_vp8_simple_filter(y + 4 * k, 1, ys, limit);
        if (mb_y > 0) webp_vp8_simple_filter(y, ys, 1, limit + 4);
        if (f->inner)
            for (int k = 1; k < 4; k++) webp_vp8_simple_filter(y + (size_t)4 * k * ys, ys, 1, limit);
        return;
    }
    uint8_t *u = d->u + (size_t)mb_y * 8 * uvs + (size_t)mb_x * 8;
    uint8_t *v = d->v + (size_t)mb_y * 8 * uvs + (size_t)mb_x * 8;
    if (mb_x > 0) {
        webp_vp8_filter_loop(y, 1, ys, 16, limit + 4, ilevel, hev_t, 1);
        webp_vp8_filter_loop(u, 1, uvs, 8, limit + 4, ilevel, hev_t, 1);
        webp_vp8_filter_loop(v, 1, uvs, 8, limit + 4, ilevel, hev_t, 1);
    }
    if (f->inner) {
        for (int k = 1; k < 4; k++)
            webp_vp8_filter_loop(y + 4 * k, 1, ys, 16, limit, ilevel, hev_t, 0);
        webp_vp8_filter_loop(u + 4, 1, uvs, 8, limit, ilevel, hev_t, 0);
        webp_vp8_filter_loop(v + 4, 1, uvs, 8, limit, ilevel, hev_t, 0);
    }
    if (mb_y > 0) {
        webp_vp8_filter_loop(y, ys, 1, 16, limit + 4, ilevel, hev_t, 1);
        webp_vp8_filter_loop(u, uvs, 1, 8, limit + 4, ilevel, hev_t, 1);
        webp_vp8_filter_loop(v, uvs, 1, 8, limit + 4, ilevel, hev_t, 1);
    }
    if (f->inner) {
        for (int k = 1; k < 4; k++)
            webp_vp8_filter_loop(y + (size_t)4 * k * ys, ys, 1, 16, limit, ilevel, hev_t, 0);
        webp_vp8_filter_loop(u + (size_t)4 * uvs, uvs, 1, 8, limit, ilevel, hev_t, 0);
        webp_vp8_filter_loop(v + (size_t)4 * uvs, uvs, 1, 8, limit, ilevel, hev_t, 0);
    }
}

/* ========================================================================
 * VP8 YUV 4:2:0 to RGB (libwebp's default "fancy" upsampling)
 * ======================================================================== */

static int webp_mul_hi(int v, int coeff) { return (v * coeff) >> 8; }

static uint8_t webp_yuv_clip(int v) {
    return (uint8_t)(((v & ~16383) == 0) ? (v >> 6) : (v < 0) ? 0 : 255);
}

static void webp_yuv_to_rgb(int y, int u, int v, uint8_t *rgb) {
    rgb[0] = webp_yuv_clip(webp_mul_hi(y, 19077) + webp_mul_hi(v, 26149) - 14234);
    rgb[1] = webp_yuv_clip(webp_mul_hi(y, 19077) - webp_mul_hi(u, 6419) -
                           webp_mul_hi(v, 13320) + 8708);
    rgb[2] = webp_yuv_clip(webp_mul_hi(y, 19077) + webp_mul_hi(u, 33050) - 17685);
}

/* Converts two luma rows sharing the chroma rows above (top_*) and below
 * (cur_*) them, interpolating chroma with 9-3-3-1 weights. 'step' is the
 * output pixel size (3 or 4); alpha bytes are left untouched. */
static void webp_upsample_pair(const uint8_t *top_y, const uint8_t *bottom_y,
                               const uint8_t *top_u, const uint8_t *top_v,
                               const uint8_t *cur_u, const uint8_t *cur_v,
                               uint8_t *top_dst, uint8_t *bottom_dst, int len, int step) {
#define WEBP_LOAD_UV(u, v) ((uint32_t)(u) | ((uint32_t)(v) << 16))
    const int last_pixel_pair = (len - 1) >> 1;
    uint32_t tl_uv = WEBP_LOAD_UV(top_u[0], top_v[0]);
    uint32_t l_uv = WEBP_LOAD_UV(cur_u[0], cur_v[0]);
    {
        const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;
        webp_yuv_to_rgb(top_y[0], uv0 & 0xff, (int)(uv0 >> 16), top_dst);
    }
    if (bottom_y) {
        const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;
        webp_yuv_to_rgb(bottom_y[0], uv0 & 0xff, (int)(uv0 >> 16), bottom_dst);
    }
    for (int x = 1; x <= last_pixel_pair; x++) {
        const uint32_t t_uv = WEBP_LOAD_UV(top_u[x], top_v[x]);
        const uint32_t uv = WEBP_LOAD_UV(cur_u[x], cur_v[x]);
        const uint32_t avg = tl_uv + t_uv + l_uv + uv + 0x00080008u;
        const uint32_t diag_12 = (avg + 2 * (t_uv + l_uv)) >> 3;
        const uint32_t diag_03 = (avg + 2 * (tl_uv + uv)) >> 3;
        {
            const uint32_t uv0 = (diag_12 + tl_uv) >> 1;
            const uint32_t uv1 = (diag_03 + t_uv) >> 1;
            webp_yuv_to_rgb(top_y[2 * x - 1], uv0 & 0xff, (int)(uv0 >> 16),
                            top_dst + (2 * x - 1) * step);
            webp_yuv_to_rgb(top_y[2 * x], uv1 & 0xff, (int)(uv1 >> 16),
                            top_dst + (2 * x) * step);
        }
        if (bottom_y) {
            const uint32_t uv0 = (diag_03 + l_uv) >> 1;
            const uint32_t uv1 = (diag_12 + uv) >> 1;
            webp_yuv_to_rgb(bottom_y[2 * x - 1], uv0 & 0xff, (int)(uv0 >> 16),
                            bottom_dst + (2 * x - 1) * step);
            webp_yuv_to_rgb(bottom_y[2 * x], uv1 & 0xff, (int)(uv1 >> 16),
                            bottom_dst + (2 * x) * step);
        }
        tl_uv = t_uv;
        l_uv = uv;
    }
    if (!(len & 1)) {
        {
            const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;
            webp_yuv_to_rgb(top_y[len - 1], uv0 & 0xff, (int)(uv0 >> 16),
                            top_dst + (len - 1) * step);
        }
        if (bottom_y) {
            const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;
            webp_yuv_to_rgb(bottom_y[len - 1], uv0 & 0xff, (int)(uv0 >> 16),
                            bottom_dst + (len - 1) * step);
        }
    }
#undef WEBP_LOAD_UV
}

static void webp_vp8_emit_rgb(const webp_vp8_dec *d, uint8_t *out, int step) {
    const int w = d->width, h = d->height;
    const size_t ys = (size_t)d->y_stride, uvs = (size_t)d->uv_stride;
    const size_t row = (size_t)w * step;
    webp_upsample_pair(d->y, NULL, d->u, d->v, d->u, d->v, out, NULL, w, step);
    for (int k = 1; 2 * k - 1 < h; k++) {
        const uint8_t *top_u = d->u + (size_t)(k - 1) * uvs;
        const uint8_t *top_v = d->v + (size_t)(k - 1) * uvs;
        const uint8_t *top_y = d->y + (size_t)(2 * k - 1) * ys;
        uint8_t *top_dst = out + (size_t)(2 * k - 1) * row;
        if (2 * k < h) {
            webp_upsample_pair(top_y, top_y + ys, top_u, top_v, top_u + uvs, top_v + uvs,
                               top_dst, top_dst + row, w, step);
        } else {
            /* Last row of an even-height image. */
            webp_upsample_pair(top_y, NULL, top_u, top_v, top_u, top_v, top_dst, NULL, w, step);
        }
    }
}

/* ========================================================================
 * VP8 Frame Decoding
 * ======================================================================== */

static void webp_vp8_dec_free(webp_vp8_dec *d) {
    free(d->y);
    free(d->u);
    free(d->v);
    free(d->mbs);
    free(d->intra_t);
    free(d->nz_top);
    free(d->finfo);
}

/* Reads the frame size from a VP8 key frame header. As in libwebp, the
 * first partition must fit in the declared chunk size. */
static int webp_vp8_get_size(const uint8_t *data, size_t size, size_t chunk_size,
                             int *width, int *height) {
    if (size < 10) return 0;
    const uint32_t bits = webp_le24(data);
    if ((bits & 1) || ((bits >> 1) & 7) > 3 || !((bits >> 4) & 1)) return 0;
    if ((bits >> 5) >= chunk_size) return 0;
    if (data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a) return 0;
    *width = (int)(((uint32_t)data[7] << 8 | data[6]) & 0x3fff);
    *height = (int)(((uint32_t)data[9] << 8 | data[8]) & 0x3fff);
    return webp_dimensions_ok((uint32_t)*width, (uint32_t)*height);
}

/* Decodes a 'VP8 ' bitstream to RGB or RGBA (alpha bytes set to 255). */
static uint8_t *webp_vp8_decode(const uint8_t *data, size_t size, size_t chunk_size,
                                int channels, int *width, int *height) {
    webp_vp8_dec d;
    int16_t coeffs[384];
    uint8_t *pixels = NULL;

    memset(&d, 0, sizeof(d));
    if (!webp_vp8_get_size(data, size, chunk_size, &d.width, &d.height)) return NULL;
    const size_t part0 = webp_le24(data) >> 5;
    data += 10;
    size -= 10;
    if (part0 > size) return NULL;

    d.mb_w = (d.width + 15) >> 4;
    d.mb_h = (d.height + 15) >> 4;
    d.absolute_delta = 1;
    d.segment_probs[0] = d.segment_probs[1] = d.segment_probs[2] = 255;

    webp_vp8_br_init(&d.br, data, part0);
    (void)webp_vp8_value(&d.br, 1);   /* color space */
    (void)webp_vp8_value(&d.br, 1);   /* clamping type */
    if (!webp_vp8_parse_segment_header(&d) || !webp_vp8_parse_filter_header(&d)) return NULL;
    if (!webp_vp8_parse_partitions(&d, data + part0, size - part0)) return NULL;
    webp_vp8_parse_quant(&d);
    (void)webp_vp8_value(&d.br, 1);   /* refresh_entropy_probs */
    webp_vp8_parse_proba(&d);
    webp_vp8_precompute_filter_strengths(&d);

    d.y_stride = d.mb_w * 16;
    d.uv_stride = d.mb_w * 8;
    d.y = (uint8_t *)malloc((size_t)d.y_stride * d.mb_h * 16);
    d.u = (uint8_t *)malloc((size_t)d.uv_stride * d.mb_h * 8);
    d.v = (uint8_t *)malloc((size_t)d.uv_stride * d.mb_h * 8);
    d.mbs = (webp_vp8_mbinfo *)calloc((size_t)d.mb_w, sizeof(*d.mbs));
    d.intra_t = (uint8_t *)malloc((size_t)d.mb_w * 4);
    d.nz_top = (uint8_t *)calloc((size_t)d.mb_w, 9);
    d.finfo = (webp_vp8_finfo *)calloc((size_t)d.mb_w * d.mb_h, sizeof(*d.finfo));
    pixels = (uint8_t *)malloc((size_t)d.width * d.height * channels);
    if (!d.y || !d.u || !d.v || !d.mbs || !d.intra_t || !d.nz_top || !d.finfo || !pixels)
        goto fail;
    memset(d.intra_t, WEBP_B_DC, (size_t)d.mb_w * 4);

    for (int mb_y = 0; mb_y < d.mb_h; mb_y++) {
        webp_vp8_br *token_br = &d.parts[mb_y & d.num_parts_minus_one];
        memset(d.intra_l, WEBP_B_DC, sizeof(d.intra_l));
        memset(d.nz_left, 0, sizeof(d.nz_left));
        for (int mb_x = 0; mb_x < d.mb_w; mb_x++)
            webp_vp8_parse_intra_mode(&d, &d.mbs[mb_x], mb_x);
        if (d.br.eof) goto fail;
        for (int mb_x = 0; mb_x < d.mb_w; mb_x++) {
            const webp_vp8_mbinfo *mb = &d.mbs[mb_x];
            int has_data = 0;
            if (!mb->skip) {
                has_data = webp_vp8_parse_residuals(&d, mb, mb_x, token_br, coeffs);
            } else {
                uint8_t *tnz = d.nz_top + 9 * mb_x;
                memset(coeffs, 0, sizeof(coeffs));
                memset(tnz, 0, 8);
                memset(d.nz_left, 0, 8);
                if (!mb->is_i4x4) tnz[8] = d.nz_left[8] = 0;
            }
            if (token_br->eof) goto fail;
            if (d.filter_type > 0) {
                webp_vp8_finfo *f = &d.finfo[(size_t)mb_y * d.mb_w + mb_x];
                *f = d.fstrengths[mb->segment][mb->is_i4x4];
                f->inner |= (uint8_t)has_data;
            }
            webp_vp8_reconstruct_mb(&d, mb, mb_x, mb_y, coeffs);
        }
    }

    /* Intra prediction reads unfiltered pixels, so the loop filter runs
     * once the whole frame is reconstructed, in macroblock raster order. */
    if (d.filter_type > 0) {
        for (int mb_y = 0; mb_y < d.mb_h; mb_y++)
            for (int mb_x = 0; mb_x < d.mb_w; mb_x++) webp_vp8_filter_mb(&d, mb_x, mb_y);
    }

    if (channels == 4) {
        const size_t n = (size_t)d.width * d.height;
        for (size_t i = 0; i < n; i++) pixels[4 * i + 3] = 255;
    }
    webp_vp8_emit_rgb(&d, pixels, channels);
    *width = d.width;
    *height = d.height;
    webp_vp8_dec_free(&d);
    return pixels;

fail:
    free(pixels);
    webp_vp8_dec_free(&d);
    return NULL;
}

/* ========================================================================
 * VP8L (Lossless) - RFC 9649 Section 3
 * ======================================================================== */

/* Distance code -> (xi, yi) from RFC 9649 Figure 20; dist = xi + yi * width. */
static const int8_t webp_vp8l_distance_map[120][2] = {
    {0, 1}, {1, 0}, {1, 1}, {-1, 1}, {0, 2}, {2, 0}, {1, 2}, {-1, 2},
    {2, 1}, {-2, 1}, {2, 2}, {-2, 2}, {0, 3}, {3, 0}, {1, 3}, {-1, 3},
    {3, 1}, {-3, 1}, {2, 3}, {-2, 3}, {3, 2}, {-3, 2}, {0, 4}, {4, 0},
    {1, 4}, {-1, 4}, {4, 1}, {-4, 1}, {3, 3}, {-3, 3}, {2, 4}, {-2, 4},
    {4, 2}, {-4, 2}, {0, 5}, {3, 4}, {-3, 4}, {4, 3}, {-4, 3}, {5, 0},
    {1, 5}, {-1, 5}, {5, 1}, {-5, 1}, {2, 5}, {-2, 5}, {5, 2}, {-5, 2},
    {4, 4}, {-4, 4}, {3, 5}, {-3, 5}, {5, 3}, {-5, 3}, {0, 6}, {6, 0},
    {1, 6}, {-1, 6}, {6, 1}, {-6, 1}, {2, 6}, {-2, 6}, {6, 2}, {-6, 2},
    {4, 5}, {-4, 5}, {5, 4}, {-5, 4}, {3, 6}, {-3, 6}, {6, 3}, {-6, 3},
    {0, 7}, {7, 0}, {1, 7}, {-1, 7}, {5, 5}, {-5, 5}, {7, 1}, {-7, 1},
    {4, 6}, {-4, 6}, {6, 4}, {-6, 4}, {2, 7}, {-2, 7}, {7, 2}, {-7, 2},
    {3, 7}, {-3, 7}, {7, 3}, {-7, 3}, {5, 6}, {-5, 6}, {6, 5}, {-6, 5},
    {8, 0}, {4, 7}, {-4, 7}, {7, 4}, {-7, 4}, {8, 1}, {8, 2}, {6, 6},
    {-6, 6}, {8, 3}, {5, 7}, {-5, 7}, {7, 5}, {-7, 5}, {8, 4}, {6, 7},
    {-6, 7}, {7, 6}, {-7, 6}, {8, 5}, {7, 7}, {-7, 7}, {8, 6}, {8, 7},
};

#define WEBP_NUM_LITERAL_CODES 256
#define WEBP_NUM_LENGTH_CODES 24
#define WEBP_NUM_DISTANCE_CODES 40
#define WEBP_MAX_CACHE_BITS 11
#define WEBP_MAX_ALPHABET \
    (WEBP_NUM_LITERAL_CODES + WEBP_NUM_LENGTH_CODES + (1 << WEBP_MAX_CACHE_BITS))
#define WEBP_MAX_CODE_LENGTH 15
#define WEBP_FAST_BITS 8

enum {
    WEBP_PREDICTOR_TRANSFORM = 0,
    WEBP_CROSS_COLOR_TRANSFORM = 1,
    WEBP_SUBTRACT_GREEN_TRANSFORM = 2,
    WEBP_COLOR_INDEXING_TRANSFORM = 3
};

/* Little-endian bit reader. Reads past the end return zeros and set eos. */
typedef struct {
    const uint8_t *buf;
    size_t len;
    size_t pos;         /* next byte to load */
    uint64_t val;
    int nbits;
    int eos;
} webp_vp8l_br;

static void webp_vp8l_fill(webp_vp8l_br *br) {
    while (br->nbits <= 56) {
        const uint64_t byte = br->pos < br->len ? br->buf[br->pos] : 0;
        br->val |= byte << br->nbits;
        br->nbits += 8;
        br->pos++;
    }
}

static uint32_t webp_vp8l_peek(webp_vp8l_br *br, int n) {
    if (br->nbits < n) webp_vp8l_fill(br);
    return (uint32_t)(br->val & ((1ull << n) - 1));
}

static void webp_vp8l_skip(webp_vp8l_br *br, int n) {
    br->val >>= n;
    br->nbits -= n;
    if ((uint64_t)br->pos * 8 - (uint64_t)br->nbits > (uint64_t)br->len * 8) br->eos = 1;
}

static uint32_t webp_vp8l_read(webp_vp8l_br *br, int n) {
    if (n == 0) return 0;
    const uint32_t v = webp_vp8l_peek(br, n);
    webp_vp8l_skip(br, n);
    return v;
}

/* Canonical prefix code: an 8-bit lookup table for short codes, and a
 * count-based canonical walk for the longer ones. */
typedef struct {
    int single;                             /* >= 0: only symbol, zero bits */
    uint16_t fast[1 << WEBP_FAST_BITS];     /* (len << 12) | symbol, 0 = slow */
    uint16_t count[WEBP_MAX_CODE_LENGTH + 1];
    uint16_t *symbols;                      /* sorted by (length, symbol) */
} webp_vp8l_code;

static void webp_vp8l_code_free(webp_vp8l_code *c) {
    free(c->symbols);
    c->symbols = NULL;
}

static int webp_vp8l_build_code(webp_vp8l_code *c, const uint8_t *lengths, int n) {
    uint16_t offset[WEBP_MAX_CODE_LENGTH + 2];
    int next_code[WEBP_MAX_CODE_LENGTH + 1];
    int num = 0, last = -1;

    memset(c, 0, sizeof(*c));
    c->single = -1;
    for (int i = 0; i < n; i++) {
        if (lengths[i]) {
            c->count[lengths[i]]++;
            num++;
            last = i;
        }
    }
    if (num == 0) return 0;
    if (num == 1) {
        c->single = last;
        return 1;
    }
    int left = 1;
    for (int len = 1; len <= WEBP_MAX_CODE_LENGTH; len++) {
        left <<= 1;
        left -= c->count[len];
        if (left < 0) return 0;     /* over-subscribed */
    }
    if (left != 0) return 0;        /* incomplete */

    c->symbols = (uint16_t *)malloc((size_t)num * sizeof(*c->symbols));
    if (!c->symbols) return 0;
    offset[1] = 0;
    for (int len = 1; len <= WEBP_MAX_CODE_LENGTH; len++)
        offset[len + 1] = (uint16_t)(offset[len] + c->count[len]);
    next_code[0] = 0;
    next_code[1] = 0;
    for (int len = 2; len <= WEBP_MAX_CODE_LENGTH; len++)
        next_code[len] = (next_code[len - 1] + c->count[len - 1]) << 1;
    for (int i = 0; i < n; i++) {
        const int l = lengths[i];
        if (!l) continue;
        c->symbols[offset[l]++] = (uint16_t)i;
        if (l <= WEBP_FAST_BITS) {
            const int code = next_code[l];
            int rev = 0;
            for (int b = 0; b < l; b++) rev |= ((code >> b) & 1) << (l - 1 - b);
            for (int k = rev; k < (1 << WEBP_FAST_BITS); k += 1 << l)
                c->fast[k] = (uint16_t)((l << 12) | i);
        }
        next_code[l]++;
    }
    return 1;
}

static int webp_vp8l_read_symbol(webp_vp8l_br *br, const webp_vp8l_code *c) {
    if (c->single >= 0) return c->single;
    const uint32_t bits = webp_vp8l_peek(br, WEBP_MAX_CODE_LENGTH);
    const uint16_t e = c->fast[bits & ((1 << WEBP_FAST_BITS) - 1)];
    if (e) {
        webp_vp8l_skip(br, e >> 12);
        return e & 0xfff;
    }
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= WEBP_MAX_CODE_LENGTH; len++) {
        const int count = c->count[len];
        code |= (int)((bits >> (len - 1)) & 1);
        if (code - first < count) {
            webp_vp8l_skip(br, len);
            return c->symbols[index + code - first];
        }
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;   /* unreachable for complete codes */
}

static const uint8_t webp_vp8l_code_length_order[19] = {
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

static int webp_vp8l_read_code_lengths(webp_vp8l_br *br, const uint8_t cl_lengths[19],
                                       int num_symbols, uint8_t *code_lengths) {
    webp_vp8l_code clc;
    int max_symbol, symbol = 0, prev_len = 8, ok = 0;
    if (!webp_vp8l_build_code(&clc, cl_lengths, 19)) return 0;
    if (webp_vp8l_read(br, 1)) {
        const int length_nbits = 2 + 2 * (int)webp_vp8l_read(br, 3);
        max_symbol = 2 + (int)webp_vp8l_read(br, length_nbits);
        if (max_symbol > num_symbols) goto done;
    } else {
        max_symbol = num_symbols;
    }
    while (symbol < num_symbols) {
        if (max_symbol-- == 0) break;
        const int code_len = webp_vp8l_read_symbol(br, &clc);
        if (code_len < 0 || br->eos) goto done;
        if (code_len < 16) {
            code_lengths[symbol++] = (uint8_t)code_len;
            if (code_len) prev_len = code_len;
        } else {
            static const uint8_t extra_bits[3] = { 2, 3, 7 };
            static const uint8_t repeat_offset[3] = { 3, 3, 11 };
            const int slot = code_len - 16;
            int repeat = (int)webp_vp8l_read(br, extra_bits[slot]) + repeat_offset[slot];
            const int value = code_len == 16 ? prev_len : 0;
            if (symbol + repeat > num_symbols) goto done;
            while (repeat-- > 0) code_lengths[symbol++] = (uint8_t)value;
        }
    }
    ok = !br->eos;
done:
    webp_vp8l_code_free(&clc);
    return ok;
}

static int webp_vp8l_read_code(webp_vp8l_br *br, int alphabet_size, webp_vp8l_code *code) {
    uint8_t code_lengths[WEBP_MAX_ALPHABET];
    memset(code_lengths, 0, (size_t)alphabet_size);
    if (webp_vp8l_read(br, 1)) {
        /* Simple code: one or two symbols with code length 1. */
        const int num_symbols = (int)webp_vp8l_read(br, 1) + 1;
        const int first_is_8bits = (int)webp_vp8l_read(br, 1);
        int symbol = (int)webp_vp8l_read(br, first_is_8bits ? 8 : 1);
        if (symbol < alphabet_size) code_lengths[symbol] = 1;
        if (num_symbols == 2) {
            symbol = (int)webp_vp8l_read(br, 8);
            if (symbol < alphabet_size) code_lengths[symbol] = 1;
        }
    } else {
        uint8_t cl_lengths[19] = {0};
        const int num_codes = (int)webp_vp8l_read(br, 4) + 4;
        for (int i = 0; i < num_codes; i++)
            cl_lengths[webp_vp8l_code_length_order[i]] = (uint8_t)webp_vp8l_read(br, 3);
        if (!webp_vp8l_read_code_lengths(br, cl_lengths, alphabet_size, code_lengths)) return 0;
    }
    if (br->eos) return 0;
    return webp_vp8l_build_code(code, code_lengths, alphabet_size);
}

/* Prefix codes #1..#5: green/length/cache, red, blue, alpha, distance. */
typedef struct {
    webp_vp8l_code codes[5];
} webp_vp8l_group;

typedef struct {
    int type;
    int bits;
    int xsize;          /* width of this transform's output */
    uint32_t *data;
} webp_vp8l_transform;

typedef struct {
    webp_vp8l_br br;
    int height;
    unsigned transforms_seen;
    int num_transforms;
    webp_vp8l_transform transforms[4];
} webp_vp8l_dec;

static int webp_vp8l_subsample(int size, int bits) {
    return (size + (1 << bits) - 1) >> bits;
}

static void webp_vp8l_free_groups(webp_vp8l_group *groups, int num) {
    if (!groups) return;
    for (int i = 0; i < num; i++)
        for (int j = 0; j < 5; j++) webp_vp8l_code_free(&groups[i].codes[j]);
    free(groups);
}

static int webp_vp8l_copy_value(webp_vp8l_br *br, int symbol) {
    if (symbol < 4) return symbol + 1;
    const int extra_bits = (symbol - 2) >> 1;
    const int offset = (2 + (symbol & 1)) << extra_bits;
    return offset + (int)webp_vp8l_read(br, extra_bits) + 1;
}

static int webp_vp8l_plane_code_to_distance(int xsize, int plane_code) {
    if (plane_code > 120) return plane_code - 120;
    const int dist = webp_vp8l_distance_map[plane_code - 1][0] +
                     webp_vp8l_distance_map[plane_code - 1][1] * xsize;
    return dist >= 1 ? dist : 1;
}

static void webp_vp8l_cache_insert(uint32_t *cache, int cache_bits, uint32_t argb) {
    if (cache) cache[(0x1e35a7bdu * argb) >> (32 - cache_bits)] = argb;
}

static int webp_vp8l_decode_image_stream(webp_vp8l_dec *d, int xsize, int ysize,
                                         int is_level0, uint32_t **out);

/* Reads the prefix code groups and decodes width x height ARGB pixels. */
static int webp_vp8l_decode_pixels(webp_vp8l_dec *d, int width, int height, int is_level0,
                                   int cache_bits, uint32_t **out) {
    webp_vp8l_br *br = &d->br;
    uint32_t *meta = NULL;
    int meta_bits = 0, meta_xsize = 0;
    int num_groups = 1;
    webp_vp8l_group *groups = NULL;
    uint32_t *cache = NULL;
    uint32_t *pixels = NULL;
    const size_t total = (size_t)width * height;
    size_t pos = 0;
    int ok = 0;

    if (is_level0 && webp_vp8l_read(br, 1)) {
        meta_bits = (int)webp_vp8l_read(br, 3) + 2;
        meta_xsize = webp_vp8l_subsample(width, meta_bits);
        const int meta_ysize = webp_vp8l_subsample(height, meta_bits);
        if (!webp_vp8l_decode_image_stream(d, meta_xsize, meta_ysize, 0, &meta)) goto done;
        const size_t n = (size_t)meta_xsize * meta_ysize;
        for (size_t k = 0; k < n; k++) {
            const int group = (int)((meta[k] >> 8) & 0xffff);
            meta[k] = (uint32_t)group;
            if (group >= num_groups) num_groups = group + 1;
        }
    }
    if (br->eos) goto done;

    groups = (webp_vp8l_group *)calloc((size_t)num_groups, sizeof(*groups));
    if (!groups) goto done;
    for (int i = 0; i < num_groups; i++) {
        static const int base_sizes[5] = {
            WEBP_NUM_LITERAL_CODES + WEBP_NUM_LENGTH_CODES, 256, 256, 256,
            WEBP_NUM_DISTANCE_CODES
        };
        for (int j = 0; j < 5; j++) {
            const int alphabet = base_sizes[j] + (j == 0 && cache_bits ? 1 << cache_bits : 0);
            if (!webp_vp8l_read_code(br, alphabet, &groups[i].codes[j])) goto done;
        }
    }

    if (cache_bits) {
        cache = (uint32_t *)calloc((size_t)1 << cache_bits, sizeof(*cache));
        if (!cache) goto done;
    }
    pixels = (uint32_t *)malloc(total * sizeof(*pixels));
    if (!pixels) goto done;

    while (pos < total) {
        const int x = (int)(pos % (size_t)width);
        const int y = (int)(pos / (size_t)width);
        const webp_vp8l_group *g = meta
            ? &groups[meta[(size_t)(y >> meta_bits) * meta_xsize + (x >> meta_bits)]]
            : &groups[0];
        const int code = webp_vp8l_read_symbol(br, &g->codes[0]);
        if (code < 0) goto done;
        if (code < WEBP_NUM_LITERAL_CODES) {
            const int red = webp_vp8l_read_symbol(br, &g->codes[1]);
            const int blue = webp_vp8l_read_symbol(br, &g->codes[2]);
            const int alpha = webp_vp8l_read_symbol(br, &g->codes[3]);
            if (red < 0 || blue < 0 || alpha < 0) goto done;
            pixels[pos] = ((uint32_t)alpha << 24) | ((uint32_t)red << 16) |
                          ((uint32_t)code << 8) | (uint32_t)blue;
            webp_vp8l_cache_insert(cache, cache_bits, pixels[pos]);
            pos++;
        } else if (code < WEBP_NUM_LITERAL_CODES + WEBP_NUM_LENGTH_CODES) {
            const int length = webp_vp8l_copy_value(br, code - WEBP_NUM_LITERAL_CODES);
            const int dist_symbol = webp_vp8l_read_symbol(br, &g->codes[4]);
            if (dist_symbol < 0) goto done;
            const int dist = webp_vp8l_plane_code_to_distance(
                width, webp_vp8l_copy_value(br, dist_symbol));
            if (br->eos) goto done;
            if (pos < (size_t)dist || total - pos < (size_t)length) goto done;
            for (int k = 0; k < length; k++) {
                pixels[pos] = pixels[pos - (size_t)dist];
                webp_vp8l_cache_insert(cache, cache_bits, pixels[pos]);
                pos++;
            }
        } else {
            const int key = code - (WEBP_NUM_LITERAL_CODES + WEBP_NUM_LENGTH_CODES);
            if (!cache || key >= (1 << cache_bits)) goto done;
            pixels[pos] = cache[key];
            webp_vp8l_cache_insert(cache, cache_bits, pixels[pos]);
            pos++;
        }
        if (br->eos) goto done;
    }
    ok = 1;

done:
    free(meta);
    free(cache);
    webp_vp8l_free_groups(groups, num_groups);
    if (!ok) {
        free(pixels);
        pixels = NULL;
    }
    *out = pixels;
    return ok;
}

static int webp_vp8l_read_transform(webp_vp8l_dec *d, int *xsize) {
    webp_vp8l_br *br = &d->br;
    const int type = (int)webp_vp8l_read(br, 2);
    if (d->transforms_seen & (1u << type)) return 0;   /* each type at most once */
    d->transforms_seen |= 1u << type;
    webp_vp8l_transform *t = &d->transforms[d->num_transforms++];
    t->type = type;
    t->xsize = *xsize;
    t->bits = 0;
    t->data = NULL;
    switch (type) {
    case WEBP_PREDICTOR_TRANSFORM:
    case WEBP_CROSS_COLOR_TRANSFORM:
        t->bits = (int)webp_vp8l_read(br, 3) + 2;
        return webp_vp8l_decode_image_stream(d, webp_vp8l_subsample(t->xsize, t->bits),
                                             webp_vp8l_subsample(d->height, t->bits), 0,
                                             &t->data);
    case WEBP_COLOR_INDEXING_TRANSFORM: {
        const int num_colors = (int)webp_vp8l_read(br, 8) + 1;
        const int bits = num_colors > 16 ? 0 : num_colors > 4 ? 1 : num_colors > 2 ? 2 : 3;
        const int final_num_colors = 1 << (8 >> bits);
        uint32_t *palette = NULL;
        *xsize = webp_vp8l_subsample(t->xsize, bits);
        t->bits = bits;
        if (!webp_vp8l_decode_image_stream(d, num_colors, 1, 0, &palette)) return 0;
        t->data = (uint32_t *)calloc((size_t)final_num_colors, sizeof(*t->data));
        if (!t->data) {
            free(palette);
            return 0;
        }
        /* The palette is delta coded; entries past num_colors stay 0. */
        t->data[0] = palette[0];
        for (int i = 1; i < num_colors; i++) {
            const uint32_t a = palette[i], b = t->data[i - 1];
            t->data[i] = (((a & 0xff00ff00u) + (b & 0xff00ff00u)) & 0xff00ff00u) |
                         (((a & 0x00ff00ffu) + (b & 0x00ff00ffu)) & 0x00ff00ffu);
        }
        free(palette);
        return 1;
    }
    default:   /* subtract green: no data */
        return 1;
    }
}

/* Decodes an image stream. Only the main image (level 0) carries transforms
 * and meta prefix codes; *out is that image's pixels before the inverse
 * transforms, at the width left by the color indexing transform. */
static int webp_vp8l_decode_image_stream(webp_vp8l_dec *d, int xsize, int ysize,
                                         int is_level0, uint32_t **out) {
    webp_vp8l_br *br = &d->br;
    int transform_xsize = xsize;
    int cache_bits = 0;
    *out = NULL;
    if (is_level0) {
        while (webp_vp8l_read(br, 1)) {
            if (br->eos || !webp_vp8l_read_transform(d, &transform_xsize)) return 0;
        }
    }
    if (webp_vp8l_read(br, 1)) {
        cache_bits = (int)webp_vp8l_read(br, 4);
        if (cache_bits < 1 || cache_bits > WEBP_MAX_CACHE_BITS) return 0;
    }
    if (br->eos) return 0;
    return webp_vp8l_decode_pixels(d, transform_xsize, ysize, is_level0, cache_bits, out);
}

/* ========================================================================
 * VP8L Inverse Transforms
 * ======================================================================== */

static uint32_t webp_vp8l_add_pixels(uint32_t a, uint32_t b) {
    const uint32_t ag = (a & 0xff00ff00u) + (b & 0xff00ff00u);
    const uint32_t rb = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
    return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

static uint32_t webp_vp8l_average2(uint32_t a, uint32_t b) {
    return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b);
}

static int webp_vp8l_sub3(int a, int b, int c) {
    return webp_abs(b - c) - webp_abs(a - c);
}

/* Returns T or L, whichever is closer to the gradient estimate L + T - TL. */
static uint32_t webp_vp8l_select(uint32_t t, uint32_t l, uint32_t tl) {
    int s = 0;
    for (int shift = 0; shift < 32; shift += 8)
        s += webp_vp8l_sub3((int)((t >> shift) & 0xff), (int)((l >> shift) & 0xff),
                            (int)((tl >> shift) & 0xff));
    return s <= 0 ? t : l;
}

static uint32_t webp_vp8l_clamped_add_subtract_full(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        const int v = (int)((a >> shift) & 0xff) + (int)((b >> shift) & 0xff) -
                      (int)((c >> shift) & 0xff);
        r |= (uint32_t)webp_clip(v, 255) << shift;
    }
    return r;
}

static uint32_t webp_vp8l_clamped_add_subtract_half(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t ave = webp_vp8l_average2(a, b);
    uint32_t r = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        const int x = (int)((ave >> shift) & 0xff);
        const int y = (int)((c >> shift) & 0xff);
        r |= (uint32_t)webp_clip(x + (x - y) / 2, 255) << shift;
    }
    return r;
}

static uint32_t webp_vp8l_predict(int mode, uint32_t l, uint32_t t, uint32_t tl, uint32_t tr) {
    switch (mode) {
    case 1: return l;
    case 2: return t;
    case 3: return tr;
    case 4: return tl;
    case 5: return webp_vp8l_average2(webp_vp8l_average2(l, tr), t);
    case 6: return webp_vp8l_average2(l, tl);
    case 7: return webp_vp8l_average2(l, t);
    case 8: return webp_vp8l_average2(tl, t);
    case 9: return webp_vp8l_average2(t, tr);
    case 10: return webp_vp8l_average2(webp_vp8l_average2(l, tl), webp_vp8l_average2(t, tr));
    case 11: return webp_vp8l_select(t, l, tl);
    case 12: return webp_vp8l_clamped_add_subtract_full(l, t, tl);
    case 13: return webp_vp8l_clamped_add_subtract_half(l, t, tl);
    default: return 0xff000000u;   /* 0, and the undefined 14 and 15 */
    }
}

static void webp_vp8l_inverse_predictor(const webp_vp8l_transform *t, uint32_t *data,
                                        int height) {
    const int width = t->xsize;
    const int tiles_per_row = webp_vp8l_subsample(width, t->bits);
    data[0] = webp_vp8l_add_pixels(data[0], 0xff000000u);
    for (int x = 1; x < width; x++) data[x] = webp_vp8l_add_pixels(data[x], data[x - 1]);
    for (int y = 1; y < height; y++) {
        uint32_t *row = data + (size_t)y * width;
        const uint32_t *up = row - width;
        const uint32_t *modes = t->data + (size_t)(y >> t->bits) * tiles_per_row;
        row[0] = webp_vp8l_add_pixels(row[0], up[0]);
        for (int x = 1; x < width; x++) {
            /* For the last column, up[x + 1] is the first pixel of this row. */
            const int mode = (int)((modes[x >> t->bits] >> 8) & 0xf);
            row[x] = webp_vp8l_add_pixels(
                row[x], webp_vp8l_predict(mode, row[x - 1], up[x], up[x - 1], up[x + 1]));
        }
    }
}

static void webp_vp8l_inverse_color(const webp_vp8l_transform *t, uint32_t *data, int height) {
    const int width = t->xsize;
    const int tiles_per_row = webp_vp8l_subsample(width, t->bits);
    for (int y = 0; y < height; y++) {
        uint32_t *row = data + (size_t)y * width;
        const uint32_t *m = t->data + (size_t)(y >> t->bits) * tiles_per_row;
        for (int x = 0; x < width; x++) {
            const uint32_t e = m[x >> t->bits];
            const int g2r = (int8_t)(e & 0xff);
            const int g2b = (int8_t)((e >> 8) & 0xff);
            const int r2b = (int8_t)((e >> 16) & 0xff);
            const uint32_t argb = row[x];
            const int green = (int8_t)((argb >> 8) & 0xff);
            int red = (int)((argb >> 16) & 0xff);
            int blue = (int)(argb & 0xff);
            red += (g2r * green) >> 5;
            red &= 0xff;
            blue += (g2b * green) >> 5;
            blue += (r2b * (int8_t)red) >> 5;
            blue &= 0xff;
            row[x] = (argb & 0xff00ff00u) | ((uint32_t)red << 16) | (uint32_t)blue;
        }
    }
}

static void webp_vp8l_add_green(uint32_t *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const uint32_t argb = data[i];
        const uint32_t green = (argb >> 8) & 0xff;
        uint32_t red_blue = argb & 0x00ff00ffu;
        red_blue += (green << 16) | green;
        data[i] = (argb & 0xff00ff00u) | (red_blue & 0x00ff00ffu);
    }
}

static uint32_t *webp_vp8l_inverse_color_indexing(const webp_vp8l_transform *t,
                                                  const uint32_t *src, int height) {
    const int width = t->xsize;
    const int bits_per_pixel = 8 >> t->bits;
    const int count_mask = (1 << t->bits) - 1;
    const uint32_t bit_mask = (1u << bits_per_pixel) - 1;
    uint32_t *dst = (uint32_t *)malloc((size_t)width * height * sizeof(*dst));
    if (!dst) return NULL;
    uint32_t *o = dst;
    for (int y = 0; y < height; y++) {
        uint32_t packed = 0;
        for (int x = 0; x < width; x++) {
            if ((x & count_mask) == 0) packed = (*src++ >> 8) & 0xff;
            *o++ = t->data[packed & bit_mask];
            packed >>= bits_per_pixel;
        }
    }
    return dst;
}

/* Decodes a VP8L image stream of known size (the main image after its
 * header, or the lossless payload of an ALPH chunk) to ARGB pixels. */
static uint32_t *webp_vp8l_decode_argb(const uint8_t *data, size_t size, int width, int height) {
    webp_vp8l_dec d;
    uint32_t *argb = NULL;

    memset(&d, 0, sizeof(d));
    d.br.buf = data;
    d.br.len = size;
    d.height = height;
    if (!webp_vp8l_decode_image_stream(&d, width, height, 1, &argb)) goto done;

    /* Inverse transforms, last one read first. */
    for (int n = d.num_transforms - 1; n >= 0; n--) {
        const webp_vp8l_transform *t = &d.transforms[n];
        switch (t->type) {
        case WEBP_PREDICTOR_TRANSFORM:
            webp_vp8l_inverse_predictor(t, argb, height);
            break;
        case WEBP_CROSS_COLOR_TRANSFORM:
            webp_vp8l_inverse_color(t, argb, height);
            break;
        case WEBP_SUBTRACT_GREEN_TRANSFORM:
            webp_vp8l_add_green(argb, (size_t)t->xsize * height);
            break;
        default: {
            uint32_t *expanded = webp_vp8l_inverse_color_indexing(t, argb, height);
            free(argb);
            argb = expanded;
            if (!argb) goto done;
            break;
        }
        }
    }

done:
    for (int n = 0; n < d.num_transforms; n++) free(d.transforms[n].data);
    return argb;
}

/* Reads the 5-byte VP8L header: signature, size, alpha hint, version. */
static int webp_vp8l_get_info(const uint8_t *data, size_t size,
                              int *width, int *height, int *has_alpha) {
    if (size < 5 || data[0] != 0x2f) return 0;
    const uint32_t bits = webp_le32(data + 1);
    if ((bits >> 29) != 0) return 0;    /* version must be 0 */
    *width = (int)(bits & 0x3fff) + 1;
    *height = (int)((bits >> 14) & 0x3fff) + 1;
    *has_alpha = (int)((bits >> 28) & 1);
    return webp_dimensions_ok((uint32_t)*width, (uint32_t)*height);
}

/* ========================================================================
 * Alpha Chunk ('ALPH')
 * ======================================================================== */

/* Undoes the alpha prediction filter of one row in place. Without a row
 * above, every filter falls back to the horizontal one. */
static void webp_alpha_unfilter_row(int filter, const uint8_t *prev, uint8_t *row, int width) {
    if (filter == 0) return;
    if (filter == 1 || !prev) {                 /* horizontal */
        uint8_t pred = prev ? prev[0] : 0;
        for (int i = 0; i < width; i++) {
            row[i] = (uint8_t)(pred + row[i]);
            pred = row[i];
        }
    } else if (filter == 2) {                   /* vertical */
        for (int i = 0; i < width; i++) row[i] = (uint8_t)(prev[i] + row[i]);
    } else {                                    /* gradient */
        uint8_t top_left = prev[0], left = prev[0];
        for (int i = 0; i < width; i++) {
            const uint8_t top = prev[i];
            left = (uint8_t)(row[i] + webp_clip(left + top - top_left, 255));
            top_left = top;
            row[i] = left;
        }
    }
}

/* Decodes an ALPH chunk payload into a width x height alpha plane. */
static int webp_alpha_decode(const uint8_t *data, size_t size, int width, int height,
                             uint8_t *alpha) {
    if (size <= 1) return 0;
    const int method = data[0] & 0x03;
    const int filter = (data[0] >> 2) & 0x03;
    const int pre_processing = (data[0] >> 4) & 0x03;
    const int reserved = (data[0] >> 6) & 0x03;
    const size_t total = (size_t)width * height;
    if (method > 1 || pre_processing > 1 || reserved != 0) return 0;
    if (method == 0) {
        if (size - 1 < total) return 0;
        memcpy(alpha, data + 1, total);
    } else {
        /* A VP8L image stream without header; alpha is in the green channel. */
        uint32_t *argb = webp_vp8l_decode_argb(data + 1, size - 1, width, height);
        if (!argb) return 0;
        for (size_t i = 0; i < total; i++) alpha[i] = (uint8_t)(argb[i] >> 8);
        free(argb);
    }
    /* Level reduction (pre_processing == 1) only matters for dithering,
     * which the default decoding does not apply. */
    for (int y = 0; y < height; y++) {
        webp_alpha_unfilter_row(filter, y > 0 ? alpha + (size_t)(y - 1) * width : NULL,
                                alpha + (size_t)y * width, width);
    }
    return 1;
}

/* ========================================================================
 * Image Management
 * ======================================================================== */

void webp_free(webp_image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

/* ========================================================================
 * RIFF Container and Loading
 * ======================================================================== */

typedef struct {
    const uint8_t *image;       /* 'VP8 ' or 'VP8L' payload */
    size_t image_size;          /* up to the end of the input, like libwebp */
    size_t image_chunk_size;    /* as declared by the chunk header */
    int lossless;
    const uint8_t *alpha;       /* 'ALPH' payload, if any */
    size_t alpha_size;
    int has_vp8x;
    uint32_t vp8x_flags;
    int canvas_width, canvas_height;
    int animated;
} webp_container;

#define WEBP_VP8X_ALPHA_FLAG 0x10
#define WEBP_VP8X_ANIMATION_FLAG 0x02

/* Walks the chunks up to the image bitstream. Returns 0 for malformed or
 * animated files (c->animated tells the two apart). */
static int webp_parse_container(const uint8_t *data, size_t len, webp_container *c) {
    memset(c, 0, sizeof(*c));
    if (!data || len < 12 || len > WEBP_MAX_INPUT_BYTES) return 0;
    if (memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WEBP", 4) != 0) return 0;
    const uint32_t riff_size = webp_le32(data + 4);
    if (riff_size < 12 || (uint64_t)riff_size > (uint64_t)len - 8) return 0;
    /* Chunks are walked up to the end of the RIFF payload only; the image
     * bitstream itself may extend to the end of the input (see below). */
    const size_t end = (size_t)riff_size + 8;
    size_t pos = 12;

    for (;;) {
        if (end - pos < 8) return 0;
        const uint8_t *fourcc = data + pos;
        const uint32_t chunk_size = webp_le32(data + pos + 4);
        if (chunk_size > end - pos - 8) return 0;
        const uint8_t *payload = data + pos + 8;
        if (memcmp(fourcc, "VP8 ", 4) == 0 || memcmp(fourcc, "VP8L", 4) == 0) {
            /* The chunk size is validated but does not bound the decoder:
             * like libwebp, the bitstream may run to the end of the input,
             * so files with a stale size field decode as they do elsewhere. */
            c->image = payload;
            c->image_size = len - (pos + 8);
            c->image_chunk_size = chunk_size;
            c->lossless = fourcc[3] == 'L';
            return 1;
        }
        if (memcmp(fourcc, "VP8X", 4) == 0) {
            /* A second VP8X is refused (libwebp skips it): RFC 9649 says
             * readers SHOULD fail when such chunks are out of order. */
            if (pos != 12 || chunk_size != 10) return 0;
            c->has_vp8x = 1;
            c->vp8x_flags = payload[0];
            c->canvas_width = (int)webp_le24(payload + 4) + 1;
            c->canvas_height = (int)webp_le24(payload + 7) + 1;
            if (c->vp8x_flags & WEBP_VP8X_ANIMATION_FLAG) {
                c->animated = 1;
                return 0;
            }
        } else if (pos == 12) {
            return 0;   /* the first chunk must be VP8X, VP8 or VP8L */
        } else if (memcmp(fourcc, "ALPH", 4) == 0) {
            c->alpha = payload;             /* the last one wins, as in libwebp */
            c->alpha_size = chunk_size;
        }
        /* ICCP, EXIF, XMP and unknown chunks are skipped. So are ANIM and
         * ANMF when the animation flag is off: RFC 9649 says they MUST be
         * ignored then, and libwebp decodes such files as still images. */
        pos += 8 + (size_t)chunk_size + (chunk_size & 1);
        if (pos > end) return 0;
    }
}

int webp_is_animated(const uint8_t *data, size_t len) {
    webp_container c;
    webp_parse_container(data, len, &c);
    return c.animated;
}

webp_image *webp_load_mem(const uint8_t *data, size_t len) {
    webp_container c;
    if (!webp_parse_container(data, len, &c)) return NULL;

    int width = 0, height = 0, has_alpha = 0;
    uint8_t *pixels = NULL;
    if (c.lossless) {
        /* Like libwebp: VP8L carries its own alpha. The ALPH payload is not
         * decoded, but the chunk's presence still marks the image as RGBA. */
        if (!webp_vp8l_get_info(c.image, c.image_size, &width, &height, &has_alpha))
            return NULL;
        has_alpha |= c.alpha != NULL;
        if (c.has_vp8x && (width != c.canvas_width || height != c.canvas_height)) return NULL;
        uint32_t *argb = webp_vp8l_decode_argb(c.image + 5, c.image_size - 5, width, height);
        if (!argb) return NULL;
        const int channels = has_alpha ? 4 : 3;
        const size_t total = (size_t)width * height;
        pixels = (uint8_t *)malloc(total * channels);
        if (pixels) {
            for (size_t i = 0; i < total; i++) {
                uint8_t *p = pixels + i * channels;
                p[0] = (uint8_t)(argb[i] >> 16);
                p[1] = (uint8_t)(argb[i] >> 8);
                p[2] = (uint8_t)argb[i];
                if (channels == 4) p[3] = (uint8_t)(argb[i] >> 24);
            }
        }
        free(argb);
        if (!pixels) return NULL;
    } else {
        has_alpha = (c.vp8x_flags & WEBP_VP8X_ALPHA_FLAG) || c.alpha != NULL;
        if (!webp_vp8_get_size(c.image, c.image_size, c.image_chunk_size, &width, &height))
            return NULL;
        if (c.has_vp8x && (width != c.canvas_width || height != c.canvas_height)) return NULL;
        pixels = webp_vp8_decode(c.image, c.image_size, c.image_chunk_size,
                                 has_alpha ? 4 : 3, &width, &height);
        if (!pixels) return NULL;
        if (c.alpha) {
            const size_t total = (size_t)width * height;
            uint8_t *alpha = (uint8_t *)malloc(total);
            if (!alpha || !webp_alpha_decode(c.alpha, c.alpha_size, width, height, alpha)) {
                free(alpha);
                free(pixels);
                return NULL;
            }
            for (size_t i = 0; i < total; i++) pixels[4 * i + 3] = alpha[i];
            free(alpha);
        }
    }

    webp_image *img = (webp_image *)malloc(sizeof(webp_image));
    if (!img) {
        free(pixels);
        return NULL;
    }
    img->width = width;
    img->height = height;
    img->channels = has_alpha ? 4 : 3;
    img->data = pixels;
    return img;
}

webp_image *webp_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long end = ftell(f);
    if (end < 0 || (unsigned long)end > WEBP_MAX_INPUT_BYTES ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    size_t file_size = (size_t)end;

    uint8_t *file_data = (uint8_t *)malloc(file_size);
    if (!file_data) {
        fclose(f);
        return NULL;
    }

    if (fread(file_data, 1, file_size, f) != file_size) {
        free(file_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    webp_image *img = webp_load_mem(file_data, file_size);
    free(file_data);
    return img;
}

/* Clean up internal macros */
#undef WEBP_BPS
#undef WEBP_NUM_LITERAL_CODES
#undef WEBP_NUM_LENGTH_CODES
#undef WEBP_NUM_DISTANCE_CODES
#undef WEBP_MAX_CACHE_BITS
#undef WEBP_MAX_ALPHABET
#undef WEBP_MAX_CODE_LENGTH
#undef WEBP_FAST_BITS
#undef WEBP_VP8X_ALPHA_FLAG
#undef WEBP_VP8X_ANIMATION_FLAG

#endif /* WEBP_IMPLEMENTATION */
#endif /* DS4_WEBP_H */
