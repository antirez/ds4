/*
 * WebP Decoder - Single-header implementation
 *
 * A dependency-free C decoder for still WebP images, covering both
 * compression modes defined by the format:
 *
 *   - Lossy   (VP8  chunk): a VP8 key frame. Intra prediction only, so no
 *                           motion compensation or reference frames.
 *   - Lossless(VP8L chunk): prefix coding, backward references, color cache
 *                           and the four inverse transforms.
 *
 * Animated WebP (ANIM/ANMF) is rejected. The alpha channel of a lossy image
 * (ALPH chunk) is ignored, so such images decode to opaque RGB.
 *
 * Usage:
 *   webp_image *img = webp_load_mem(data, len);
 *   if (!img) { handle error }
 *
 *   // Access pixel data
 *   uint8_t *pixel = img->data + (y * img->width + x) * img->channels;
 *
 *   webp_free(img);
 *
 * To use as header-only, define WEBP_IMPLEMENTATION before including:
 *   #define WEBP_IMPLEMENTATION
 *   #include "webp.h"
 *
 * The bitstream format, the probability and quantizer tables, and the
 * reconstruction arithmetic all come from the normative descriptions in
 * RFC 6386 (VP8) and the WebP Lossless Bitstream Specification. Decoded
 * output is bit-exact with libwebp's `dwebp -nofancy`.
 */

#ifndef WEBP_H
#define WEBP_H

#include <stddef.h>
#include <stdint.h>

/* These limits make the memory decoder suitable for untrusted server input. */
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
 * Load WebP image from memory buffer.
 * Returns NULL on error.
 */
webp_image *webp_load_mem(const uint8_t *data, size_t len);

/*
 * Load WebP image from memory buffer, reporting why decoding failed.
 * On failure a human-readable reason is written to 'err' (if non-NULL) and
 * NULL is returned. Thread-safe: no global error state is used.
 */
webp_image *webp_load_mem_err(const uint8_t *data, size_t len,
                              char *err, size_t err_cap);

/*
 * Load WebP image from file.
 * Returns NULL on error.
 */
webp_image *webp_load(const char *path);

/*
 * Returns 1 if the buffer starts with a RIFF/WEBP signature.
 */
int webp_is_webp(const uint8_t *data, size_t len);

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

#define WEBP_ARG_UNUSED(x) (void)(x)

static void webp_err_set(char *err, size_t cap, const char *msg) {
    if (!err || !cap) return;
    snprintf(err, cap, "%s", msg);
}

static int webp_clip_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static int webp_clip_s8(int v) { return v < -128 ? -128 : (v > 127 ? 127 : v); }
static int webp_clip_s4(int v) { return v < -16 ? -16 : (v > 15 ? 15 : v); }
static int webp_abs(int v) { return v < 0 ? -v : v; }

static uint32_t webp_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint32_t webp_le24(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
}
static uint32_t webp_le16(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

/* ------------------------------------------------------------------------
 * VP8 boolean entropy decoder (RFC 6386 section 7)
 * ------------------------------------------------------------------------ */

typedef struct {
    const uint8_t *buf;
    size_t len;
    size_t pos;
    uint32_t value;     /* 16 significant bits */
    int range;          /* 128..255 */
    int bit_count;      /* bits consumed from the low byte of 'value' */
    int eof;            /* set once we read past the end of the partition */
} vp8_bool;

static int vp8_bool_byte(vp8_bool *d) {
    if (d->pos < d->len) return d->buf[d->pos++];
    d->eof = 1;
    return 0;
}

static void vp8_bool_init(vp8_bool *d, const uint8_t *buf, size_t len) {
    d->buf = buf;
    d->len = len;
    d->pos = 0;
    d->range = 255;
    d->bit_count = 0;
    d->eof = 0;
    d->value = (uint32_t)vp8_bool_byte(d) << 8;
    d->value |= (uint32_t)vp8_bool_byte(d);
}

static int vp8_get(vp8_bool *d, int prob) {
    const uint32_t split = 1u + (((uint32_t)(d->range - 1) * (uint32_t)prob) >> 8);
    const uint32_t big = split << 8;
    int bit;
    if (d->value >= big) {
        bit = 1;
        d->range -= (int)split;
        d->value -= big;
    } else {
        bit = 0;
        d->range = (int)split;
    }
    while (d->range < 128) {
        d->value <<= 1;
        d->range <<= 1;
        if (++d->bit_count == 8) {
            d->bit_count = 0;
            d->value |= (uint32_t)vp8_bool_byte(d);
        }
    }
    return bit;
}

static int vp8_get_bit(vp8_bool *d) { return vp8_get(d, 128); }

static int vp8_get_uint(vp8_bool *d, int bits) {
    int v = 0;
    while (bits-- > 0) v = (v << 1) | vp8_get(d, 128);
    return v;
}

static int vp8_get_sint(vp8_bool *d, int bits) {
    const int v = vp8_get_uint(d, bits);
    return vp8_get_bit(d) ? -v : v;
}

/* ------------------------------------------------------------------------
 * VP8 constant tables (RFC 6386)
 * ------------------------------------------------------------------------ */

enum {
    WEBP_B_DC_PRED = 0,
    WEBP_B_TM_PRED = 1,
    WEBP_B_VE_PRED = 2,
    WEBP_B_HE_PRED = 3,
    WEBP_B_RD_PRED = 4,
    WEBP_B_VR_PRED = 5,
    WEBP_B_LD_PRED = 6,
    WEBP_B_VL_PRED = 7,
    WEBP_B_HD_PRED = 8,
    WEBP_B_HU_PRED = 9,
    WEBP_NUM_BMODES = 10
};

/* The first four 4x4 modes double as the 16x16 and chroma modes. */
#define WEBP_DC_PRED WEBP_B_DC_PRED
#define WEBP_TM_PRED WEBP_B_TM_PRED
#define WEBP_V_PRED  WEBP_B_VE_PRED
#define WEBP_H_PRED  WEBP_B_HE_PRED

#define WEBP_NUM_SEGMENTS 4
#define WEBP_NUM_TYPES    4
#define WEBP_NUM_BANDS    8
#define WEBP_NUM_CTX      3
#define WEBP_NUM_PROBAS   11

/* Paragraph 13.5: default coefficient probabilities. */
static const uint8_t kCoeffsProba0[4][8][3][11] = {
    {
        {
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        },
        {
            {253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128},
            {189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128},
            {106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128},
        },
        {
            {  1,  98, 248, 255, 236, 226, 255, 255, 128, 128, 128},
            {181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128},
            { 78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128},
        },
        {
            {  1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128},
            {184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128},
            { 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128},
        },
        {
            {  1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128},
            {170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128},
            { 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128},
        },
        {
            {  1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128},
            {207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128},
            {102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128},
        },
        {
            {  1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128},
            {177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128},
            { 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128},
        },
        {
            {  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {246,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        },
    },
    {
        {
            {198,  35, 237, 223, 193, 187, 162, 160, 145, 155,  62},
            {131,  45, 198, 221, 172, 176, 220, 157, 252, 221,   1},
            { 68,  47, 146, 208, 149, 167, 221, 162, 255, 223, 128},
        },
        {
            {  1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128},
            {184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128},
            { 81,  99, 181, 242, 176, 190, 249, 202, 255, 255, 128},
        },
        {
            {  1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128},
            { 99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128},
            { 23,  91, 163, 242, 170, 187, 247, 210, 255, 255, 128},
        },
        {
            {  1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128},
            {109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128},
            { 44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128},
        },
        {
            {  1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128},
            { 94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128},
            { 22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128},
        },
        {
            {  1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128},
            {124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128},
            { 35,  77, 181, 251, 193, 211, 255, 205, 128, 128, 128},
        },
        {
            {  1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128},
            {121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128},
            { 45,  99, 188, 251, 195, 217, 255, 224, 128, 128, 128},
        },
        {
            {  1,   1, 251, 255, 213, 255, 128, 128, 128, 128, 128},
            {203,   1, 248, 255, 255, 128, 128, 128, 128, 128, 128},
            {137,   1, 177, 255, 224, 255, 128, 128, 128, 128, 128},
        },
    },
    {
        {
            {253,   9, 248, 251, 207, 208, 255, 192, 128, 128, 128},
            {175,  13, 224, 243, 193, 185, 249, 198, 255, 255, 128},
            { 73,  17, 171, 221, 161, 179, 236, 167, 255, 234, 128},
        },
        {
            {  1,  95, 247, 253, 212, 183, 255, 255, 128, 128, 128},
            {239,  90, 244, 250, 211, 209, 255, 255, 128, 128, 128},
            {155,  77, 195, 248, 188, 195, 255, 255, 128, 128, 128},
        },
        {
            {  1,  24, 239, 251, 218, 219, 255, 205, 128, 128, 128},
            {201,  51, 219, 255, 196, 186, 128, 128, 128, 128, 128},
            { 69,  46, 190, 239, 201, 218, 255, 228, 128, 128, 128},
        },
        {
            {  1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128},
            {223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128},
            {141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128},
        },
        {
            {  1,  16, 248, 255, 255, 128, 128, 128, 128, 128, 128},
            {190,  36, 230, 255, 236, 255, 128, 128, 128, 128, 128},
            {149,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
        },
        {
            {  1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128},
        },
        {
            {  1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128},
            {213,  62, 250, 255, 255, 128, 128, 128, 128, 128, 128},
            { 55,  93, 255, 128, 128, 128, 128, 128, 128, 128, 128},
        },
        {
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
            {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128},
        },
    },
    {
        {
            {202,  24, 213, 235, 186, 191, 220, 160, 240, 175, 255},
            {126,  38, 182, 232, 169, 184, 228, 174, 255, 187, 128},
            { 61,  46, 138, 219, 151, 178, 240, 170, 255, 216, 128},
        },
        {
            {  1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128},
            {166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128},
            { 39,  77, 162, 232, 172, 180, 245, 178, 255, 255, 128},
        },
        {
            {  1,  52, 220, 246, 198, 199, 249, 220, 255, 255, 128},
            {124,  74, 191, 243, 183, 193, 250, 221, 255, 255, 128},
            { 24,  71, 130, 219, 154, 170, 243, 182, 255, 255, 128},
        },
        {
            {  1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128},
            {149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128},
            { 28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128},
        },
        {
            {  1,  81, 230, 252, 204, 203, 255, 192, 128, 128, 128},
            {123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128},
            { 20,  95, 153, 243, 164, 173, 255, 203, 128, 128, 128},
        },
        {
            {  1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128},
            {168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128},
            { 47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128},
        },
        {
            {  1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128},
            {141,  84, 213, 252, 201, 202, 255, 219, 128, 128, 128},
            { 42,  80, 160, 240, 162, 185, 255, 205, 128, 128, 128},
        },
        {
            {  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {244,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
            {238,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128},
        },
    },
};

/* Paragraph 13.4: probability that each entry above is updated. */
static const uint8_t kCoeffsUpdateProba[4][8][3][11] = {
    {
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255},
            {250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
    },
    {
        {
            {217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255},
            {234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255},
        },
        {
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
    },
    {
        {
            {186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255},
            {234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255},
            {251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255},
        },
        {
            {255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
    },
    {
        {
            {248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255},
            {248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255},
            {248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255},
            {250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
        {
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
            {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
        },
    },
};

/* Paragraph 11.4: 4x4 mode probabilities, indexed by the modes above
 * and to the left of the subblock. */
static const uint8_t kBModesProba[10][10][9] = {
    {
        {231, 120,  48,  89, 115, 113, 120, 152, 112},
        {152, 179,  64, 126, 170, 118,  46,  70,  95},
        {175,  69, 143,  80,  85,  82,  72, 155, 103},
        { 56,  58,  10, 171, 218, 189,  17,  13, 152},
        {114,  26,  17, 163,  44, 195,  21,  10, 173},
        {121,  24,  80, 195,  26,  62,  44,  64,  85},
        {144,  71,  10,  38, 171, 213, 144,  34,  26},
        {170,  46,  55,  19, 136, 160,  33, 206,  71},
        { 63,  20,   8, 114, 114, 208,  12,   9, 226},
        { 81,  40,  11,  96, 182,  84,  29,  16,  36},
    },
    {
        {134, 183,  89, 137,  98, 101, 106, 165, 148},
        { 72, 187, 100, 130, 157, 111,  32,  75,  80},
        { 66, 102, 167,  99,  74,  62,  40, 234, 128},
        { 41,  53,   9, 178, 241, 141,  26,   8, 107},
        { 74,  43,  26, 146,  73, 166,  49,  23, 157},
        { 65,  38, 105, 160,  51,  52,  31, 115, 128},
        {104,  79,  12,  27, 217, 255,  87,  17,   7},
        { 87,  68,  71,  44, 114,  51,  15, 186,  23},
        { 47,  41,  14, 110, 182, 183,  21,  17, 194},
        { 66,  45,  25, 102, 197, 189,  23,  18,  22},
    },
    {
        { 88,  88, 147, 150,  42,  46,  45, 196, 205},
        { 43,  97, 183, 117,  85,  38,  35, 179,  61},
        { 39,  53, 200,  87,  26,  21,  43, 232, 171},
        { 56,  34,  51, 104, 114, 102,  29,  93,  77},
        { 39,  28,  85, 171,  58, 165,  90,  98,  64},
        { 34,  22, 116, 206,  23,  34,  43, 166,  73},
        {107,  54,  32,  26,  51,   1,  81,  43,  31},
        { 68,  25, 106,  22,  64, 171,  36, 225, 114},
        { 34,  19,  21, 102, 132, 188,  16,  76, 124},
        { 62,  18,  78,  95,  85,  57,  50,  48,  51},
    },
    {
        {193, 101,  35, 159, 215, 111,  89,  46, 111},
        { 60, 148,  31, 172, 219, 228,  21,  18, 111},
        {112, 113,  77,  85, 179, 255,  38, 120, 114},
        { 40,  42,   1, 196, 245, 209,  10,  25, 109},
        { 88,  43,  29, 140, 166, 213,  37,  43, 154},
        { 61,  63,  30, 155,  67,  45,  68,   1, 209},
        {100,  80,   8,  43, 154,   1,  51,  26,  71},
        {142,  78,  78,  16, 255, 128,  34, 197, 171},
        { 41,  40,   5, 102, 211, 183,   4,   1, 221},
        { 51,  50,  17, 168, 209, 192,  23,  25,  82},
    },
    {
        {138,  31,  36, 171,  27, 166,  38,  44, 229},
        { 67,  87,  58, 169,  82, 115,  26,  59, 179},
        { 63,  59,  90, 180,  59, 166,  93,  73, 154},
        { 40,  40,  21, 116, 143, 209,  34,  39, 175},
        { 47,  15,  16, 183,  34, 223,  49,  45, 183},
        { 46,  17,  33, 183,   6,  98,  15,  32, 183},
        { 57,  46,  22,  24, 128,   1,  54,  17,  37},
        { 65,  32,  73, 115,  28, 128,  23, 128, 205},
        { 40,   3,   9, 115,  51, 192,  18,   6, 223},
        { 87,  37,   9, 115,  59,  77,  64,  21,  47},
    },
    {
        {104,  55,  44, 218,   9,  54,  53, 130, 226},
        { 64,  90,  70, 205,  40,  41,  23,  26,  57},
        { 54,  57, 112, 184,   5,  41,  38, 166, 213},
        { 30,  34,  26, 133, 152, 116,  10,  32, 134},
        { 39,  19,  53, 221,  26, 114,  32,  73, 255},
        { 31,   9,  65, 234,   2,  15,   1, 118,  73},
        { 75,  32,  12,  51, 192, 255, 160,  43,  51},
        { 88,  31,  35,  67, 102,  85,  55, 186,  85},
        { 56,  21,  23, 111,  59, 205,  45,  37, 192},
        { 55,  38,  70, 124,  73, 102,   1,  34,  98},
    },
    {
        {125,  98,  42,  88, 104,  85, 117, 175,  82},
        { 95,  84,  53,  89, 128, 100, 113, 101,  45},
        { 75,  79, 123,  47,  51, 128,  81, 171,   1},
        { 57,  17,   5,  71, 102,  57,  53,  41,  49},
        { 38,  33,  13, 121,  57,  73,  26,   1,  85},
        { 41,  10,  67, 138,  77, 110,  90,  47, 114},
        {115,  21,   2,  10, 102, 255, 166,  23,   6},
        {101,  29,  16,  10,  85, 128, 101, 196,  26},
        { 57,  18,  10, 102, 102, 213,  34,  20,  43},
        {117,  20,  15,  36, 163, 128,  68,   1,  26},
    },
    {
        {102,  61,  71,  37,  34,  53,  31, 243, 192},
        { 69,  60,  71,  38,  73, 119,  28, 222,  37},
        { 68,  45, 128,  34,   1,  47,  11, 245, 171},
        { 62,  17,  19,  70, 146,  85,  55,  62,  70},
        { 37,  43,  37, 154, 100, 163,  85, 160,   1},
        { 63,   9,  92, 136,  28,  64,  32, 201,  85},
        { 75,  15,   9,   9,  64, 255, 184, 119,  16},
        { 86,   6,  28,   5,  64, 255,  25, 248,   1},
        { 56,   8,  17, 132, 137, 255,  55, 116, 128},
        { 58,  15,  20,  82, 135,  57,  26, 121,  40},
    },
    {
        {164,  50,  31, 137, 154, 133,  25,  35, 218},
        { 51, 103,  44, 131, 131, 123,  31,   6, 158},
        { 86,  40,  64, 135, 148, 224,  45, 183, 128},
        { 22,  26,  17, 131, 240, 154,  14,   1, 209},
        { 45,  16,  21,  91,  64, 222,   7,   1, 197},
        { 56,  21,  39, 155,  60, 138,  23, 102, 213},
        { 83,  12,  13,  54, 192, 255,  68,  47,  28},
        { 85,  26,  85,  85, 128, 128,  32, 146, 171},
        { 18,  11,   7,  63, 144, 171,   4,   4, 246},
        { 35,  27,  10, 146, 174, 171,  12,  26, 128},
    },
    {
        {190,  80,  35,  99, 180,  80, 126,  54,  45},
        { 85, 126,  47,  87, 176,  51,  41,  20,  32},
        {101,  75, 128, 139, 118, 146, 116, 128,  85},
        { 56,  41,  15, 176, 236,  85,  37,   9,  62},
        { 71,  30,  17, 119, 118, 255,  17,  18, 138},
        {101,  38,  60, 138,  55,  70,  43,  26, 142},
        {146,  36,  19,  30, 171, 255,  97,  27,  20},
        {138,  45,  61,  62, 219,   1,  81, 188,  64},
        { 32,  41,  20, 117, 151, 142,  20,  21, 163},
        {112,  19,  12,  61, 195, 128,  48,   4,  24},
    },
};

/* Paragraph 9.9: coefficient index -> probability band. */
static const uint8_t kBands[16 + 1] = {
    0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7,
    0  /* extra entry as sentinel */
};

static const uint8_t kZigzag[16] = {
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
};

/* Paragraph 13.2: extra-bit probabilities for token categories 3..6. */
static const uint8_t kCat3[] = {173, 148, 140, 0};
static const uint8_t kCat4[] = {176, 155, 140, 135, 0};
static const uint8_t kCat5[] = {180, 157, 141, 134, 130, 0};
static const uint8_t kCat6[] = {254, 254, 243, 230, 196, 177,
                                153, 140, 133, 130, 129, 0};
static const uint8_t *const kCat3456[] = {kCat3, kCat4, kCat5, kCat6};

/* Paragraph 14.1: dequantization lookups. */
static const uint8_t kDcTable[128] = {
      4,   5,   6,   7,   8,   9,  10,  10,  11,  12,  13,  14,  15,  16,  17,
     17,  18,  19,  20,  20,  21,  21,  22,  22,  23,  23,  24,  25,  25,  26,
     27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  37,  38,  39,  40,
     41,  42,  43,  44,  45,  46,  46,  47,  48,  49,  50,  51,  52,  53,  54,
     55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
     70,  71,  72,  73,  74,  75,  76,  76,  77,  78,  79,  80,  81,  82,  83,
     84,  85,  86,  87,  88,  89,  91,  93,  95,  96,  98, 100, 101, 102, 104,
    106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, 132, 134, 136,
    138, 140, 143, 145, 148, 151, 154, 157
};

static const uint16_t kAcTable[128] = {
      4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,
     19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
     49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  62,  64,  66,  68,
     70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
    100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134,
    137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181,
    185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 234, 239, 245,
    249, 254, 259, 264, 269, 274, 279, 284
};

/* ------------------------------------------------------------------------
 * VP8 decoder state
 * ------------------------------------------------------------------------ */

typedef struct {
    uint16_t y1[2];     /* [0]=DC quantizer, [1]=AC quantizer */
    uint16_t y2[2];
    uint16_t uv[2];
} vp8_quant;

typedef struct {
    uint8_t limit;      /* 0 means "do not filter this macroblock" */
    uint8_t ilevel;
    uint8_t hev;
    uint8_t inner;
} vp8_finfo;

typedef struct {
    /* Frame geometry. */
    int width, height;
    int mb_w, mb_h;

    /* Bitstream partitions. */
    vp8_bool br;                /* first partition: headers and modes */
    vp8_bool parts[8];
    int num_parts;

    /* Segmentation (paragraph 9.3). */
    int use_segment;
    int update_map;
    int absolute_delta;
    int8_t seg_quant[WEBP_NUM_SEGMENTS];
    int8_t seg_filter[WEBP_NUM_SEGMENTS];
    uint8_t seg_prob[3];

    /* Loop filter (paragraph 9.4). */
    int filter_type;            /* 0=off, 1=simple, 2=normal */
    int filter_level;
    int filter_sharpness;
    int use_lf_delta;
    int8_t ref_lf_delta[4];
    int8_t mode_lf_delta[4];

    /* Entropy state. */
    uint8_t coeff_probs[WEBP_NUM_TYPES][WEBP_NUM_BANDS][WEBP_NUM_CTX][WEBP_NUM_PROBAS];
    int use_skip_proba;
    uint8_t skip_prob;

    vp8_quant dqm[WEBP_NUM_SEGMENTS];
    vp8_finfo fstrength[WEBP_NUM_SEGMENTS][2];

    /* Reconstruction planes, with a one pixel top/left border and a right
     * border wide enough for the four "above-right" prediction samples. */
    uint8_t *y_plane, *u_plane, *v_plane;
    int y_stride, uv_stride;

    /* Per-macroblock filter parameters, applied after the whole frame is
     * reconstructed (intra prediction must see unfiltered neighbours). */
    vp8_finfo *fstore;

    /* Intra mode prediction contexts. */
    uint8_t *top_modes;         /* 4 per macroblock column */
    uint8_t left_modes[4];

    /* Non-zero coefficient contexts. */
    uint8_t *top_nz;            /* 9 per macroblock column: 4 Y, 2 U, 2 V, 1 Y2 */
    uint8_t left_nz[9];

    /* Current macroblock. */
    int segment;
    int skip;
    int is_i4x4;
    int uvmode;
    uint8_t imodes[16];
    int16_t coeffs[400];        /* 25 blocks: 16 Y, 4 U, 4 V, 1 Y2 */
} vp8_dec;



/* Pixel accessors. Coordinates are relative to the visible frame origin, so
 * (-1, -1) addresses the top-left border sample. */
static uint8_t *vp8_y_at(vp8_dec *d, int x, int y) {
    return d->y_plane + (size_t)(y + 1) * (size_t)d->y_stride + (x + 1);
}
static uint8_t *vp8_u_at(vp8_dec *d, int x, int y) {
    return d->u_plane + (size_t)(y + 1) * (size_t)d->uv_stride + (x + 1);
}
static uint8_t *vp8_v_at(vp8_dec *d, int x, int y) {
    return d->v_plane + (size_t)(y + 1) * (size_t)d->uv_stride + (x + 1);
}

/* ------------------------------------------------------------------------
 * Frame header parsing (RFC 6386 section 9)
 * ------------------------------------------------------------------------ */

static int vp8_parse_segment_header(vp8_dec *d) {
    vp8_bool *br = &d->br;
    d->use_segment = vp8_get_bit(br);
    if (!d->use_segment) {
        d->update_map = 0;
        return 1;
    }
    d->update_map = vp8_get_bit(br);
    if (vp8_get_bit(br)) {              /* update segment feature data */
        int s;
        d->absolute_delta = vp8_get_bit(br);
        for (s = 0; s < WEBP_NUM_SEGMENTS; s++)
            d->seg_quant[s] = vp8_get_bit(br) ? (int8_t)vp8_get_sint(br, 7) : 0;
        for (s = 0; s < WEBP_NUM_SEGMENTS; s++)
            d->seg_filter[s] = vp8_get_bit(br) ? (int8_t)vp8_get_sint(br, 6) : 0;
    }
    if (d->update_map) {
        int s;
        for (s = 0; s < 3; s++)
            d->seg_prob[s] = vp8_get_bit(br) ? (uint8_t)vp8_get_uint(br, 8) : 255;
    }
    return !br->eof;
}

static int vp8_parse_filter_header(vp8_dec *d) {
    vp8_bool *br = &d->br;
    const int simple = vp8_get_bit(br);
    d->filter_level = vp8_get_uint(br, 6);
    d->filter_sharpness = vp8_get_uint(br, 3);
    d->use_lf_delta = vp8_get_bit(br);
    if (d->use_lf_delta && vp8_get_bit(br)) {
        int i;
        for (i = 0; i < 4; i++)
            if (vp8_get_bit(br)) d->ref_lf_delta[i] = (int8_t)vp8_get_sint(br, 6);
        for (i = 0; i < 4; i++)
            if (vp8_get_bit(br)) d->mode_lf_delta[i] = (int8_t)vp8_get_sint(br, 6);
    }
    d->filter_type = (d->filter_level == 0) ? 0 : (simple ? 1 : 2);
    return !br->eof;
}

static int vp8_clip_q(int v, int max) { return v < 0 ? 0 : (v > max ? max : v); }

/* Paragraph 9.6 */
static void vp8_parse_quant(vp8_dec *d) {
    vp8_bool *br = &d->br;
    const int base = vp8_get_uint(br, 7);
    const int dqy1_dc = vp8_get_bit(br) ? vp8_get_sint(br, 4) : 0;
    const int dqy2_dc = vp8_get_bit(br) ? vp8_get_sint(br, 4) : 0;
    const int dqy2_ac = vp8_get_bit(br) ? vp8_get_sint(br, 4) : 0;
    const int dquv_dc = vp8_get_bit(br) ? vp8_get_sint(br, 4) : 0;
    const int dquv_ac = vp8_get_bit(br) ? vp8_get_sint(br, 4) : 0;
    int i;

    for (i = 0; i < WEBP_NUM_SEGMENTS; i++) {
        vp8_quant *m = &d->dqm[i];
        int q;
        if (d->use_segment) {
            q = d->seg_quant[i];
            if (!d->absolute_delta) q += base;
        } else {
            if (i > 0) {
                d->dqm[i] = d->dqm[0];
                continue;
            }
            q = base;
        }
        m->y1[0] = kDcTable[vp8_clip_q(q + dqy1_dc, 127)];
        m->y1[1] = kAcTable[vp8_clip_q(q, 127)];
        m->y2[0] = (uint16_t)(kDcTable[vp8_clip_q(q + dqy2_dc, 127)] * 2);
        /* The spec scales the Y2 AC quantizer by 155/100. */
        m->y2[1] = (uint16_t)((kAcTable[vp8_clip_q(q + dqy2_ac, 127)] * 101581) >> 16);
        if (m->y2[1] < 8) m->y2[1] = 8;
        m->uv[0] = kDcTable[vp8_clip_q(q + dquv_dc, 117)];
        m->uv[1] = kAcTable[vp8_clip_q(q + dquv_ac, 127)];
    }
}

/* Paragraph 13.4 */
static void vp8_parse_proba(vp8_dec *d) {
    vp8_bool *br = &d->br;
    int t, b, c, p;
    for (t = 0; t < WEBP_NUM_TYPES; t++)
        for (b = 0; b < WEBP_NUM_BANDS; b++)
            for (c = 0; c < WEBP_NUM_CTX; c++)
                for (p = 0; p < WEBP_NUM_PROBAS; p++)
                    if (vp8_get(br, kCoeffsUpdateProba[t][b][c][p]))
                        d->coeff_probs[t][b][c][p] = (uint8_t)vp8_get_uint(br, 8);
    d->use_skip_proba = vp8_get_bit(br);
    if (d->use_skip_proba) d->skip_prob = (uint8_t)vp8_get_uint(br, 8);
}

/* Paragraph 15.2: per-segment loop filter strengths. */
static void vp8_precompute_filter_strengths(vp8_dec *d) {
    int s, i4x4;
    if (d->filter_type == 0) {
        memset(d->fstrength, 0, sizeof(d->fstrength));
        return;
    }
    for (s = 0; s < WEBP_NUM_SEGMENTS; s++) {
        int base_level;
        if (d->use_segment) {
            base_level = d->seg_filter[s];
            if (!d->absolute_delta) base_level += d->filter_level;
        } else {
            base_level = d->filter_level;
        }
        for (i4x4 = 0; i4x4 <= 1; i4x4++) {
            vp8_finfo *info = &d->fstrength[s][i4x4];
            int level = base_level;
            if (d->use_lf_delta) {
                level += d->ref_lf_delta[0];        /* intra frame */
                if (i4x4) level += d->mode_lf_delta[0];
            }
            level = level < 0 ? 0 : (level > 63 ? 63 : level);
            if (level > 0) {
                int ilevel = level;
                if (d->filter_sharpness > 0) {
                    ilevel >>= (d->filter_sharpness > 4) ? 2 : 1;
                    if (ilevel > 9 - d->filter_sharpness)
                        ilevel = 9 - d->filter_sharpness;
                }
                if (ilevel < 1) ilevel = 1;
                info->ilevel = (uint8_t)ilevel;
                info->limit = (uint8_t)(2 * level + ilevel);
                info->hev = (uint8_t)((level >= 40) ? 2 : (level >= 15) ? 1 : 0);
            } else {
                info->limit = 0;
                info->ilevel = 0;
                info->hev = 0;
            }
            info->inner = (uint8_t)i4x4;
        }
    }
}

/* ------------------------------------------------------------------------
 * Macroblock mode parsing (RFC 6386 sections 8 and 11)
 * ------------------------------------------------------------------------ */

static void vp8_parse_intra_modes(vp8_dec *d, int mb_x) {
    vp8_bool *br = &d->br;
    uint8_t *top = d->top_modes + 4 * mb_x;
    uint8_t *left = d->left_modes;

    d->segment = d->update_map
        ? (!vp8_get(br, d->seg_prob[0])
               ? vp8_get(br, d->seg_prob[1])
               : vp8_get(br, d->seg_prob[2]) + 2)
        : 0;
    d->skip = d->use_skip_proba ? vp8_get(br, d->skip_prob) : 0;
    d->is_i4x4 = !vp8_get(br, 145);

    if (!d->is_i4x4) {
        const int ymode = vp8_get(br, 156)
            ? (vp8_get(br, 128) ? WEBP_TM_PRED : WEBP_H_PRED)
            : (vp8_get(br, 163) ? WEBP_V_PRED : WEBP_DC_PRED);
        memset(d->imodes, ymode, sizeof(d->imodes));
        memset(top, ymode, 4);
        memset(left, ymode, 4);
    } else {
        int y;
        for (y = 0; y < 4; y++) {
            int mode = left[y];
            int x;
            for (x = 0; x < 4; x++) {
                const uint8_t *p = kBModesProba[top[x]][mode];
                mode = !vp8_get(br, p[0]) ? WEBP_B_DC_PRED
                     : !vp8_get(br, p[1]) ? WEBP_B_TM_PRED
                     : !vp8_get(br, p[2]) ? WEBP_B_VE_PRED
                     : !vp8_get(br, p[3])
                         ? (!vp8_get(br, p[4])
                                ? WEBP_B_HE_PRED
                                : (!vp8_get(br, p[5]) ? WEBP_B_RD_PRED
                                                      : WEBP_B_VR_PRED))
                         : (!vp8_get(br, p[6])
                                ? WEBP_B_LD_PRED
                                : (!vp8_get(br, p[7])
                                       ? WEBP_B_VL_PRED
                                       : (!vp8_get(br, p[8]) ? WEBP_B_HD_PRED
                                                             : WEBP_B_HU_PRED)));
                top[x] = (uint8_t)mode;
                d->imodes[4 * y + x] = (uint8_t)mode;
            }
            left[y] = (uint8_t)mode;
        }
    }
    d->uvmode = !vp8_get(br, 142) ? WEBP_DC_PRED
              : !vp8_get(br, 114) ? WEBP_V_PRED
              : vp8_get(br, 183)  ? WEBP_TM_PRED
                                  : WEBP_H_PRED;
}

/* ------------------------------------------------------------------------
 * Inverse transforms (RFC 6386 section 14.3 and 14.4)
 * ------------------------------------------------------------------------ */

#define WEBP_MUL1(a) ((((a) * 20091) >> 16) + (a))
#define WEBP_MUL2(a) (((a) * 35468) >> 16)

static void vp8_transform_wht(const int16_t *in, int16_t *out) {
    int tmp[16];
    int i;
    for (i = 0; i < 4; i++) {
        const int a0 = in[0 + i] + in[12 + i];
        const int a1 = in[4 + i] + in[8 + i];
        const int a2 = in[4 + i] - in[8 + i];
        const int a3 = in[0 + i] - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for (i = 0; i < 4; i++) {
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

static void vp8_transform_add(const int16_t *in, uint8_t *dst, int stride) {
    int C[4 * 4];
    int *tmp = C;
    int i;
    for (i = 0; i < 4; i++) {            /* vertical pass */
        const int a = in[0] + in[8];
        const int b = in[0] - in[8];
        const int c = WEBP_MUL2(in[4]) - WEBP_MUL1(in[12]);
        const int d = WEBP_MUL1(in[4]) + WEBP_MUL2(in[12]);
        tmp[0] = a + d;
        tmp[1] = b + c;
        tmp[2] = b - c;
        tmp[3] = a - d;
        tmp += 4;
        in++;
    }
    tmp = C;
    for (i = 0; i < 4; i++) {            /* horizontal pass */
        const int dc = tmp[0] + 4;
        const int a = dc + tmp[8];
        const int b = dc - tmp[8];
        const int c = WEBP_MUL2(tmp[4]) - WEBP_MUL1(tmp[12]);
        const int d = WEBP_MUL1(tmp[4]) + WEBP_MUL2(tmp[12]);
        dst[0] = (uint8_t)webp_clip_u8(dst[0] + ((a + d) >> 3));
        dst[1] = (uint8_t)webp_clip_u8(dst[1] + ((b + c) >> 3));
        dst[2] = (uint8_t)webp_clip_u8(dst[2] + ((b - c) >> 3));
        dst[3] = (uint8_t)webp_clip_u8(dst[3] + ((a - d) >> 3));
        tmp++;
        dst += stride;
    }
}

/* ------------------------------------------------------------------------
 * Coefficient decoding (RFC 6386 section 13)
 * ------------------------------------------------------------------------ */

static int vp8_get_large_value(vp8_bool *br, const uint8_t *p) {
    int v;
    if (!vp8_get(br, p[3])) {
        if (!vp8_get(br, p[4])) {
            v = 2;
        } else {
            v = 3 + vp8_get(br, p[5]);
        }
    } else {
        if (!vp8_get(br, p[6])) {
            if (!vp8_get(br, p[7])) {
                v = 5 + vp8_get(br, 159);
            } else {
                v = 7 + 2 * vp8_get(br, 165);
                v += vp8_get(br, 145);
            }
        } else {
            const uint8_t *tab;
            const int bit1 = vp8_get(br, p[8]);
            const int bit0 = vp8_get(br, p[9 + bit1]);
            const int cat = 2 * bit1 + bit0;
            v = 0;
            for (tab = kCat3456[cat]; *tab; tab++) v += v + vp8_get(br, *tab);
            v += 3 + (8 << cat);
        }
    }
    return v;
}

/* Returns the position of the last non-zero coefficient plus one. */
static int vp8_get_coeffs(vp8_bool *br,
                          const uint8_t probs[WEBP_NUM_BANDS][WEBP_NUM_CTX][WEBP_NUM_PROBAS],
                          int ctx, const uint16_t dq[2], int n, int16_t *out) {
    const uint8_t *p = probs[kBands[n]][ctx];
    for (; n < 16; n++) {
        if (!vp8_get(br, p[0])) return n;   /* previous coeff was the last one */
        while (!vp8_get(br, p[1])) {        /* a run of zero coefficients */
            p = probs[kBands[++n]][0];
            if (n == 16) return 16;
        }
        {
            int v;
            if (!vp8_get(br, p[2])) {
                v = 1;
                p = probs[kBands[n + 1]][1];
            } else {
                v = vp8_get_large_value(br, p);
                p = probs[kBands[n + 1]][2];
            }
            if (vp8_get_bit(br)) v = -v;
            out[kZigzag[n]] = (int16_t)(v * dq[n > 0]);
        }
    }
    return 16;
}

/*
 * Paragraph 13.3: residuals of one macroblock, tracking the non-zero
 * context of the neighbours above and to the left.
 *
 * Returns 1 when the macroblock carries no non-zero coefficient at all,
 * which lets the caller leave the inner edges unfiltered.
 */
static int vp8_parse_residuals(vp8_dec *d, int mb_x, vp8_bool *token_br) {
    const vp8_quant *q = &d->dqm[d->segment];
    uint8_t *tnz = d->top_nz + 9 * mb_x;
    uint8_t *lnz = d->left_nz;
    int16_t *dst = d->coeffs;
    int first, ac_type, any_nz = 0;
    int x, y, ch;

    memset(d->coeffs, 0, sizeof(d->coeffs));

    if (!d->is_i4x4) {
        /* The Y2 block holds the DC of each luma subblock; the inverse
         * Walsh-Hadamard transform scatters them back at stride 16. */
        int16_t dc[16];
        const int ctx = tnz[8] + lnz[8];
        int nz;
        memset(dc, 0, sizeof(dc));
        nz = vp8_get_coeffs(token_br, d->coeff_probs[1], ctx, q->y2, 0, dc);
        tnz[8] = lnz[8] = (uint8_t)(nz > 0);
        vp8_transform_wht(dc, d->coeffs);
        first = 1;
        ac_type = 0;
    } else {
        first = 0;
        ac_type = 3;
    }

    for (y = 0; y < 4; y++) {
        int l = lnz[y];
        for (x = 0; x < 4; x++) {
            const int ctx = l + tnz[x];
            const int nz = vp8_get_coeffs(token_br, d->coeff_probs[ac_type], ctx,
                                          q->y1, first, dst);
            l = (nz > first);
            tnz[x] = (uint8_t)l;
            if (nz > 1 || dst[0] != 0) any_nz = 1;
            dst += 16;
        }
        lnz[y] = (uint8_t)l;
    }

    for (ch = 0; ch < 2; ch++) {
        for (y = 0; y < 2; y++) {
            int l = lnz[4 + 2 * ch + y];
            for (x = 0; x < 2; x++) {
                const int ctx = l + tnz[4 + 2 * ch + x];
                const int nz = vp8_get_coeffs(token_br, d->coeff_probs[2], ctx,
                                              q->uv, 0, dst);
                l = (nz > 0);
                tnz[4 + 2 * ch + x] = (uint8_t)l;
                if (nz > 1 || dst[0] != 0) any_nz = 1;
                dst += 16;
            }
            lnz[4 + 2 * ch + y] = (uint8_t)l;
        }
    }
    return !any_nz;
}

/* ------------------------------------------------------------------------
 * Intra prediction (RFC 6386 section 12)
 * ------------------------------------------------------------------------ */

#define WEBP_AVG3(a, b, c) ((uint8_t)(((a) + 2 * (b) + (c) + 2) >> 2))
#define WEBP_AVG2(a, b) ((uint8_t)(((a) + (b) + 1) >> 1))

/* True motion prediction, shared by all block sizes. */
static void vp8_pred_tm(uint8_t *dst, int stride, int size) {
    const uint8_t *top = dst - stride;
    const int corner = top[-1];
    int y;
    for (y = 0; y < size; y++) {
        const int left = dst[-1];
        int x;
        for (x = 0; x < size; x++)
            dst[x] = (uint8_t)webp_clip_u8(left + top[x] - corner);
        dst += stride;
    }
}

static void vp8_pred_fill(uint8_t *dst, int stride, int size, int value) {
    int y;
    for (y = 0; y < size; y++) memset(dst + y * stride, value, (size_t)size);
}

/* 16x16 and chroma 8x8 share the same four modes; only the DC rounding and
 * the edge fallbacks differ with the block size. */
static void vp8_pred_block(uint8_t *dst, int stride, int size, int mode,
                           int have_top, int have_left) {
    switch (mode) {
        case WEBP_V_PRED: {
            int y;
            for (y = 0; y < size; y++)
                memcpy(dst + y * stride, dst - stride, (size_t)size);
            break;
        }
        case WEBP_H_PRED: {
            int y;
            for (y = 0; y < size; y++)
                memset(dst + y * stride, dst[y * stride - 1], (size_t)size);
            break;
        }
        case WEBP_TM_PRED:
            vp8_pred_tm(dst, stride, size);
            break;
        default: {                       /* WEBP_DC_PRED */
            int dc = 0, shift = (size == 16) ? 5 : 4, i;
            if (have_top && have_left) {
                dc = size;               /* rounding */
                for (i = 0; i < size; i++) dc += dst[i - stride] + dst[i * stride - 1];
            } else if (have_top) {
                dc = size / 2;
                for (i = 0; i < size; i++) dc += dst[i - stride];
                shift--;
            } else if (have_left) {
                dc = size / 2;
                for (i = 0; i < size; i++) dc += dst[i * stride - 1];
                shift--;
            } else {
                dc = 0x80 << shift;
            }
            vp8_pred_fill(dst, stride, size, dc >> shift);
            break;
        }
    }
}

/*
 * 4x4 luma prediction. The samples are passed explicitly because a subblock
 * on the right edge of a macroblock takes its above-right samples from the
 * row above the macroblock, not from the (not yet decoded) neighbour to the
 * right.
 *
 *   A[0..3] above, A[4..7] above-right, L[0..3] left, X the corner sample.
 */
static void vp8_pred4(uint8_t *dst, int stride, int mode,
                      const uint8_t *A, const uint8_t *L, int X) {
#define DST(x, y) dst[(x) + (y) * stride]
    switch (mode) {
        case WEBP_B_DC_PRED: {
            int dc = 4, i;
            for (i = 0; i < 4; i++) dc += A[i] + L[i];
            vp8_pred_fill(dst, stride, 4, dc >> 3);
            break;
        }
        case WEBP_B_TM_PRED: {
            int y;
            for (y = 0; y < 4; y++) {
                int x;
                for (x = 0; x < 4; x++)
                    DST(x, y) = (uint8_t)webp_clip_u8(L[y] + A[x] - X);
            }
            break;
        }
        case WEBP_B_VE_PRED: {
            const uint8_t vals[4] = {
                WEBP_AVG3(X, A[0], A[1]),
                WEBP_AVG3(A[0], A[1], A[2]),
                WEBP_AVG3(A[1], A[2], A[3]),
                WEBP_AVG3(A[2], A[3], A[4]),
            };
            int y;
            for (y = 0; y < 4; y++) memcpy(dst + y * stride, vals, 4);
            break;
        }
        case WEBP_B_HE_PRED: {
            const uint8_t vals[4] = {
                WEBP_AVG3(X, L[0], L[1]),
                WEBP_AVG3(L[0], L[1], L[2]),
                WEBP_AVG3(L[1], L[2], L[3]),
                WEBP_AVG3(L[2], L[3], L[3]),
            };
            int y;
            for (y = 0; y < 4; y++) memset(dst + y * stride, vals[y], 4);
            break;
        }
        case WEBP_B_RD_PRED:
            DST(0, 3) = WEBP_AVG3(L[1], L[2], L[3]);
            DST(1, 3) = DST(0, 2) = WEBP_AVG3(L[0], L[1], L[2]);
            DST(2, 3) = DST(1, 2) = DST(0, 1) = WEBP_AVG3(X, L[0], L[1]);
            DST(3, 3) = DST(2, 2) = DST(1, 1) = DST(0, 0) = WEBP_AVG3(A[0], X, L[0]);
            DST(3, 2) = DST(2, 1) = DST(1, 0) = WEBP_AVG3(A[1], A[0], X);
            DST(3, 1) = DST(2, 0) = WEBP_AVG3(A[2], A[1], A[0]);
            DST(3, 0) = WEBP_AVG3(A[3], A[2], A[1]);
            break;
        case WEBP_B_VR_PRED:
            DST(0, 0) = DST(1, 2) = WEBP_AVG2(X, A[0]);
            DST(1, 0) = DST(2, 2) = WEBP_AVG2(A[0], A[1]);
            DST(2, 0) = DST(3, 2) = WEBP_AVG2(A[1], A[2]);
            DST(3, 0) = WEBP_AVG2(A[2], A[3]);
            DST(0, 3) = WEBP_AVG3(L[2], L[1], L[0]);
            DST(0, 2) = WEBP_AVG3(L[1], L[0], X);
            DST(0, 1) = DST(1, 3) = WEBP_AVG3(L[0], X, A[0]);
            DST(1, 1) = DST(2, 3) = WEBP_AVG3(X, A[0], A[1]);
            DST(2, 1) = DST(3, 3) = WEBP_AVG3(A[0], A[1], A[2]);
            DST(3, 1) = WEBP_AVG3(A[1], A[2], A[3]);
            break;
        case WEBP_B_LD_PRED:
            DST(0, 0) = WEBP_AVG3(A[0], A[1], A[2]);
            DST(1, 0) = DST(0, 1) = WEBP_AVG3(A[1], A[2], A[3]);
            DST(2, 0) = DST(1, 1) = DST(0, 2) = WEBP_AVG3(A[2], A[3], A[4]);
            DST(3, 0) = DST(2, 1) = DST(1, 2) = DST(0, 3) = WEBP_AVG3(A[3], A[4], A[5]);
            DST(3, 1) = DST(2, 2) = DST(1, 3) = WEBP_AVG3(A[4], A[5], A[6]);
            DST(3, 2) = DST(2, 3) = WEBP_AVG3(A[5], A[6], A[7]);
            DST(3, 3) = WEBP_AVG3(A[6], A[7], A[7]);
            break;
        case WEBP_B_VL_PRED:
            DST(0, 0) = WEBP_AVG2(A[0], A[1]);
            DST(1, 0) = DST(0, 2) = WEBP_AVG2(A[1], A[2]);
            DST(2, 0) = DST(1, 2) = WEBP_AVG2(A[2], A[3]);
            DST(3, 0) = DST(2, 2) = WEBP_AVG2(A[3], A[4]);
            DST(0, 1) = WEBP_AVG3(A[0], A[1], A[2]);
            DST(1, 1) = DST(0, 3) = WEBP_AVG3(A[1], A[2], A[3]);
            DST(2, 1) = DST(1, 3) = WEBP_AVG3(A[2], A[3], A[4]);
            DST(3, 1) = DST(2, 3) = WEBP_AVG3(A[3], A[4], A[5]);
            DST(3, 2) = WEBP_AVG3(A[4], A[5], A[6]);
            DST(3, 3) = WEBP_AVG3(A[5], A[6], A[7]);
            break;
        case WEBP_B_HD_PRED:
            DST(0, 0) = DST(2, 1) = WEBP_AVG2(L[0], X);
            DST(0, 1) = DST(2, 2) = WEBP_AVG2(L[1], L[0]);
            DST(0, 2) = DST(2, 3) = WEBP_AVG2(L[2], L[1]);
            DST(0, 3) = WEBP_AVG2(L[3], L[2]);
            DST(3, 0) = WEBP_AVG3(A[0], A[1], A[2]);
            DST(2, 0) = WEBP_AVG3(X, A[0], A[1]);
            DST(1, 0) = DST(3, 1) = WEBP_AVG3(L[0], X, A[0]);
            DST(1, 1) = DST(3, 2) = WEBP_AVG3(L[1], L[0], X);
            DST(1, 2) = DST(3, 3) = WEBP_AVG3(L[2], L[1], L[0]);
            DST(1, 3) = WEBP_AVG3(L[3], L[2], L[1]);
            break;
        default:                         /* WEBP_B_HU_PRED */
            DST(0, 0) = WEBP_AVG2(L[0], L[1]);
            DST(2, 0) = DST(0, 1) = WEBP_AVG2(L[1], L[2]);
            DST(2, 1) = DST(0, 2) = WEBP_AVG2(L[2], L[3]);
            DST(1, 0) = WEBP_AVG3(L[0], L[1], L[2]);
            DST(3, 0) = DST(1, 1) = WEBP_AVG3(L[1], L[2], L[3]);
            DST(3, 1) = DST(1, 2) = WEBP_AVG3(L[2], L[3], L[3]);
            DST(3, 2) = DST(2, 2) = DST(0, 3) = DST(1, 3) = DST(2, 3) =
                DST(3, 3) = L[3];
            break;
    }
#undef DST
}

/* ------------------------------------------------------------------------
 * Macroblock reconstruction
 * ------------------------------------------------------------------------ */

/* Border samples for macroblocks on the frame edges: 127 above, 129 to the
 * left (paragraph 12.2). */
static void vp8_init_top_border(vp8_dec *d) {
    const int y_span = d->mb_w * 16 + 5;    /* includes above-right samples */
    const int uv_span = d->mb_w * 8 + 5;
    memset(vp8_y_at(d, -1, -1), 127, (size_t)y_span);
    memset(vp8_u_at(d, -1, -1), 127, (size_t)uv_span);
    memset(vp8_v_at(d, -1, -1), 127, (size_t)uv_span);
}

static void vp8_init_left_border(vp8_dec *d, int mb_y) {
    int i;
    for (i = 0; i < 16; i++) *vp8_y_at(d, -1, mb_y * 16 + i) = 129;
    for (i = 0; i < 8; i++) {
        *vp8_u_at(d, -1, mb_y * 8 + i) = 129;
        *vp8_v_at(d, -1, mb_y * 8 + i) = 129;
    }
    if (mb_y > 0) {
        *vp8_y_at(d, -1, mb_y * 16 - 1) = 129;
        *vp8_u_at(d, -1, mb_y * 8 - 1) = 129;
        *vp8_v_at(d, -1, mb_y * 8 - 1) = 129;
    }
}

static void vp8_reconstruct(vp8_dec *d, int mb_x, int mb_y) {
    const int px = mb_x * 16, py = mb_y * 16;
    const int cx = mb_x * 8, cy = mb_y * 8;
    const int have_top = (mb_y > 0), have_left = (mb_x > 0);
    uint8_t *ydst = vp8_y_at(d, px, py);
    uint8_t *udst = vp8_u_at(d, cx, cy);
    uint8_t *vdst = vp8_v_at(d, cx, cy);
    const int ys = d->y_stride, cs = d->uv_stride;
    int n;

    /* The rightmost macroblock has no neighbour to supply above-right
     * samples, so the last above sample is replicated. */
    if (mb_x == d->mb_w - 1 && have_top) {
        uint8_t *above = vp8_y_at(d, px, py - 1);
        memset(above + 16, above[15], 4);
    }

    if (d->is_i4x4) {
        /* Predict and reconstruct each subblock in raster order, so that a
         * subblock always sees its already reconstructed neighbours. */
        for (n = 0; n < 16; n++) {
            const int sx = n & 3, sy = n >> 2;
            uint8_t *dst = ydst + sy * 4 * ys + sx * 4;
            uint8_t A[8], L[4];
            int X, i;
            memcpy(A, dst - ys, 4);
            if (sx == 3 && sy > 0) {
                /* Above-right comes from the row above the macroblock. */
                memcpy(A + 4, vp8_y_at(d, px + 16, py - 1), 4);
            } else {
                memcpy(A + 4, dst - ys + 4, 4);
            }
            for (i = 0; i < 4; i++) L[i] = dst[i * ys - 1];
            X = dst[-ys - 1];
            vp8_pred4(dst, ys, d->imodes[n], A, L, X);
            vp8_transform_add(d->coeffs + 16 * n, dst, ys);
        }
    } else {
        vp8_pred_block(ydst, ys, 16, d->imodes[0], have_top, have_left);
        for (n = 0; n < 16; n++)
            vp8_transform_add(d->coeffs + 16 * n,
                              ydst + (n >> 2) * 4 * ys + (n & 3) * 4, ys);
    }

    vp8_pred_block(udst, cs, 8, d->uvmode, have_top, have_left);
    vp8_pred_block(vdst, cs, 8, d->uvmode, have_top, have_left);
    for (n = 0; n < 4; n++) {
        vp8_transform_add(d->coeffs + 16 * (16 + n),
                          udst + (n >> 1) * 4 * cs + (n & 1) * 4, cs);
        vp8_transform_add(d->coeffs + 16 * (20 + n),
                          vdst + (n >> 1) * 4 * cs + (n & 1) * 4, cs);
    }
}

/* ------------------------------------------------------------------------
 * Loop filter (RFC 6386 section 15)
 * ------------------------------------------------------------------------ */

static void vp8_filter2(uint8_t *p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0) + webp_clip_s8(p1 - q1);
    const int a1 = webp_clip_s4((a + 4) >> 3);
    const int a2 = webp_clip_s4((a + 3) >> 3);
    p[-step] = (uint8_t)webp_clip_u8(p0 + a2);
    p[0] = (uint8_t)webp_clip_u8(q0 - a1);
}

static void vp8_filter4(uint8_t *p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0);
    const int a1 = webp_clip_s4((a + 4) >> 3);
    const int a2 = webp_clip_s4((a + 3) >> 3);
    const int a3 = (a1 + 1) >> 1;
    p[-2 * step] = (uint8_t)webp_clip_u8(p1 + a3);
    p[-step] = (uint8_t)webp_clip_u8(p0 + a2);
    p[0] = (uint8_t)webp_clip_u8(q0 - a1);
    p[step] = (uint8_t)webp_clip_u8(q1 - a3);
}

static void vp8_filter6(uint8_t *p, int step) {
    const int p2 = p[-3 * step], p1 = p[-2 * step], p0 = p[-step];
    const int q0 = p[0], q1 = p[step], q2 = p[2 * step];
    const int a = webp_clip_s8(3 * (q0 - p0) + webp_clip_s8(p1 - q1));
    const int a1 = (27 * a + 63) >> 7;
    const int a2 = (18 * a + 63) >> 7;
    const int a3 = (9 * a + 63) >> 7;
    p[-3 * step] = (uint8_t)webp_clip_u8(p2 + a3);
    p[-2 * step] = (uint8_t)webp_clip_u8(p1 + a2);
    p[-step] = (uint8_t)webp_clip_u8(p0 + a1);
    p[0] = (uint8_t)webp_clip_u8(q0 - a1);
    p[step] = (uint8_t)webp_clip_u8(q1 - a2);
    p[2 * step] = (uint8_t)webp_clip_u8(q2 - a3);
}

static int vp8_hev(const uint8_t *p, int step, int thresh) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    return webp_abs(p1 - p0) > thresh || webp_abs(q1 - q0) > thresh;
}

static int vp8_needs_filter(const uint8_t *p, int step, int t) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    return (4 * webp_abs(p0 - q0) + webp_abs(p1 - q1)) <= t;
}

static int vp8_needs_filter2(const uint8_t *p, int step, int t, int it) {
    const int p3 = p[-4 * step], p2 = p[-3 * step], p1 = p[-2 * step];
    const int p0 = p[-step], q0 = p[0];
    const int q1 = p[step], q2 = p[2 * step], q3 = p[3 * step];
    if ((4 * webp_abs(p0 - q0) + webp_abs(p1 - q1)) > t) return 0;
    return webp_abs(p3 - p2) <= it && webp_abs(p2 - p1) <= it &&
           webp_abs(p1 - p0) <= it && webp_abs(q3 - q2) <= it &&
           webp_abs(q2 - q1) <= it && webp_abs(q1 - q0) <= it;
}

/* Simple filter (paragraph 15.2), luma only. */
static void vp8_simple_filter(uint8_t *p, int hstride, int vstride, int count,
                              int thresh) {
    const int thresh2 = 2 * thresh + 1;
    int i;
    for (i = 0; i < count; i++) {
        uint8_t *q = p + i * vstride;
        if (vp8_needs_filter(q, hstride, thresh2)) vp8_filter2(q, hstride);
    }
}

/* Normal filter (paragraph 15.3). 'mb_edge' selects the wider filter used on
 * macroblock boundaries. */
static void vp8_normal_filter(uint8_t *p, int hstride, int vstride, int count,
                              int thresh, int ithresh, int hev_thresh,
                              int mb_edge) {
    const int thresh2 = 2 * thresh + 1;
    int i;
    for (i = 0; i < count; i++) {
        uint8_t *q = p + i * vstride;
        if (!vp8_needs_filter2(q, hstride, thresh2, ithresh)) continue;
        if (vp8_hev(q, hstride, hev_thresh)) {
            vp8_filter2(q, hstride);
        } else if (mb_edge) {
            vp8_filter6(q, hstride);
        } else {
            vp8_filter4(q, hstride);
        }
    }
}

static void vp8_filter_mb(vp8_dec *d, int mb_x, int mb_y) {
    const vp8_finfo *f = &d->fstore[(size_t)mb_y * d->mb_w + mb_x];
    const int limit = f->limit;
    const int ys = d->y_stride, cs = d->uv_stride;
    uint8_t *ydst, *udst, *vdst;
    int k;

    if (limit == 0) return;
    ydst = vp8_y_at(d, mb_x * 16, mb_y * 16);
    udst = vp8_u_at(d, mb_x * 8, mb_y * 8);
    vdst = vp8_v_at(d, mb_x * 8, mb_y * 8);

    if (d->filter_type == 1) {
        if (mb_x > 0) vp8_simple_filter(ydst, 1, ys, 16, limit + 4);
        if (f->inner)
            for (k = 1; k <= 3; k++)
                vp8_simple_filter(ydst + 4 * k, 1, ys, 16, limit);
        if (mb_y > 0) vp8_simple_filter(ydst, ys, 1, 16, limit + 4);
        if (f->inner)
            for (k = 1; k <= 3; k++)
                vp8_simple_filter(ydst + 4 * k * ys, ys, 1, 16, limit);
        return;
    }

    {
        const int ilevel = f->ilevel, hev = f->hev;
        if (mb_x > 0) {
            vp8_normal_filter(ydst, 1, ys, 16, limit + 4, ilevel, hev, 1);
            vp8_normal_filter(udst, 1, cs, 8, limit + 4, ilevel, hev, 1);
            vp8_normal_filter(vdst, 1, cs, 8, limit + 4, ilevel, hev, 1);
        }
        if (f->inner) {
            for (k = 1; k <= 3; k++)
                vp8_normal_filter(ydst + 4 * k, 1, ys, 16, limit, ilevel, hev, 0);
            vp8_normal_filter(udst + 4, 1, cs, 8, limit, ilevel, hev, 0);
            vp8_normal_filter(vdst + 4, 1, cs, 8, limit, ilevel, hev, 0);
        }
        if (mb_y > 0) {
            vp8_normal_filter(ydst, ys, 1, 16, limit + 4, ilevel, hev, 1);
            vp8_normal_filter(udst, cs, 1, 8, limit + 4, ilevel, hev, 1);
            vp8_normal_filter(vdst, cs, 1, 8, limit + 4, ilevel, hev, 1);
        }
        if (f->inner) {
            for (k = 1; k <= 3; k++)
                vp8_normal_filter(ydst + 4 * k * ys, ys, 1, 16, limit, ilevel,
                                  hev, 0);
            vp8_normal_filter(udst + 4 * cs, cs, 1, 8, limit, ilevel, hev, 0);
            vp8_normal_filter(vdst + 4 * cs, cs, 1, 8, limit, ilevel, hev, 0);
        }
    }
}

/* ------------------------------------------------------------------------
 * YUV 4:2:0 -> RGB (BT.601, matching libwebp's fixed point arithmetic)
 * ------------------------------------------------------------------------ */

#define WEBP_YUV_FIX2 6
#define WEBP_YUV_MASK2 ((256 << WEBP_YUV_FIX2) - 1)

static int webp_yuv_mult_hi(int v, int coeff) { return (v * coeff) >> 8; }

static int webp_yuv_clip8(int v) {
    return ((v & ~WEBP_YUV_MASK2) == 0) ? (v >> WEBP_YUV_FIX2)
                                        : (v < 0 ? 0 : 255);
}

static void webp_yuv_to_rgb(int y, int u, int v, uint8_t *rgb) {
    rgb[0] = (uint8_t)webp_yuv_clip8(webp_yuv_mult_hi(y, 19077) +
                                     webp_yuv_mult_hi(v, 26149) - 14234);
    rgb[1] = (uint8_t)webp_yuv_clip8(webp_yuv_mult_hi(y, 19077) -
                                     webp_yuv_mult_hi(u, 6419) -
                                     webp_yuv_mult_hi(v, 13320) + 8708);
    rgb[2] = (uint8_t)webp_yuv_clip8(webp_yuv_mult_hi(y, 19077) +
                                     webp_yuv_mult_hi(u, 33050) - 17685);
}

/* ------------------------------------------------------------------------
 * Lossy (VP8) frame decoding
 * ------------------------------------------------------------------------ */

static void vp8_free(vp8_dec *d) {
    free(d->y_plane);
    free(d->u_plane);
    free(d->v_plane);
    free(d->fstore);
    free(d->top_modes);
    free(d->top_nz);
    memset(d, 0, sizeof(*d));
}

static webp_image *webp_image_alloc(int width, int height, int channels) {
    webp_image *img = (webp_image *)calloc(1, sizeof(*img));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t *)malloc((size_t)width * (size_t)height *
                                  (size_t)channels);
    if (!img->data) {
        free(img);
        return NULL;
    }
    return img;
}

static int webp_dims_ok(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return 0;
    if (w > WEBP_MAX_DIMENSION || h > WEBP_MAX_DIMENSION) return 0;
    if ((uint64_t)w * h > WEBP_MAX_PIXELS) return 0;
    return 1;
}

static webp_image *webp_decode_vp8(const uint8_t *data, size_t len,
                                   char *err, size_t err_cap) {
    vp8_dec d;
    webp_image *img = NULL;
    uint32_t tag, first_part_size;
    uint32_t w, h;
    size_t part_off;
    int mb_x, mb_y, p;

    memset(&d, 0, sizeof(d));

    if (len < 10) {
        webp_err_set(err, err_cap, "truncated VP8 chunk");
        return NULL;
    }
    tag = webp_le24(data);
    if (tag & 1) {
        webp_err_set(err, err_cap, "VP8 stream does not start with a key frame");
        return NULL;
    }
    first_part_size = (tag >> 5) & 0x7ffff;
    if (data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a) {
        webp_err_set(err, err_cap, "bad VP8 key frame start code");
        return NULL;
    }
    w = webp_le16(data + 6) & 0x3fff;
    h = webp_le16(data + 8) & 0x3fff;
    if (!webp_dims_ok(w, h)) {
        webp_err_set(err, err_cap, "unsupported VP8 image dimensions");
        return NULL;
    }
    if ((size_t)first_part_size + 10 > len) {
        webp_err_set(err, err_cap, "truncated VP8 first partition");
        return NULL;
    }

    d.width = (int)w;
    d.height = (int)h;
    d.mb_w = ((int)w + 15) / 16;
    d.mb_h = ((int)h + 15) / 16;

    vp8_bool_init(&d.br, data + 10, first_part_size);
    vp8_get_bit(&d.br);                 /* color space */
    vp8_get_bit(&d.br);                 /* clamping type */
    if (!vp8_parse_segment_header(&d) || !vp8_parse_filter_header(&d)) {
        webp_err_set(err, err_cap, "corrupt VP8 frame header");
        goto fail;
    }

    /* Token partitions follow the first partition; their sizes are stored as
     * three byte little endian values at the start of that area. */
    d.num_parts = 1 << vp8_get_uint(&d.br, 2);
    part_off = 10 + first_part_size;
    {
        const size_t table = 3 * (size_t)(d.num_parts - 1);
        size_t avail;
        const uint8_t *sizes = data + part_off;
        if (part_off + table > len) {
            webp_err_set(err, err_cap, "truncated VP8 partition table");
            goto fail;
        }
        part_off += table;
        avail = len - part_off;
        for (p = 0; p < d.num_parts - 1; p++) {
            size_t psize = webp_le24(sizes + 3 * p);
            if (psize > avail) psize = avail;
            vp8_bool_init(&d.parts[p], data + part_off, psize);
            part_off += psize;
            avail -= psize;
        }
        vp8_bool_init(&d.parts[d.num_parts - 1], data + part_off, avail);
    }

    vp8_parse_quant(&d);
    vp8_get_bit(&d.br);                 /* refresh entropy probabilities */
    memcpy(d.coeff_probs, kCoeffsProba0, sizeof(d.coeff_probs));
    vp8_parse_proba(&d);
    vp8_precompute_filter_strengths(&d);

    /* Planes carry a one pixel top/left border plus room on the right for
     * the four above-right prediction samples. */
    d.y_stride = d.mb_w * 16 + 8;
    d.uv_stride = d.mb_w * 8 + 8;
    d.y_plane = (uint8_t *)calloc((size_t)d.y_stride * (d.mb_h * 16 + 1), 1);
    d.u_plane = (uint8_t *)calloc((size_t)d.uv_stride * (d.mb_h * 8 + 1), 1);
    d.v_plane = (uint8_t *)calloc((size_t)d.uv_stride * (d.mb_h * 8 + 1), 1);
    d.fstore = (vp8_finfo *)calloc((size_t)d.mb_w * d.mb_h, sizeof(vp8_finfo));
    d.top_modes = (uint8_t *)malloc((size_t)d.mb_w * 4);
    d.top_nz = (uint8_t *)calloc((size_t)d.mb_w * 9, 1);
    if (!d.y_plane || !d.u_plane || !d.v_plane || !d.fstore || !d.top_modes ||
        !d.top_nz) {
        webp_err_set(err, err_cap, "out of memory decoding VP8 image");
        goto fail;
    }
    memset(d.top_modes, WEBP_B_DC_PRED, (size_t)d.mb_w * 4);
    vp8_init_top_border(&d);

    for (mb_y = 0; mb_y < d.mb_h; mb_y++) {
        vp8_bool *token_br = &d.parts[mb_y & (d.num_parts - 1)];
        memset(d.left_modes, WEBP_B_DC_PRED, sizeof(d.left_modes));
        memset(d.left_nz, 0, sizeof(d.left_nz));
        vp8_init_left_border(&d, mb_y);
        for (mb_x = 0; mb_x < d.mb_w; mb_x++) {
            int empty;
            vp8_parse_intra_modes(&d, mb_x);
            if (d.skip) {
                uint8_t *tnz = d.top_nz + 9 * mb_x;
                memset(d.coeffs, 0, sizeof(d.coeffs));
                memset(tnz, 0, 8);
                memset(d.left_nz, 0, 8);
                if (!d.is_i4x4) tnz[8] = d.left_nz[8] = 0;
                empty = 1;
            } else {
                empty = vp8_parse_residuals(&d, mb_x, token_br);
            }
            vp8_reconstruct(&d, mb_x, mb_y);
            if (d.filter_type > 0) {
                vp8_finfo *f = &d.fstore[(size_t)mb_y * d.mb_w + mb_x];
                *f = d.fstrength[d.segment][d.is_i4x4];
                /* Macroblocks that carry residuals also need their inner
                 * edges filtered, whatever the prediction size. */
                if (!empty) f->inner = 1;
            }
        }
        if (d.br.eof) {
            webp_err_set(err, err_cap, "truncated VP8 bitstream");
            goto fail;
        }
    }

    if (d.filter_type > 0)
        for (mb_y = 0; mb_y < d.mb_h; mb_y++)
            for (mb_x = 0; mb_x < d.mb_w; mb_x++) vp8_filter_mb(&d, mb_x, mb_y);

    img = webp_image_alloc(d.width, d.height, 3);
    if (!img) {
        webp_err_set(err, err_cap, "out of memory decoding VP8 image");
        goto fail;
    }
    {
        int y;
        for (y = 0; y < d.height; y++) {
            const uint8_t *ysrc = vp8_y_at(&d, 0, y);
            const uint8_t *usrc = vp8_u_at(&d, 0, y >> 1);
            const uint8_t *vsrc = vp8_v_at(&d, 0, y >> 1);
            uint8_t *dst = img->data + (size_t)y * d.width * 3;
            int x;
            for (x = 0; x < d.width; x++)
                webp_yuv_to_rgb(ysrc[x], usrc[x >> 1], vsrc[x >> 1],
                                dst + x * 3);
        }
    }

fail:
    vp8_free(&d);
    return img;
}

/* ------------------------------------------------------------------------
 * Container parsing (RIFF)
 * ------------------------------------------------------------------------ */

int webp_is_webp(const uint8_t *data, size_t len) {
    return data && len >= 12 && !memcmp(data, "RIFF", 4) &&
           !memcmp(data + 8, "WEBP", 4);
}

/* ------------------------------------------------------------------------
 * Lossless (VP8L) decoding
 *
 * Pixels are carried as 0xAARRGGBB words throughout, which is the layout the
 * transforms and the color cache are defined against.
 * ------------------------------------------------------------------------ */

#define VP8L_MAGIC 0x2f
#define VP8L_LITERAL_CODES 256
#define VP8L_LENGTH_CODES 24
#define VP8L_DISTANCE_CODES 40
#define VP8L_CODE_LENGTH_CODES 19
#define VP8L_MAX_CODE_LENGTH 15
#define VP8L_MAX_CACHE_BITS 11
#define VP8L_HUFF_TREES 5          /* green, red, blue, alpha, distance */
#define VP8L_MAX_TRANSFORMS 4
#define VP8L_ARGB_BLACK 0xff000000u

enum {
    VP8L_PREDICTOR_TRANSFORM = 0,
    VP8L_CROSS_COLOR_TRANSFORM = 1,
    VP8L_SUBTRACT_GREEN = 2,
    VP8L_COLOR_INDEXING_TRANSFORM = 3
};

static const uint8_t kCodeLengthCodeOrder[VP8L_CODE_LENGTH_CODES] = {
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};
static const uint8_t kCodeLengthExtraBits[3] = {2, 3, 7};
static const uint8_t kCodeLengthRepeatOffsets[3] = {3, 3, 11};

#define VP8L_CODE_TO_PLANE_CODES 120
static const uint8_t kCodeToPlane[VP8L_CODE_TO_PLANE_CODES] = {
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a,
    0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04,
    0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45,
    0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
    0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
    0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e,
    0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f, 0x64, 0x6c, 0x42, 0x4e,
    0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
    0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d,
    0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70
};

/* --- Bit reader: least significant bit first --- */

typedef struct {
    const uint8_t *buf;
    size_t len;
    size_t pos;
    uint64_t value;
    int bits;
    int eos;
} vp8l_bits;

static void vp8l_bits_init(vp8l_bits *br, const uint8_t *buf, size_t len) {
    br->buf = buf;
    br->len = len;
    br->pos = 0;
    br->value = 0;
    br->bits = 0;
    br->eos = 0;
}

static uint32_t vp8l_read(vp8l_bits *br, int n) {
    uint32_t v;
    while (br->bits < n) {
        uint64_t byte = 0;
        if (br->pos < br->len) {
            byte = br->buf[br->pos++];
        } else {
            br->eos = 1;
        }
        br->value |= byte << br->bits;
        br->bits += 8;
    }
    v = (uint32_t)(br->value & ((n == 32) ? 0xffffffffu : ((1u << n) - 1u)));
    br->value >>= n;
    br->bits -= n;
    return v;
}

/* --- Canonical prefix codes, decoded one bit at a time --- */

typedef struct {
    uint16_t counts[VP8L_MAX_CODE_LENGTH + 1];
    uint16_t *symbols;
    int num_symbols;
} vp8l_huff;

static void vp8l_huff_free(vp8l_huff *h) {
    free(h->symbols);
    h->symbols = NULL;
}

/*
 * Builds a canonical code from a list of code lengths, as in DEFLATE.
 * Returns 0 if the lengths do not describe a complete code.
 */
static int vp8l_huff_build(vp8l_huff *h, const uint8_t *lengths, int n) {
    int len, symbol, left;
    uint16_t offsets[VP8L_MAX_CODE_LENGTH + 2];

    memset(h->counts, 0, sizeof(h->counts));
    h->symbols = NULL;
    h->num_symbols = 0;
    for (symbol = 0; symbol < n; symbol++) {
        if (lengths[symbol] > VP8L_MAX_CODE_LENGTH) return 0;
        h->counts[lengths[symbol]]++;
    }
    if (h->counts[0] == n) return 0;        /* no symbol at all */

    /* A single symbol is coded with zero bits, which is a valid special case
     * in the WebP lossless format. */
    left = 1;
    for (len = 1; len <= VP8L_MAX_CODE_LENGTH; len++) {
        left <<= 1;
        left -= h->counts[len];
        if (left < 0) return 0;             /* over-subscribed */
    }

    offsets[1] = 0;
    for (len = 1; len <= VP8L_MAX_CODE_LENGTH; len++)
        offsets[len + 1] = (uint16_t)(offsets[len] + h->counts[len]);
    h->num_symbols = offsets[VP8L_MAX_CODE_LENGTH + 1];
    h->symbols = (uint16_t *)calloc((size_t)(h->num_symbols ? h->num_symbols : 1),
                                    sizeof(uint16_t));
    if (!h->symbols) return 0;
    for (symbol = 0; symbol < n; symbol++)
        if (lengths[symbol]) h->symbols[offsets[lengths[symbol]]++] = (uint16_t)symbol;

    /* 'left' > 0 means an incomplete code, which is only allowed when a
     * single symbol carries the whole alphabet. */
    if (left > 0 && h->num_symbols != 1) return 0;
    return 1;
}

static int vp8l_huff_read(const vp8l_huff *h, vp8l_bits *br) {
    int len, code = 0, first = 0, index = 0;
    if (h->num_symbols == 1) return h->symbols[0];   /* zero-bit code */
    for (len = 1; len <= VP8L_MAX_CODE_LENGTH; len++) {
        const int count = h->counts[len];
        code |= (int)vp8l_read(br, 1);
        if (code - first < count) return h->symbols[index + code - first];
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

/* --- Color cache --- */

typedef struct {
    uint32_t *colors;
    int bits;
} vp8l_cache;

static int vp8l_cache_key(uint32_t argb, int bits) {
    /* Deliberately a wrapping 32 bit multiply. */
    return (int)((argb * 0x1e35a7bdu) >> (32 - bits));
}

/* --- Decoder state --- */

typedef struct {
    int xsize, ysize;               /* size of the entropy coded image */
    int type;
    int bits;
    uint32_t *data;
} vp8l_transform;

typedef struct {
    vp8l_huff trees[VP8L_HUFF_TREES];
} vp8l_group;

typedef struct {
    vp8l_bits br;
    int width, height;
    int has_alpha;
    vp8l_transform transforms[VP8L_MAX_TRANSFORMS];
    int num_transforms;
    uint32_t transforms_seen;
    const char *error;
} vp8l_dec;

static int vp8l_sub_size(int size, int bits) {
    return (size + (1 << bits) - 1) >> bits;
}

static void vp8l_groups_free(vp8l_group *groups, int n) {
    int i, j;
    if (!groups) return;
    for (i = 0; i < n; i++)
        for (j = 0; j < VP8L_HUFF_TREES; j++) vp8l_huff_free(&groups[i].trees[j]);
    free(groups);
}

static int vp8l_decode_image(vp8l_dec *d, int xsize, int ysize, int is_level0,
                             uint32_t **out);

/* Reads the code lengths of one prefix code (specification: "Decoding and
 * Building the Prefix Codes"). */
static int vp8l_read_code_lengths(vp8l_dec *d, const uint8_t *cl_lengths,
                                  int num_symbols, uint8_t *lengths) {
    vp8l_bits *br = &d->br;
    vp8l_huff cl;
    int symbol = 0, max_symbol, prev_len = 8, ok = 1;

    if (!vp8l_huff_build(&cl, cl_lengths, VP8L_CODE_LENGTH_CODES)) {
        d->error = "corrupt VP8L code length code";
        return 0;
    }
    if (vp8l_read(br, 1)) {
        const int nbits = 2 + 2 * (int)vp8l_read(br, 3);
        max_symbol = 2 + (int)vp8l_read(br, nbits);
        if (max_symbol > num_symbols) {
            d->error = "corrupt VP8L code length limit";
            ok = 0;
        }
    } else {
        max_symbol = num_symbols;
    }

    while (ok && symbol < num_symbols) {
        int len;
        if (max_symbol-- == 0) break;
        len = vp8l_huff_read(&cl, br);
        if (len < 0 || br->eos) {
            ok = 0;
            break;
        }
        if (len < 16) {
            lengths[symbol++] = (uint8_t)len;
            if (len != 0) prev_len = len;
        } else {
            const int slot = len - 16;
            const int use_prev = (len == 16);
            int repeat = (int)vp8l_read(br, kCodeLengthExtraBits[slot]) +
                         kCodeLengthRepeatOffsets[slot];
            if (symbol + repeat > num_symbols) {
                ok = 0;
                break;
            }
            while (repeat-- > 0) lengths[symbol++] = (uint8_t)(use_prev ? prev_len : 0);
        }
    }
    vp8l_huff_free(&cl);
    if (!ok && !d->error) d->error = "corrupt VP8L prefix code";
    return ok;
}

static int vp8l_read_huffman_code(vp8l_dec *d, int alphabet_size,
                                  vp8l_huff *tree) {
    vp8l_bits *br = &d->br;
    /* A simple code may name a symbol using eight bits even when the
     * alphabet is smaller, so the scratch buffer is never below 256. */
    const int cap = alphabet_size > 256 ? alphabet_size : 256;
    uint8_t *lengths = (uint8_t *)calloc((size_t)cap, 1);
    int ok;

    if (!lengths) {
        d->error = "out of memory decoding VP8L image";
        return 0;
    }
    if (vp8l_read(br, 1)) {             /* simple code */
        const int num_symbols = (int)vp8l_read(br, 1) + 1;
        const int first_wide = (int)vp8l_read(br, 1);
        int symbol = (int)vp8l_read(br, first_wide ? 8 : 1);
        lengths[symbol] = 1;
        if (num_symbols == 2) {
            symbol = (int)vp8l_read(br, 8);
            lengths[symbol] = 1;
        }
        ok = 1;
    } else {
        uint8_t cl_lengths[VP8L_CODE_LENGTH_CODES];
        const int num_codes = (int)vp8l_read(br, 4) + 4;
        int i;
        memset(cl_lengths, 0, sizeof(cl_lengths));
        for (i = 0; i < num_codes; i++)
            cl_lengths[kCodeLengthCodeOrder[i]] = (uint8_t)vp8l_read(br, 3);
        ok = vp8l_read_code_lengths(d, cl_lengths, alphabet_size, lengths);
    }
    /* Built over the real alphabet: a simple code naming a symbol outside it
     * leaves no usable code, which the build below rejects. */
    ok = ok && !br->eos && vp8l_huff_build(tree, lengths, alphabet_size);
    free(lengths);
    if (!ok && !d->error) d->error = "corrupt VP8L prefix code";
    return ok;
}

/*
 * Reads the prefix codes for an entropy image, including the optional meta
 * prefix code image that assigns a different code group to each tile.
 */
static int vp8l_read_huffman_codes(vp8l_dec *d, int xsize, int ysize,
                                   int cache_bits, int allow_recursion,
                                   vp8l_group **out_groups, int *out_num,
                                   uint32_t **out_image, int *out_bits,
                                   int *out_image_xsize) {
    vp8l_bits *br = &d->br;
    uint32_t *image = NULL;
    vp8l_group *groups = NULL;
    int num_groups = 1;
    int precision = 0;
    int image_xsize = 0;
    int i, j;

    if (allow_recursion && vp8l_read(br, 1)) {
        const int hx = vp8l_sub_size(xsize, precision = 2 + (int)vp8l_read(br, 3));
        const int hy = vp8l_sub_size(ysize, precision);
        int npix = hx * hy;
        if (!vp8l_decode_image(d, hx, hy, 0, &image)) return 0;
        image_xsize = hx;
        for (i = 0; i < npix; i++) {
            /* The group index lives in the red and green bytes. */
            const int group = (int)((image[i] >> 8) & 0xffff);
            image[i] = (uint32_t)group;
            if (group >= num_groups) num_groups = group + 1;
        }
        if (num_groups > npix) {
            free(image);
            d->error = "corrupt VP8L meta prefix code image";
            return 0;
        }
    }

    groups = (vp8l_group *)calloc((size_t)num_groups, sizeof(vp8l_group));
    if (!groups) {
        free(image);
        d->error = "out of memory decoding VP8L image";
        return 0;
    }
    for (i = 0; i < num_groups; i++) {
        for (j = 0; j < VP8L_HUFF_TREES; j++) {
            int alphabet = (j == 0)
                ? VP8L_LITERAL_CODES + VP8L_LENGTH_CODES +
                      (cache_bits > 0 ? (1 << cache_bits) : 0)
                : (j == 4 ? VP8L_DISTANCE_CODES : VP8L_LITERAL_CODES);
            if (!vp8l_read_huffman_code(d, alphabet, &groups[i].trees[j])) {
                vp8l_groups_free(groups, num_groups);
                free(image);
                return 0;
            }
        }
    }
    *out_groups = groups;
    *out_num = num_groups;
    *out_image = image;
    *out_bits = precision;
    *out_image_xsize = image_xsize;
    return 1;
}

static int vp8l_copy_distance(vp8l_bits *br, int symbol) {
    int extra_bits, offset;
    if (symbol < 4) return symbol + 1;
    extra_bits = (symbol - 2) >> 1;
    offset = (2 + (symbol & 1)) << extra_bits;
    return offset + (int)vp8l_read(br, extra_bits) + 1;
}

static int vp8l_plane_to_distance(int xsize, int plane_code) {
    if (plane_code > VP8L_CODE_TO_PLANE_CODES) {
        return plane_code - VP8L_CODE_TO_PLANE_CODES;
    } else {
        const int code = kCodeToPlane[plane_code - 1];
        const int dist = (code >> 4) * xsize + (8 - (code & 0x0f));
        return dist >= 1 ? dist : 1;
    }
}

/* --- Inverse transforms --- */

static uint32_t vp8l_add_pixels(uint32_t a, uint32_t b) {
    const uint32_t ag = (a & 0xff00ff00u) + (b & 0xff00ff00u);
    const uint32_t rb = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
    return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

static uint32_t vp8l_average2(uint32_t a, uint32_t b) {
    return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b);
}

static uint32_t vp8l_clip255(uint32_t a) { return a < 256 ? a : (~a >> 24); }

static uint32_t vp8l_add_sub_full(uint32_t c0, uint32_t c1, uint32_t c2) {
    const uint32_t a = vp8l_clip255((c0 >> 24) + (c1 >> 24) - (c2 >> 24));
    const uint32_t r = vp8l_clip255(((c0 >> 16) & 0xff) + ((c1 >> 16) & 0xff) -
                                    ((c2 >> 16) & 0xff));
    const uint32_t g = vp8l_clip255(((c0 >> 8) & 0xff) + ((c1 >> 8) & 0xff) -
                                    ((c2 >> 8) & 0xff));
    const uint32_t b = vp8l_clip255((c0 & 0xff) + (c1 & 0xff) - (c2 & 0xff));
    return (a << 24) | (r << 16) | (g << 8) | b;
}

static uint32_t vp8l_add_sub_half(uint32_t c0, uint32_t c1, uint32_t c2) {
    const uint32_t ave = vp8l_average2(c0, c1);
#define WEBP_HALF(shift)                                                      \
    vp8l_clip255((uint32_t)((int)((ave >> (shift)) & 0xff) +                  \
                            (((int)((ave >> (shift)) & 0xff) -                \
                              (int)((c2 >> (shift)) & 0xff)) / 2)))
    const uint32_t a = WEBP_HALF(24);
    const uint32_t r = WEBP_HALF(16);
    const uint32_t g = WEBP_HALF(8);
    const uint32_t b = WEBP_HALF(0);
#undef WEBP_HALF
    return (a << 24) | (r << 16) | (g << 8) | b;
}

static int vp8l_sub3(int a, int b, int c) {
    return webp_abs(b - c) - webp_abs(a - c);
}

static uint32_t vp8l_select(uint32_t a, uint32_t b, uint32_t c) {
    const int diff = vp8l_sub3((int)(a >> 24), (int)(b >> 24), (int)(c >> 24)) +
                     vp8l_sub3((int)((a >> 16) & 0xff), (int)((b >> 16) & 0xff),
                               (int)((c >> 16) & 0xff)) +
                     vp8l_sub3((int)((a >> 8) & 0xff), (int)((b >> 8) & 0xff),
                               (int)((c >> 8) & 0xff)) +
                     vp8l_sub3((int)(a & 0xff), (int)(b & 0xff), (int)(c & 0xff));
    return (diff <= 0) ? a : b;
}

/* 'top' points at the pixel directly above, so top[-1] and top[1] are the
 * upper-left and upper-right neighbours. */
static uint32_t vp8l_predict(int mode, uint32_t left, const uint32_t *top) {
    switch (mode) {
        case 0: return VP8L_ARGB_BLACK;
        case 1: return left;
        case 2: return top[0];
        case 3: return top[1];
        case 4: return top[-1];
        case 5: return vp8l_average2(vp8l_average2(left, top[1]), top[0]);
        case 6: return vp8l_average2(left, top[-1]);
        case 7: return vp8l_average2(left, top[0]);
        case 8: return vp8l_average2(top[-1], top[0]);
        case 9: return vp8l_average2(top[0], top[1]);
        case 10: return vp8l_average2(vp8l_average2(left, top[-1]),
                                      vp8l_average2(top[0], top[1]));
        case 11: return vp8l_select(top[0], left, top[-1]);
        case 12: return vp8l_add_sub_full(left, top[0], top[-1]);
        default: return vp8l_add_sub_half(left, top[0], top[-1]);
    }
}

static void vp8l_inverse_predictor(const vp8l_transform *t, const uint32_t *in,
                                   uint32_t *out) {
    const int width = t->xsize, height = t->ysize;
    const int tiles_per_row = vp8l_sub_size(width, t->bits);
    int x, y;

    /* The first pixel is predicted from black, the rest of the first row
     * from the pixel on its left. */
    out[0] = vp8l_add_pixels(in[0], VP8L_ARGB_BLACK);
    for (x = 1; x < width; x++) out[x] = vp8l_add_pixels(in[x], out[x - 1]);

    for (y = 1; y < height; y++) {
        const uint32_t *modes = t->data + (y >> t->bits) * tiles_per_row;
        const size_t row = (size_t)y * width;
        /* The first pixel of every other row is predicted from above. */
        out[row] = vp8l_add_pixels(in[row], out[row - width]);
        for (x = 1; x < width; x++) {
            const int mode = (int)((modes[x >> t->bits] >> 8) & 0x0f);
            const uint32_t pred =
                vp8l_predict(mode, out[row + x - 1], &out[row + x - width]);
            out[row + x] = vp8l_add_pixels(in[row + x], pred);
        }
    }
}

static void vp8l_inverse_cross_color(const vp8l_transform *t,
                                     const uint32_t *in, uint32_t *out) {
    const int width = t->xsize, height = t->ysize;
    const int tiles_per_row = vp8l_sub_size(width, t->bits);
    int x, y;
    for (y = 0; y < height; y++) {
        const uint32_t *codes = t->data + (y >> t->bits) * tiles_per_row;
        const size_t row = (size_t)y * width;
        for (x = 0; x < width; x++) {
            const uint32_t code = codes[x >> t->bits];
            const int8_t green_to_red = (int8_t)(code & 0xff);
            const int8_t green_to_blue = (int8_t)((code >> 8) & 0xff);
            const int8_t red_to_blue = (int8_t)((code >> 16) & 0xff);
            const uint32_t argb = in[row + x];
            const int8_t green = (int8_t)(argb >> 8);
            int red = (int)((argb >> 16) & 0xff);
            int blue = (int)(argb & 0xff);
            red += ((int)green_to_red * green) >> 5;
            red &= 0xff;
            blue += ((int)green_to_blue * green) >> 5;
            blue += ((int)red_to_blue * (int8_t)red) >> 5;
            blue &= 0xff;
            out[row + x] = (argb & 0xff00ff00u) | ((uint32_t)red << 16) |
                           (uint32_t)blue;
        }
    }
}

static void vp8l_inverse_subtract_green(const vp8l_transform *t,
                                        const uint32_t *in, uint32_t *out) {
    const size_t n = (size_t)t->xsize * t->ysize;
    size_t i;
    for (i = 0; i < n; i++) {
        const uint32_t argb = in[i];
        const uint32_t green = (argb >> 8) & 0xff;
        uint32_t rb = argb & 0x00ff00ffu;
        rb += (green << 16) | green;
        rb &= 0x00ff00ffu;
        out[i] = (argb & 0xff00ff00u) | rb;
    }
}

static void vp8l_inverse_color_index(const vp8l_transform *t,
                                     const uint32_t *in, uint32_t *out) {
    const int width = t->xsize, height = t->ysize;
    const int bits_per_pixel = 8 >> t->bits;
    const int in_width = vp8l_sub_size(width, t->bits);
    int x, y;
    if (bits_per_pixel < 8) {
        const int pixels_per_byte = 1 << t->bits;
        const int count_mask = pixels_per_byte - 1;
        const uint32_t bit_mask = (1u << bits_per_pixel) - 1;
        for (y = 0; y < height; y++) {
            uint32_t packed = 0;
            for (x = 0; x < width; x++) {
                if ((x & count_mask) == 0)
                    packed = (in[(size_t)y * in_width + (x >> t->bits)] >> 8) & 0xff;
                out[(size_t)y * width + x] = t->data[packed & bit_mask];
                packed >>= bits_per_pixel;
            }
        }
    } else {
        for (y = 0; y < height; y++)
            for (x = 0; x < width; x++)
                out[(size_t)y * width + x] =
                    t->data[(in[(size_t)y * width + x] >> 8) & 0xff];
    }
}

/* The palette is stored as deltas between successive entries. */
static int vp8l_expand_color_map(vp8l_transform *t, int num_colors) {
    const int final_colors = 1 << (8 >> t->bits);
    uint32_t *map = (uint32_t *)calloc((size_t)final_colors, sizeof(uint32_t));
    uint8_t *dst, *src;
    int i;
    if (!map) return 0;
    dst = (uint8_t *)map;
    src = (uint8_t *)t->data;
    map[0] = t->data[0];
    for (i = 4; i < 4 * num_colors; i++) dst[i] = (uint8_t)(src[i] + dst[i - 4]);
    free(t->data);
    t->data = map;
    return 1;
}

static int vp8l_read_transform(vp8l_dec *d, int *xsize, int ysize) {
    vp8l_bits *br = &d->br;
    vp8l_transform *t = &d->transforms[d->num_transforms];
    const int type = (int)vp8l_read(br, 2);

    if (d->transforms_seen & (1u << type)) {
        d->error = "duplicate VP8L transform";
        return 0;
    }
    d->transforms_seen |= 1u << type;
    t->type = type;
    t->xsize = *xsize;
    t->ysize = ysize;
    t->data = NULL;
    t->bits = 0;
    d->num_transforms++;

    switch (type) {
        case VP8L_PREDICTOR_TRANSFORM:
        case VP8L_CROSS_COLOR_TRANSFORM:
            t->bits = 2 + (int)vp8l_read(br, 3);
            return vp8l_decode_image(d, vp8l_sub_size(t->xsize, t->bits),
                                     vp8l_sub_size(t->ysize, t->bits), 0,
                                     &t->data);
        case VP8L_COLOR_INDEXING_TRANSFORM: {
            const int num_colors = (int)vp8l_read(br, 8) + 1;
            t->bits = (num_colors > 16) ? 0
                    : (num_colors > 4)  ? 1
                    : (num_colors > 2)  ? 2
                                        : 3;
            *xsize = vp8l_sub_size(t->xsize, t->bits);
            if (!vp8l_decode_image(d, num_colors, 1, 0, &t->data)) return 0;
            if (!vp8l_expand_color_map(t, num_colors)) {
                d->error = "out of memory decoding VP8L image";
                return 0;
            }
            return 1;
        }
        default:                        /* VP8L_SUBTRACT_GREEN */
            return 1;
    }
}

/*
 * Decodes one entropy coded image. At level 0 this is the image itself,
 * preceded by the transform descriptions; the recursive calls decode the
 * transform and meta prefix code images.
 */
static int vp8l_decode_image(vp8l_dec *d, int xsize, int ysize, int is_level0,
                             uint32_t **out) {
    vp8l_bits *br = &d->br;
    int width = xsize;
    vp8l_group *groups = NULL;
    uint32_t *huff_image = NULL;
    uint32_t *data = NULL;
    vp8l_cache cache;
    int num_groups = 0, huff_bits = 0, huff_xsize = 0;
    int cache_bits = 0;
    size_t npix, pos = 0;
    int x = 0, y = 0;
    int ok = 0;
    const vp8l_group *group;

    cache.colors = NULL;
    cache.bits = 0;

    if (is_level0) {
        while (vp8l_read(br, 1)) {
            if (d->num_transforms == VP8L_MAX_TRANSFORMS) {
                d->error = "too many VP8L transforms";
                return 0;
            }
            if (!vp8l_read_transform(d, &width, ysize)) return 0;
        }
    }

    if (vp8l_read(br, 1)) {
        cache_bits = (int)vp8l_read(br, 4);
        if (cache_bits < 1 || cache_bits > VP8L_MAX_CACHE_BITS) {
            d->error = "invalid VP8L color cache size";
            return 0;
        }
    }

    if (!vp8l_read_huffman_codes(d, width, ysize, cache_bits, is_level0,
                                 &groups, &num_groups, &huff_image, &huff_bits,
                                 &huff_xsize))
        return 0;

    if (cache_bits > 0) {
        cache.bits = cache_bits;
        cache.colors = (uint32_t *)calloc((size_t)1 << cache_bits, sizeof(uint32_t));
        if (!cache.colors) {
            d->error = "out of memory decoding VP8L image";
            goto done;
        }
    }

    npix = (size_t)width * (size_t)ysize;
    data = (uint32_t *)malloc(npix * sizeof(uint32_t));
    if (!data) {
        d->error = "out of memory decoding VP8L image";
        goto done;
    }

    group = &groups[0];
    while (pos < npix) {
        int code;
        if (huff_image) {
            const int index = (int)huff_image[(size_t)(y >> huff_bits) * huff_xsize +
                                              (x >> huff_bits)];
            if (index >= num_groups) {
                d->error = "corrupt VP8L meta prefix code index";
                goto done;
            }
            group = &groups[index];
        }
        code = vp8l_huff_read(&group->trees[0], br);
        if (code < 0 || br->eos) {
            d->error = "truncated VP8L bitstream";
            goto done;
        }
        if (code < VP8L_LITERAL_CODES) {            /* literal */
            const int red = vp8l_huff_read(&group->trees[1], br);
            const int blue = vp8l_huff_read(&group->trees[2], br);
            const int alpha = vp8l_huff_read(&group->trees[3], br);
            if (red < 0 || blue < 0 || alpha < 0) {
                d->error = "corrupt VP8L literal";
                goto done;
            }
            data[pos] = ((uint32_t)alpha << 24) | ((uint32_t)red << 16) |
                        ((uint32_t)code << 8) | (uint32_t)blue;
            if (cache.colors) cache.colors[vp8l_cache_key(data[pos], cache.bits)] = data[pos];
            pos++;
            if (++x == width) { x = 0; y++; }
        } else if (code < VP8L_LITERAL_CODES + VP8L_LENGTH_CODES) {
            /* backward reference */
            const int length = vp8l_copy_distance(br, code - VP8L_LITERAL_CODES);
            const int dist_symbol = vp8l_huff_read(&group->trees[4], br);
            int dist;
            size_t i;
            if (dist_symbol < 0) {
                d->error = "corrupt VP8L distance code";
                goto done;
            }
            dist = vp8l_plane_to_distance(width, vp8l_copy_distance(br, dist_symbol));
            if ((size_t)dist > pos || (size_t)length > npix - pos) {
                d->error = "corrupt VP8L backward reference";
                goto done;
            }
            for (i = 0; i < (size_t)length; i++) {
                data[pos] = data[pos - (size_t)dist];
                if (cache.colors)
                    cache.colors[vp8l_cache_key(data[pos], cache.bits)] = data[pos];
                pos++;
                if (++x == width) { x = 0; y++; }
            }
        } else {                                    /* color cache lookup */
            const int key = code - VP8L_LITERAL_CODES - VP8L_LENGTH_CODES;
            if (!cache.colors || key >= (1 << cache.bits)) {
                d->error = "corrupt VP8L color cache index";
                goto done;
            }
            data[pos] = cache.colors[key];
            pos++;
            if (++x == width) { x = 0; y++; }
        }
    }
    ok = 1;

done:
    vp8l_groups_free(groups, num_groups);
    free(huff_image);
    free(cache.colors);
    if (!ok) {
        free(data);
        return 0;
    }
    *out = data;
    return 1;
}

static webp_image *webp_decode_vp8l(const uint8_t *data, size_t len,
                                    char *err, size_t err_cap) {
    vp8l_dec d;
    webp_image *img = NULL;
    uint32_t *pixels = NULL;
    uint32_t w, h;
    int i, channels;

    memset(&d, 0, sizeof(d));
    vp8l_bits_init(&d.br, data, len);

    if (len < 5 || vp8l_read(&d.br, 8) != VP8L_MAGIC) {
        webp_err_set(err, err_cap, "bad VP8L signature");
        return NULL;
    }
    w = vp8l_read(&d.br, 14) + 1;
    h = vp8l_read(&d.br, 14) + 1;
    d.has_alpha = (int)vp8l_read(&d.br, 1);
    if (vp8l_read(&d.br, 3) != 0) {
        webp_err_set(err, err_cap, "unsupported VP8L version");
        return NULL;
    }
    if (!webp_dims_ok(w, h)) {
        webp_err_set(err, err_cap, "unsupported VP8L image dimensions");
        return NULL;
    }
    d.width = (int)w;
    d.height = (int)h;

    if (!vp8l_decode_image(&d, d.width, d.height, 1, &pixels)) {
        webp_err_set(err, err_cap,
                     d.error ? d.error : "invalid VP8L image data");
        goto done;
    }

    /* Transforms are undone in the reverse of the order they were read. */
    for (i = d.num_transforms - 1; i >= 0; i--) {
        const vp8l_transform *t = &d.transforms[i];
        const size_t out_pix = (size_t)t->xsize * (size_t)t->ysize;
        uint32_t *out = (uint32_t *)malloc(out_pix * sizeof(uint32_t));
        if (!out) {
            webp_err_set(err, err_cap, "out of memory decoding VP8L image");
            goto done;
        }
        switch (t->type) {
            case VP8L_PREDICTOR_TRANSFORM:
                vp8l_inverse_predictor(t, pixels, out);
                break;
            case VP8L_CROSS_COLOR_TRANSFORM:
                vp8l_inverse_cross_color(t, pixels, out);
                break;
            case VP8L_SUBTRACT_GREEN:
                vp8l_inverse_subtract_green(t, pixels, out);
                break;
            default:
                vp8l_inverse_color_index(t, pixels, out);
                break;
        }
        free(pixels);
        pixels = out;
    }

    channels = d.has_alpha ? 4 : 3;
    img = webp_image_alloc(d.width, d.height, channels);
    if (!img) {
        webp_err_set(err, err_cap, "out of memory decoding VP8L image");
        goto done;
    }
    {
        const size_t npix = (size_t)d.width * (size_t)d.height;
        size_t p;
        uint8_t *dst = img->data;
        for (p = 0; p < npix; p++) {
            const uint32_t argb = pixels[p];
            *dst++ = (uint8_t)((argb >> 16) & 0xff);
            *dst++ = (uint8_t)((argb >> 8) & 0xff);
            *dst++ = (uint8_t)(argb & 0xff);
            if (channels == 4) *dst++ = (uint8_t)(argb >> 24);
        }
    }

done:
    for (i = 0; i < d.num_transforms; i++) free(d.transforms[i].data);
    free(pixels);
    return img;
}

webp_image *webp_load_mem_err(const uint8_t *data, size_t len,
                              char *err, size_t err_cap) {
    size_t off = 12;
    size_t riff_end;

    if (!webp_is_webp(data, len)) {
        webp_err_set(err, err_cap, "not a RIFF/WEBP file");
        return NULL;
    }
    if (len > WEBP_MAX_INPUT_BYTES) {
        webp_err_set(err, err_cap, "WebP image exceeds the input size limit");
        return NULL;
    }
    riff_end = (size_t)webp_le32(data + 4) + 8;
    if (riff_end > len || riff_end < 12) riff_end = len;

    while (off + 8 <= riff_end) {
        const uint8_t *id = data + off;
        size_t size = webp_le32(data + off + 4);
        size_t body = off + 8;
        if (size > riff_end - body) size = riff_end - body;
        if (!memcmp(id, "VP8 ", 4))
            return webp_decode_vp8(data + body, size, err, err_cap);
        if (!memcmp(id, "VP8L", 4))
            return webp_decode_vp8l(data + body, size, err, err_cap);
        if (!memcmp(id, "VP8X", 4)) {
            if (size >= 1 && (data[body] & 0x02)) {
                webp_err_set(err, err_cap,
                             "animated WebP is not supported; send a still image");
                return NULL;
            }
        } else if (!memcmp(id, "ANIM", 4) || !memcmp(id, "ANMF", 4)) {
            webp_err_set(err, err_cap,
                         "animated WebP is not supported; send a still image");
            return NULL;
        }
        /* ICCP, EXIF, XMP and ALPH are ignored. */
        off = body + size + (size & 1);
    }
    webp_err_set(err, err_cap, "WebP file contains no image data");
    return NULL;
}

webp_image *webp_load_mem(const uint8_t *data, size_t len) {
    return webp_load_mem_err(data, len, NULL, 0);
}

webp_image *webp_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    uint8_t *buf;
    long end;
    size_t got;
    webp_image *img;

    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    end = ftell(fp);
    if (end <= 0 || (unsigned long)end > WEBP_MAX_INPUT_BYTES) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    buf = (uint8_t *)malloc((size_t)end);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    got = fread(buf, 1, (size_t)end, fp);
    fclose(fp);
    if (got != (size_t)end) {
        free(buf);
        return NULL;
    }
    img = webp_load_mem(buf, got);
    free(buf);
    return img;
}

void webp_free(webp_image *img) {
    if (!img) return;
    free(img->data);
    free(img);
}

#endif /* WEBP_IMPLEMENTATION */

#endif /* WEBP_H */
