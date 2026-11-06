#include "ds4_image.h"

#include <stdio.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_jpeg(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    char got[65] = {0};
    int ok = hex_fingerprint(image.fingerprint, got, sizeof(got)) &&
             image.width == width && image.height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                path, image.width, image.height, got,
                width, height, expected_fp);
    }
    ds4_image_free(&image);
    return ok;
}

static int check_jpeg_rejected(const char *path) {
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "%s: malformed JPEG was accepted\n", path);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

/* 1x1 progressive grayscale JPEG with a single scan whose SOS carries the
 * given Ss/Se. Used for scan headers that must be rejected. */
static int check_jpeg_sos_rejected(uint8_t ss, uint8_t se) {
    uint8_t b[128];
    size_t n = 0;
    static const uint8_t sof2[] = {0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x01,
                                   0x00, 0x01, 0x01, 0x01, 0x11, 0x00};
    b[n++] = 0xFF; b[n++] = 0xD8;
    b[n++] = 0xFF; b[n++] = 0xDB; b[n++] = 0x00; b[n++] = 0x43; b[n++] = 0x00;
    for (int i = 0; i < 64; i++) b[n++] = 0x01;
    memcpy(b + n, sof2, sizeof(sof2)); n += sizeof(sof2);
    /* DC table 0: one 1-bit code for category 0. */
    b[n++] = 0xFF; b[n++] = 0xC4; b[n++] = 0x00; b[n++] = 0x14; b[n++] = 0x00;
    b[n++] = 0x01;
    for (int i = 1; i < 16; i++) b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xDA; b[n++] = 0x00; b[n++] = 0x08; b[n++] = 0x01;
    b[n++] = 0x01; b[n++] = 0x00; b[n++] = ss; b[n++] = se; b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xD9;

    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, b, n, error, sizeof(error))) {
        fprintf(stderr, "SOS Ss=%u Se=%u was accepted\n", ss, se);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

int main(void) {
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    /* Progressive AC scan with Se=200 (issue #1292). Unpatched Iris indexes
     * jpeg_zigzag[] past 63 and writes outside the coefficient block. */
    if (!check_jpeg_rejected("tests/vision-fixtures/jpeg/prog_sos_se_oob.jpg"))
        return 1;
    /* DC scans must have Se == 0. */
    if (!check_jpeg_sos_rejected(0, 5)) return 1;
    return 0;
}
