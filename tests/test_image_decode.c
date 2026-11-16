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

/* Appends a DHT segment for table class tc, id 0, with the 16 code-length
 * counts in bits and the given symbols. */
static size_t put_dht(uint8_t *b, size_t n, int tc, const uint8_t bits[16],
                      const uint8_t *symbols, size_t count) {
    size_t len = 2 + 1 + 16 + count;
    b[n++] = 0xFF; b[n++] = 0xC4;
    b[n++] = (uint8_t)(len >> 8); b[n++] = (uint8_t)len;
    b[n++] = (uint8_t)(tc << 4);
    memcpy(b + n, bits, 16); n += 16;
    memcpy(b + n, symbols, count); n += count;
    return n;
}

/* 1x1 baseline grayscale JPEG built in memory. quant is the value of every
 * quantizer entry, stored with 16-bit precision. scan holds the entropy-coded
 * bits as a '0'/'1' string, padded with 1s and byte-stuffed here. */
static size_t build_baseline_gray(uint8_t *b, uint16_t quant,
                                  const uint8_t dc_bits[16], const uint8_t *dc_sym,
                                  size_t dc_count, const uint8_t ac_bits[16],
                                  const uint8_t *ac_sym, size_t ac_count,
                                  const char *scan) {
    static const uint8_t sof0[] = {0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
                                   0x00, 0x01, 0x01, 0x01, 0x11, 0x00};
    static const uint8_t sos[] = {0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00,
                                  0x00, 0x3F, 0x00};
    size_t n = 0;
    b[n++] = 0xFF; b[n++] = 0xD8;
    b[n++] = 0xFF; b[n++] = 0xDB; b[n++] = 0x00; b[n++] = 0x83; b[n++] = 0x10;
    for (int i = 0; i < 64; i++) {
        b[n++] = (uint8_t)(quant >> 8);
        b[n++] = (uint8_t)quant;
    }
    memcpy(b + n, sof0, sizeof(sof0)); n += sizeof(sof0);
    n = put_dht(b, n, 0, dc_bits, dc_sym, dc_count);
    if (ac_count) n = put_dht(b, n, 1, ac_bits, ac_sym, ac_count);
    memcpy(b + n, sos, sizeof(sos)); n += sizeof(sos);
    size_t nbits = strlen(scan);
    for (size_t i = 0; i < nbits; i += 8) {
        uint8_t byte = 0;
        for (size_t j = 0; j < 8; j++)
            byte = (uint8_t)(byte << 1 | (i + j < nbits ? scan[i + j] == '1' : 1));
        b[n++] = byte;
        if (byte == 0xFF) b[n++] = 0x00;
    }
    b[n++] = 0xFF; b[n++] = 0xD9;
    return n;
}

static int decode_ok(const uint8_t *b, size_t n) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_memory(&image, b, n, error, sizeof(error))) return 0;
    ds4_image_free(&image);
    return 1;
}

/* Two 1-bit codes already use the whole code space, so a 9-bit code cannot
 * fit. Unpatched Iris accepts the table, and oversubscribed long codes make
 * the slow Huffman path compute a negative value index. */
static int check_huffman_oversubscribed_rejected(void) {
    uint8_t b[512];
    const uint8_t dc_bits[16] = {2, 0, 0, 0, 0, 0, 0, 0, 1};
    const uint8_t dc_sym[3] = {0, 0, 0};
    const uint8_t ac_bits[16] = {1};
    const uint8_t ac_sym[1] = {0x00};
    size_t n = build_baseline_gray(b, 1, dc_bits, dc_sym, 3, ac_bits, ac_sym, 1, "00");
    if (decode_ok(b, n)) {
        fprintf(stderr, "oversubscribed Huffman table was accepted\n");
        return 0;
    }
    return 1;
}

/* The scan names AC table 0, which the file never defines. Unpatched Iris
 * decodes with its zeroed derived tables and reads values[-1]. */
static int check_undefined_huffman_table_rejected(void) {
    uint8_t b[512];
    const uint8_t dc_bits[16] = {1};
    const uint8_t dc_sym[1] = {0};
    size_t n = build_baseline_gray(b, 1, dc_bits, dc_sym, 1, NULL, NULL, 0,
                                   "0" "000000000");
    if (decode_ok(b, n)) {
        fprintf(stderr, "scan with an undefined Huffman table was accepted\n");
        return 0;
    }
    return 1;
}

/* A DC category of 17 asks the bit reader for 17 extra bits, more than JPEG
 * ever codes; unpatched Iris shifts past the width of an int. */
static int check_dc_category_rejected(void) {
    uint8_t b[512];
    const uint8_t dc_bits[16] = {1};
    const uint8_t dc_sym[1] = {17};
    const uint8_t ac_bits[16] = {1};
    const uint8_t ac_sym[1] = {0x00};
    size_t n = build_baseline_gray(b, 1, dc_bits, dc_sym, 1, ac_bits, ac_sym, 1,
                                   "0" "11111111111111111" "0");
    if (decode_ok(b, n)) {
        fprintf(stderr, "DC category 17 was accepted\n");
        return 0;
    }
    return 1;
}

/* Largest 15-bit DC and AC values with 16-bit quantizers. The decode must
 * succeed; under UBSan, unpatched Iris overflows int in the IDCT. */
static int check_extreme_coefficients_decode(void) {
    uint8_t b[512];
    const uint8_t dc_bits[16] = {1};
    const uint8_t dc_sym[1] = {15};
    const uint8_t ac_bits[16] = {2};
    const uint8_t ac_sym[2] = {0x0F, 0x00};
    size_t n = build_baseline_gray(b, 65535, dc_bits, dc_sym, 1, ac_bits, ac_sym, 2,
                                   "0" "111111111111111" "0" "111111111111111" "1");
    if (!decode_ok(b, n)) {
        fprintf(stderr, "JPEG with extreme coefficients was rejected\n");
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
    if (!check_huffman_oversubscribed_rejected()) return 1;
    if (!check_undefined_huffman_table_rejected()) return 1;
    if (!check_dc_category_rejected()) return 1;
    if (!check_extreme_coefficients_decode()) return 1;
    return 0;
}
