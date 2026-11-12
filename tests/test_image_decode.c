#include "ds4_image.h"
#include "ds4_webp.h"
#include "third_party/iris/png.h"

#include <stdio.h>
#include <stdlib.h>
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

#define WEBP_FIXTURES "tests/vision-fixtures/webp/"

/* The WebP decode must give exactly the pixels of dwebp's PNG output, alpha
 * included, both from webp_load() and through ds4_image (RGB only). */
static int check_webp(const char *name, int width, int height, int channels) {
    char webp_path[256], ref_path[256];
    snprintf(webp_path, sizeof(webp_path), WEBP_FIXTURES "%s.webp", name);
    snprintf(ref_path, sizeof(ref_path), WEBP_FIXTURES "%s.ref.png", name);

    webp_image *img = webp_load(webp_path);
    png_image *ref = png_load(ref_path);
    int ok = img && ref &&
             img->width == width && img->height == height && img->channels == channels &&
             ref->width == width && ref->height == height && ref->channels == channels &&
             memcmp(img->data, ref->data, (size_t)width * height * channels) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %dx%dx%d, reference %dx%dx%d, expected %dx%dx%d\n",
                webp_path, img ? img->width : 0, img ? img->height : 0,
                img ? img->channels : 0, ref ? ref->width : 0, ref ? ref->height : 0,
                ref ? ref->channels : 0, width, height, channels);
    }
    webp_free(img);
    png_free(ref);
    if (!ok) return 0;

    ds4_image got = {0}, ref_rgb = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&got, webp_path, error, sizeof(error)) ||
        !ds4_image_decode_file(&ref_rgb, ref_path, error, sizeof(error))) {
        fprintf(stderr, "ds4_image decode failed for %s: %s\n", name, error);
        ds4_image_free(&got);
        return 0;
    }
    ok = memcmp(got.fingerprint, ref_rgb.fingerprint, sizeof(got.fingerprint)) == 0;
    if (!ok) fprintf(stderr, "%s: ds4_image RGB differs from the reference\n", webp_path);
    ds4_image_free(&got);
    ds4_image_free(&ref_rgb);
    return ok;
}

static uint8_t *read_fixture(const char *name, size_t *len) {
    char path[256];
    FILE *fp;
    uint8_t *buf;
    long end;
    snprintf(path, sizeof(path), WEBP_FIXTURES "%s.webp", name);
    fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0 || (end = ftell(fp)) <= 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    buf = malloc((size_t)end);
    if (buf && fread(buf, 1, (size_t)end, fp) != (size_t)end) {
        free(buf);
        buf = NULL;
    }
    fclose(fp);
    *len = (size_t)end;
    return buf;
}

static void put_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static int decode_rejected(const uint8_t *buf, size_t len, const char *what,
                           const char *expect_error) {
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, buf, len, error, sizeof(error))) {
        fprintf(stderr, "%s was accepted\n", what);
        ds4_image_free(&image);
        return 0;
    }
    if (expect_error && !strstr(error, expect_error)) {
        fprintf(stderr, "%s: unexpected error \"%s\"\n", what, error);
        return 0;
    }
    return 1;
}

/* Cuts the bitstream short, keeping the RIFF and chunk sizes consistent so
 * the decoder itself has to notice. Lossy streams end with up to two bytes
 * of padding, so cuts of one or two bytes may still decode. */
static int check_webp_truncations(const char *name) {
    size_t len = 0, image_at = 12, cut, cuts = 0;
    uint8_t *buf = read_fixture(name, &len);
    uint8_t *copy;
    int ok = 1;
    if (!buf) {
        fprintf(stderr, "cannot read %s.webp\n", name);
        return 0;
    }
    while (image_at + 8 <= len && memcmp(buf + image_at, "VP8 ", 4) != 0 &&
           memcmp(buf + image_at, "VP8L", 4) != 0) {
        const uint32_t size = buf[image_at + 4] | (uint32_t)buf[image_at + 5] << 8 |
                              (uint32_t)buf[image_at + 6] << 16 | (uint32_t)buf[image_at + 7] << 24;
        image_at += 8 + (size_t)size + (size & 1);
    }
    if (image_at + 8 > len) {
        fprintf(stderr, "%s.webp has no VP8 or VP8L chunk\n", name);
        free(buf);
        return 0;
    }
    copy = malloc(len);
    if (!copy) {
        fprintf(stderr, "out of memory copying %s.webp\n", name);
        free(buf);
        return 0;
    }
    for (cut = image_at + 8; ok && cut + 3 <= len; cut++, cuts++) {
        char what[96];
        memcpy(copy, buf, cut);
        put_le32(copy + 4, (uint32_t)(cut - 8));
        put_le32(copy + image_at + 4, (uint32_t)(cut - image_at - 8));
        snprintf(what, sizeof(what), "%s.webp cut to %zu of %zu bytes", name, cut, len);
        ok = decode_rejected(copy, cut, what, "WebP");
    }
    if (ok && cuts == 0) {
        fprintf(stderr, "%s.webp is too short to cut\n", name);
        ok = 0;
    }
    free(copy);
    free(buf);
    return ok;
}

static int check_webp_container_errors(void) {
    size_t len = 0;
    uint8_t *buf = read_fixture("lossy_alpha", &len);
    int ok = 1;
    if (!buf || len < 30 || memcmp(buf + 12, "VP8X", 4) != 0) {
        fprintf(stderr, "lossy_alpha.webp is not a VP8X file\n");
        free(buf);
        return 0;
    }
    /* Animation flag set: animated files are refused, not half-decoded. */
    buf[20] |= 0x02;
    ok &= decode_rejected(buf, len, "animated VP8X", "animated");
    buf[20] &= (uint8_t)~0x02;
    /* Canvas width that disagrees with the bitstream. */
    buf[24] ^= 1;
    ok &= decode_rejected(buf, len, "VP8X canvas mismatch", "invalid or unsupported WebP");
    buf[24] ^= 1;
    /* ALPH header with the reserved bits set. */
    if (memcmp(buf + 30, "ALPH", 4) != 0) {
        fprintf(stderr, "lossy_alpha.webp has no ALPH chunk after VP8X\n");
        ok = 0;
    } else {
        buf[38] |= 0xc0;
        ok &= decode_rejected(buf, len, "ALPH with reserved bits", "invalid or unsupported WebP");
        buf[38] &= 0x3f;
    }
    /* RIFF size larger than the file. */
    put_le32(buf + 4, (uint32_t)len);
    ok &= decode_rejected(buf, len, "oversized RIFF", "invalid or unsupported WebP");
    free(buf);
    return ok;
}

/* A still image may carry an ANIM chunk with the animation flag off. RFC 9649
 * says it MUST be ignored, and libwebp decodes such files normally. */
static int check_webp_still_with_anim(void) {
    size_t len = 0;
    uint8_t *buf = read_fixture("lossy_alpha", &len);
    if (!buf || len < 30 || memcmp(buf + 12, "VP8X", 4) != 0 || (buf[20] & 0x02)) {
        fprintf(stderr, "lossy_alpha.webp is not a still VP8X file\n");
        free(buf);
        return 0;
    }
    static const uint8_t anim[14] = { 'A', 'N', 'I', 'M', 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const size_t out_len = len + sizeof(anim);
    uint8_t *out = malloc(out_len);
    int ok = 0;
    if (out) {
        memcpy(out, buf, 30);                       /* RIFF header + VP8X */
        memcpy(out + 30, anim, sizeof(anim));
        memcpy(out + 30 + sizeof(anim), buf + 30, len - 30);
        put_le32(out + 4, (uint32_t)(out_len - 8));
        webp_image *img = webp_load_mem(out, out_len);
        png_image *ref = png_load(WEBP_FIXTURES "lossy_alpha.ref.png");
        ok = img && ref && !webp_is_animated(out, out_len) &&
             img->width == ref->width && img->height == ref->height &&
             img->channels == ref->channels &&
             memcmp(img->data, ref->data, (size_t)img->width * img->height * img->channels) == 0;
        if (!ok) fprintf(stderr, "still image with an ANIM chunk was not decoded like dwebp\n");
        webp_free(img);
        png_free(ref);
    }
    free(out);
    free(buf);
    return ok;
}

int main(void) {
    /* WebP, lossy (VP8) and lossless (VP8L), compared with dwebp output:
     * normal and simple loop filter with segments, VP8X with lossless and
     * raw ALPH chunks under all four alpha filters, all four lossless
     * transforms (predictor, cross-color, subtract-green, a bundled palette)
     * and lossless alpha with -exact. */
    if (!check_webp("lossy", 61, 37, 3)) return 1;
    if (!check_webp("lossy_simple_filter", 61, 37, 3)) return 1;
    if (!check_webp("lossy_alpha", 33, 21, 4)) return 1;
    if (!check_webp("lossy_alpha_raw", 33, 21, 4)) return 1;
    if (!check_webp("lossy_alpha_vertical", 33, 21, 4)) return 1;
    if (!check_webp("lossy_alpha_gradient", 33, 21, 4)) return 1;
    if (!check_webp("lossless", 61, 37, 3)) return 1;
    if (!check_webp("lossless_subtract_green", 61, 37, 3)) return 1;
    if (!check_webp("lossless_palette", 23, 11, 3)) return 1;
    if (!check_webp("lossless_alpha", 33, 21, 4)) return 1;
    if (!check_webp_truncations("lossy")) return 1;
    if (!check_webp_truncations("lossy_alpha")) return 1;
    if (!check_webp_truncations("lossless")) return 1;
    if (!check_webp_truncations("lossless_palette")) return 1;
    if (!check_webp_container_errors()) return 1;
    if (!check_webp_still_with_anim()) return 1;

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
