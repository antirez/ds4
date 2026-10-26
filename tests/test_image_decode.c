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

/* Load a JPEG, set Ss/Se of its scan-th SOS header to ss/se and require the
 * decoder to refuse it. orig_ss/orig_se pin the header the test expects. */
static int check_scan_range_rejected(const char *path, int scan, int orig_ss,
                                     int orig_se, int ss, int se) {
    static uint8_t buf[4096];
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    size_t n = fread(buf, 1, sizeof(buf), f);
    int whole = feof(f);
    fclose(f);
    size_t pos = 2, at = 0;
    int seen = 0;
    while (whole && pos + 4 < n && buf[pos] == 0xFF && buf[pos + 1] != 0xD9) {
        size_t len = ((size_t)buf[pos + 2] << 8) | buf[pos + 3];
        if (buf[pos + 1] == 0xDA) {
            if (seen++ == scan) {
                at = pos + 5 + 2 * (size_t)buf[pos + 4];
                break;
            }
            /* skip the entropy-coded data up to the next marker */
            pos += 2 + len;
            while (pos + 1 < n && !(buf[pos] == 0xFF && buf[pos + 1] != 0x00 &&
                                    (buf[pos + 1] < 0xD0 || buf[pos + 1] > 0xD7)))
                pos++;
            continue;
        }
        pos += 2 + len;
    }
    if (!at || at + 1 >= n || buf[at] != orig_ss || buf[at + 1] != orig_se) {
        fprintf(stderr, "%s: scan %d header not found as Ss=%d Se=%d\n",
                path, scan, orig_ss, orig_se);
        return 0;
    }
    buf[at] = (uint8_t)ss;
    buf[at + 1] = (uint8_t)se;
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, buf, n, error, sizeof(error))) {
        fprintf(stderr, "%s: scan %d with Ss=%d Se=%d decoded, expected a rejection\n",
                path, scan, ss, se);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

int main(void) {
    /* A progressive scan ends its band at Se <= 63 (T.81 Table B.3): the
     * decoder walks jpeg_zigzag[64] up to Se. Scan 1 of this file is its
     * first AC scan, Ss=1 Se=63; Se=64 must be refused. */
    if (!check_scan_range_rejected("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                                   1, 1, 63, 1, 64))
        return 1;
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
    return 0;
}
