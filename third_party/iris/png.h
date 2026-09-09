/*
 * PNG Decoder/Encoder - Single-header implementation
 *
 * A dependency-free C implementation for reading and writing PNG images.
 * Uses zlib-style deflate compression (store mode for writing, full inflate for reading).
 *
 * Usage:
 *   png_image *img = png_load("image.png");
 *   if (!img) { handle error }
 *
 *   // Access pixel data
 *   uint8_t *pixel = img->data + (y * img->width + x) * img->channels;
 *
 *   png_save(img, "output.png");
 *   png_free(img);
 *
 * To use as header-only, define PNG_IMPLEMENTATION before including:
 *   #define PNG_IMPLEMENTATION
 *   #include "png.h"
 */

#ifndef PNG_H
#define PNG_H

#include <stddef.h>
#include <stdint.h>

/* DS4 imports IRIS at commit 9873887d4aa0646c650adc5b86b986d2f653b7e0.
 * These limits make the memory decoder suitable for untrusted server input. */
#ifndef PNG_MAX_INPUT_BYTES
#define PNG_MAX_INPUT_BYTES (64u * 1024u * 1024u)
#endif
#ifndef PNG_MAX_DIMENSION
#define PNG_MAX_DIMENSION 16384
#endif
#ifndef PNG_MAX_PIXELS
#define PNG_MAX_PIXELS (64u * 1024u * 1024u)
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
    int channels;       /* 1=Grayscale, 2=Gray+Alpha, 3=RGB, 4=RGBA */
    uint8_t *data;      /* Row-major, channel-interleaved */
} png_image;

/* ========================================================================
 * Public API
 * ======================================================================== */

/*
 * Load PNG image from file.
 * Returns NULL on error.
 */
png_image *png_load(const char *path);

/*
 * Load PNG image from memory buffer.
 * Returns NULL on error.
 */
png_image *png_load_mem(const uint8_t *data, size_t len);

/*
 * Save PNG image to file.
 * Returns 0 on success, -1 on error.
 */
int png_save(const png_image *img, const char *path);

/*
 * Save PNG image with text metadata.
 * keyword: up to 79 characters, text: arbitrary length.
 * Returns 0 on success, -1 on error.
 */
int png_save_with_text(const png_image *img, const char *path,
                       const char *keyword, const char *text);

/*
 * Create a new image with given dimensions.
 * Allocates zeroed pixel data.
 */
png_image *png_create(int width, int height, int channels);

/*
 * Free image and pixel data.
 */
void png_free(png_image *img);

/*
 * Clone an image (deep copy).
 */
png_image *png_clone(const png_image *img);

#ifdef __cplusplus
}
#endif

#endif /* PNG_H */

/* ========================================================================
 * Implementation
 * ======================================================================== */

#ifdef PNG_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Image Creation and Management
 * ======================================================================== */

png_image *png_create(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels < 1 || channels > 4) return NULL;
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / (size_t)channels) return NULL;
    size_t bytes = pixels * (size_t)channels;
    png_image *img = (png_image *)malloc(sizeof(png_image));
    if (!img) return NULL;

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t *)calloc(bytes, sizeof(uint8_t));

    if (!img->data) {
        free(img);
        return NULL;
    }

    return img;
}

void png_free(png_image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

png_image *png_clone(const png_image *img) {
    if (!img) return NULL;

    png_image *clone = png_create(img->width, img->height, img->channels);
    if (!clone) return NULL;

    size_t bytes = (size_t)img->width * (size_t)img->height *
                   (size_t)img->channels;
    memcpy(clone->data, img->data, bytes);
    return clone;
}

/* ========================================================================
 * CRC32 for PNG
 * ======================================================================== */

static uint32_t png_update_crc(uint32_t crc, const uint8_t *buf, size_t len) {
    uint32_t c = crc;
    for (size_t n = 0; n < len; n++) {
        c ^= buf[n];
        for (int bit = 0; bit < 8; bit++) {
            c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
        }
    }
    return c;
}

static uint32_t png_crc(const uint8_t *buf, size_t len) {
    return png_update_crc(0xffffffffu, buf, len) ^ 0xffffffffu;
}

/* ========================================================================
 * Adler-32 for zlib
 * ======================================================================== */

static uint32_t png_adler32(const uint8_t *data, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; i++) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

/* ========================================================================
 * Deflate Store Mode (for writing)
 * ======================================================================== */

static uint8_t *png_deflate_store(const uint8_t *data, size_t len, size_t *out_len) {
    /* Zlib header (2 bytes) + deflate blocks + adler32 (4 bytes) */
    size_t max_block = 65535;
    size_t num_blocks = (len + max_block - 1) / max_block;
    size_t total = 2 + num_blocks * 5 + len + 4;

    uint8_t *out = (uint8_t *)malloc(total);
    if (!out) return NULL;

    size_t pos = 0;

    /* Zlib header: CMF=0x78 (deflate, 32K window), FLG=0x01 (no dict, level 0) */
    out[pos++] = 0x78;
    out[pos++] = 0x01;

    /* Deflate stored blocks */
    size_t remaining = len;
    const uint8_t *src = data;
    while (remaining > 0) {
        size_t block_len = (remaining > max_block) ? max_block : remaining;
        int is_final = (remaining <= max_block) ? 1 : 0;

        /* Block header: BFINAL (1 bit) + BTYPE=00 (2 bits) = stored */
        out[pos++] = is_final;

        /* LEN and NLEN (little-endian) */
        out[pos++] = block_len & 0xff;
        out[pos++] = (block_len >> 8) & 0xff;
        out[pos++] = (~block_len) & 0xff;
        out[pos++] = ((~block_len) >> 8) & 0xff;

        memcpy(out + pos, src, block_len);
        pos += block_len;

        src += block_len;
        remaining -= block_len;
    }

    /* Adler-32 checksum (big-endian) */
    uint32_t checksum = png_adler32(data, len);
    out[pos++] = (checksum >> 24) & 0xff;
    out[pos++] = (checksum >> 16) & 0xff;
    out[pos++] = (checksum >> 8) & 0xff;
    out[pos++] = checksum & 0xff;

    *out_len = pos;
    return out;
}

/* ========================================================================
 * Chunk Writing
 * ======================================================================== */

static void png_write_chunk(FILE *f, const char *type, const uint8_t *data, size_t len) {
    /* Length (big-endian) */
    uint8_t len_bytes[4] = {
        (len >> 24) & 0xff,
        (len >> 16) & 0xff,
        (len >> 8) & 0xff,
        len & 0xff
    };
    fwrite(len_bytes, 1, 4, f);

    /* Type */
    fwrite(type, 1, 4, f);

    /* Data */
    if (len > 0 && data) {
        fwrite(data, 1, len, f);
    }

    /* CRC (over type + data) */
    uint8_t *crc_data = (uint8_t *)malloc(4 + len);
    memcpy(crc_data, type, 4);
    if (len > 0 && data) {
        memcpy(crc_data + 4, data, len);
    }
    uint32_t crc = png_crc(crc_data, 4 + len);
    free(crc_data);

    uint8_t crc_bytes[4] = {
        (crc >> 24) & 0xff,
        (crc >> 16) & 0xff,
        (crc >> 8) & 0xff,
        crc & 0xff
    };
    fwrite(crc_bytes, 1, 4, f);
}

static void png_write_text_chunk(FILE *f, const char *keyword, const char *text) {
    size_t key_len = strlen(keyword);
    size_t text_len = strlen(text);
    size_t data_len = key_len + 1 + text_len;  /* keyword + null + text */

    uint8_t *data = (uint8_t *)malloc(data_len);
    if (!data) return;

    memcpy(data, keyword, key_len);
    data[key_len] = 0;  /* Null separator */
    memcpy(data + key_len + 1, text, text_len);

    png_write_chunk(f, "tEXt", data, data_len);
    free(data);
}

/* ========================================================================
 * PNG Writing
 * ======================================================================== */

static int png_save_internal(const png_image *img, FILE *f,
                             const char *keyword, const char *text) {
    /* PNG signature */
    const uint8_t signature[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    fwrite(signature, 1, 8, f);

    /* IHDR chunk */
    uint8_t ihdr[13];
    ihdr[0] = (img->width >> 24) & 0xff;
    ihdr[1] = (img->width >> 16) & 0xff;
    ihdr[2] = (img->width >> 8) & 0xff;
    ihdr[3] = img->width & 0xff;
    ihdr[4] = (img->height >> 24) & 0xff;
    ihdr[5] = (img->height >> 16) & 0xff;
    ihdr[6] = (img->height >> 8) & 0xff;
    ihdr[7] = img->height & 0xff;
    ihdr[8] = 8;  /* Bit depth */
    ihdr[9] = (img->channels == 4) ? 6 : (img->channels == 3) ? 2 :
              (img->channels == 2) ? 4 : 0;  /* Color type */
    ihdr[10] = 0;  /* Compression */
    ihdr[11] = 0;  /* Filter */
    ihdr[12] = 0;  /* Interlace */

    png_write_chunk(f, "IHDR", ihdr, 13);

    /* Write metadata if provided */
    if (keyword && text) {
        png_write_text_chunk(f, keyword, text);
    }

    /* Prepare raw image data with filter bytes */
    int channels = img->channels;
    size_t row_bytes = 1 + img->width * channels;  /* +1 for filter byte */
    size_t raw_len = img->height * row_bytes;
    uint8_t *raw = (uint8_t *)malloc(raw_len);

    for (int y = 0; y < img->height; y++) {
        raw[y * row_bytes] = 0;  /* Filter: None */
        memcpy(raw + y * row_bytes + 1,
               img->data + y * img->width * channels,
               img->width * channels);
    }

    /* Compress with zlib (store mode) */
    size_t compressed_len;
    uint8_t *compressed = png_deflate_store(raw, raw_len, &compressed_len);
    free(raw);

    if (!compressed) return -1;

    /* IDAT chunk */
    png_write_chunk(f, "IDAT", compressed, compressed_len);
    free(compressed);

    /* IEND chunk */
    png_write_chunk(f, "IEND", NULL, 0);

    return 0;
}

int png_save(const png_image *img, const char *path) {
    if (!img || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int result = png_save_internal(img, f, NULL, NULL);
    fclose(f);
    return result;
}

int png_save_with_text(const png_image *img, const char *path,
                       const char *keyword, const char *text) {
    if (!img || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int result = png_save_internal(img, f, keyword, text);
    fclose(f);
    return result;
}

/* ========================================================================
 * Inflate (Decompression)
 * ======================================================================== */

#define PNG_MAXBITS 15

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t bytepos;
    uint32_t bitbuf;
    int bitcount;
} png_bitstream;

static int png_bitstream_fill(png_bitstream *bs, int n) {
    while (bs->bitcount < n && bs->bytepos < bs->len) {
        bs->bitbuf |= (uint32_t)bs->data[bs->bytepos++] << bs->bitcount;
        bs->bitcount += 8;
    }
    return bs->bitcount >= n;
}

static int png_bitstream_get(png_bitstream *bs, int n, uint32_t *out) {
    if (n == 0) {
        *out = 0;
        return 1;
    }
    if (!png_bitstream_fill(bs, n)) return 0;
    *out = bs->bitbuf & ((1u << n) - 1u);
    bs->bitbuf >>= n;
    bs->bitcount -= n;
    return 1;
}

static int png_bitstream_align(png_bitstream *bs) {
    uint32_t discard;
    int skip = bs->bitcount & 7;
    if (skip == 0) return 1;
    return png_bitstream_get(bs, skip, &discard);
}

static int png_bitstream_read_bytes(png_bitstream *bs, uint8_t *out, size_t len) {
    if (bs->bitcount == 0) {
        if (bs->bytepos + len > bs->len) return 0;
        memcpy(out, bs->data + bs->bytepos, len);
        bs->bytepos += len;
        return 1;
    }
    for (size_t i = 0; i < len; i++) {
        uint32_t v;
        if (!png_bitstream_get(bs, 8, &v)) return 0;
        out[i] = (uint8_t)v;
    }
    return 1;
}

typedef struct {
    uint16_t count[PNG_MAXBITS + 1];
    uint16_t symbol[288];
} png_huffman;

static int png_huffman_build(png_huffman *h, const uint8_t *lengths, int n) {
    uint16_t offs[PNG_MAXBITS + 1];
    int left = 1;

    memset(h->count, 0, sizeof(h->count));
    for (int i = 0; i < n; i++) {
        if (lengths[i] > PNG_MAXBITS) return 0;
        h->count[lengths[i]]++;
    }

    for (int len = 1; len <= PNG_MAXBITS; len++) {
        left <<= 1;
        left -= h->count[len];
        if (left < 0) return 0;
    }

    offs[1] = 0;
    for (int len = 1; len < PNG_MAXBITS; len++) {
        offs[len + 1] = offs[len] + h->count[len];
    }

    for (int i = 0; i < n; i++) {
        int len = lengths[i];
        if (len) {
            h->symbol[offs[len]++] = (uint16_t)i;
        }
    }

    return 1;
}

static int png_huffman_decode(png_bitstream *bs, const png_huffman *h, int *symbol) {
    uint32_t code = 0;
    uint32_t first = 0;
    uint32_t index = 0;

    for (int len = 1; len <= PNG_MAXBITS; len++) {
        uint32_t bit;
        if (!png_bitstream_get(bs, 1, &bit)) return 0;
        code |= bit;
        uint32_t count = h->count[len];
        if (code < first + count) {
            *symbol = h->symbol[index + (code - first)];
            return 1;
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return 0;
}

static int png_build_fixed_huffman(png_huffman *litlen, png_huffman *dist) {
    uint8_t litlen_lengths[288];
    uint8_t dist_lengths[32];

    for (int i = 0; i <= 143; i++) litlen_lengths[i] = 8;
    for (int i = 144; i <= 255; i++) litlen_lengths[i] = 9;
    for (int i = 256; i <= 279; i++) litlen_lengths[i] = 7;
    for (int i = 280; i <= 287; i++) litlen_lengths[i] = 8;
    for (int i = 0; i < 32; i++) dist_lengths[i] = 5;

    if (!png_huffman_build(litlen, litlen_lengths, 288)) return 0;
    if (!png_huffman_build(dist, dist_lengths, 32)) return 0;
    return 1;
}

static int png_build_dynamic_huffman(png_bitstream *bs, png_huffman *litlen, png_huffman *dist) {
    static const uint8_t order[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };
    uint32_t hlit, hdist, hclen;
    uint8_t code_lengths[19] = {0};
    png_huffman code_huff;

    if (!png_bitstream_get(bs, 5, &hlit)) return 0;
    if (!png_bitstream_get(bs, 5, &hdist)) return 0;
    if (!png_bitstream_get(bs, 4, &hclen)) return 0;
    int nlen = (int)hlit + 257;
    int ndist = (int)hdist + 1;
    int ncode = (int)hclen + 4;

    if (nlen > 288 || ndist > 32) return 0;

    for (int i = 0; i < ncode; i++) {
        uint32_t v;
        if (!png_bitstream_get(bs, 3, &v)) return 0;
        code_lengths[order[i]] = (uint8_t)v;
    }

    if (!png_huffman_build(&code_huff, code_lengths, 19)) return 0;

    uint8_t lengths[320];
    int total = nlen + ndist;
    int i = 0;
    int prev = 0;

    while (i < total) {
        int sym;
        if (!png_huffman_decode(bs, &code_huff, &sym)) return 0;
        if (sym <= 15) {
            lengths[i++] = (uint8_t)sym;
            prev = sym;
        } else if (sym == 16) {
            uint32_t repeat;
            if (i == 0) return 0;
            if (!png_bitstream_get(bs, 2, &repeat)) return 0;
            repeat += 3;
            if (i + (int)repeat > total) return 0;
            for (uint32_t r = 0; r < repeat; r++) lengths[i++] = (uint8_t)prev;
        } else if (sym == 17) {
            uint32_t repeat;
            if (!png_bitstream_get(bs, 3, &repeat)) return 0;
            repeat += 3;
            if (i + (int)repeat > total) return 0;
            for (uint32_t r = 0; r < repeat; r++) lengths[i++] = 0;
            prev = 0;
        } else if (sym == 18) {
            uint32_t repeat;
            if (!png_bitstream_get(bs, 7, &repeat)) return 0;
            repeat += 11;
            if (i + (int)repeat > total) return 0;
            for (uint32_t r = 0; r < repeat; r++) lengths[i++] = 0;
            prev = 0;
        } else {
            return 0;
        }
    }

    if (!png_huffman_build(litlen, lengths, nlen)) return 0;
    if (!png_huffman_build(dist, lengths + nlen, ndist)) return 0;

    return 1;
}

/* Zlib inflate (stored, fixed, and dynamic blocks) */
static uint8_t *png_inflate_zlib(const uint8_t *data, size_t len, size_t expected_len) {
    if (len < 6) return NULL;

    uint8_t cmf = data[0];
    uint8_t flg = data[1];
    if ((cmf & 0x0f) != 8) return NULL;
    if (((cmf << 8) + flg) % 31 != 0) return NULL;

    size_t pos = 2;
    if (flg & 0x20) {
        if (len < 10) return NULL;
        pos += 4;
    }
    if (len < pos + 4) return NULL;

    size_t deflate_len = len - pos - 4;
    png_bitstream bs = {data + pos, deflate_len, 0, 0, 0};

    uint8_t *out = (uint8_t *)malloc(expected_len);
    if (!out) return NULL;
    size_t out_pos = 0;

    static const int len_base[29] = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
        67, 83, 99, 115, 131, 163, 195, 227, 258
    };
    static const int len_extra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 0
    };
    static const int dist_base[30] = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25,
        33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
    };
    static const int dist_extra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
        4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
        9, 9, 10, 10, 11, 11, 12, 12, 13, 13
    };

    int final = 0;
    while (!final) {
        uint32_t bfinal, btype;
        if (!png_bitstream_get(&bs, 1, &bfinal)) goto fail;
        if (!png_bitstream_get(&bs, 2, &btype)) goto fail;
        final = (int)bfinal;

        if (btype == 0) {
            if (!png_bitstream_align(&bs)) goto fail;
            uint32_t stored_len, stored_nlen;
            if (!png_bitstream_get(&bs, 16, &stored_len)) goto fail;
            if (!png_bitstream_get(&bs, 16, &stored_nlen)) goto fail;
            if ((stored_len ^ 0xffffu) != stored_nlen) goto fail;
            if (out_pos + stored_len > expected_len) goto fail;
            if (!png_bitstream_read_bytes(&bs, out + out_pos, stored_len)) goto fail;
            out_pos += stored_len;
        } else if (btype == 1 || btype == 2) {
            png_huffman litlen, dist;
            if (btype == 1) {
                if (!png_build_fixed_huffman(&litlen, &dist)) goto fail;
            } else {
                if (!png_build_dynamic_huffman(&bs, &litlen, &dist)) goto fail;
            }

            for (;;) {
                int sym;
                if (!png_huffman_decode(&bs, &litlen, &sym)) goto fail;
                if (sym < 256) {
                    if (out_pos >= expected_len) goto fail;
                    out[out_pos++] = (uint8_t)sym;
                } else if (sym == 256) {
                    break;
                } else if (sym <= 285) {
                    int len_sym = sym - 257;
                    uint32_t extra, dist_extra_bits;
                    int dist_sym;
                    int length = len_base[len_sym];
                    if (len_extra[len_sym]) {
                        if (!png_bitstream_get(&bs, len_extra[len_sym], &extra)) goto fail;
                        length += (int)extra;
                    }
                    if (!png_huffman_decode(&bs, &dist, &dist_sym)) goto fail;
                    if (dist_sym >= 30) goto fail;
                    int distance = dist_base[dist_sym];
                    if (dist_extra[dist_sym]) {
                        if (!png_bitstream_get(&bs, dist_extra[dist_sym], &dist_extra_bits)) goto fail;
                        distance += (int)dist_extra_bits;
                    }
                    if (distance <= 0 || (size_t)distance > out_pos) goto fail;
                    if (out_pos + length > expected_len) goto fail;
                    for (int i = 0; i < length; i++) {
                        out[out_pos] = out[out_pos - distance];
                        out_pos++;
                    }
                } else {
                    goto fail;
                }
            }
        } else {
            goto fail;
        }
    }

    if (out_pos != expected_len) goto fail;

    uint32_t expected_adler = ((uint32_t)data[len - 4] << 24) |
                              ((uint32_t)data[len - 3] << 16) |
                              ((uint32_t)data[len - 2] << 8) |
                              (uint32_t)data[len - 1];
    if (png_adler32(out, expected_len) != expected_adler) goto fail;

    return out;

fail:
    free(out);
    return NULL;
}

/* ========================================================================
 * PNG Filtering
 * ======================================================================== */

static int png_abs(int x) { return x < 0 ? -x : x; }

static int png_unfilter_row(uint8_t *row, const uint8_t *prev_row,
                            int filter, int width, int channels) {
    int bpp = channels;

    switch (filter) {
        case 0:  /* None */
            break;
        case 1:  /* Sub */
            for (int i = bpp; i < width * channels; i++) {
                row[i] = row[i] + row[i - bpp];
            }
            break;
        case 2:  /* Up */
            if (prev_row) {
                for (int i = 0; i < width * channels; i++) {
                    row[i] = row[i] + prev_row[i];
                }
            }
            break;
        case 3:  /* Average */
            for (int i = 0; i < width * channels; i++) {
                int a = (i >= bpp) ? row[i - bpp] : 0;
                int b = prev_row ? prev_row[i] : 0;
                row[i] = row[i] + (a + b) / 2;
            }
            break;
        case 4:  /* Paeth */
            for (int i = 0; i < width * channels; i++) {
                int a = (i >= bpp) ? row[i - bpp] : 0;
                int b = prev_row ? prev_row[i] : 0;
                int c = (prev_row && i >= bpp) ? prev_row[i - bpp] : 0;
                int p = a + b - c;
                int pa = png_abs(p - a);
                int pb = png_abs(p - b);
                int pc = png_abs(p - c);
                int pr = (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c;
                row[i] = row[i] + pr;
            }
            break;
        default:
            return 0;
    }
    return 1;
}

/* ========================================================================
 * PNG Reading
 * ======================================================================== */

/* Read 4-byte big-endian integer from buffer */
static uint32_t png_read_be32_mem(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

png_image *png_load_mem(const uint8_t *data, size_t len) {
    if (!data || len < 8 || len > PNG_MAX_INPUT_BYTES) return NULL;

    /* Verify signature */
    const uint8_t expected[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (memcmp(data, expected, 8) != 0) return NULL;

    int width = 0, height = 0, color_type = -1, bit_depth = 0;
    int source_channels = 0;
    int seen_ihdr = 0, seen_idat = 0, seen_iend = 0;
    uint8_t *idat_data = NULL;
    size_t idat_len = 0;
    uint8_t palette[256 * 3];
    size_t palette_entries = 0;
    uint8_t palette_alpha[256];
    size_t palette_alpha_entries = 0;
    memset(palette_alpha, 255, sizeof(palette_alpha));

    size_t pos = 8;

    /* Read chunks */
    while (pos + 8 <= len) {
        uint32_t chunk_len = png_read_be32_mem(data + pos);
        const uint8_t *chunk_type = data + pos + 4;
        pos += 8;

        if ((size_t)chunk_len > len - pos || len - pos - chunk_len < 4) goto fail;
        const uint8_t *chunk_data = data + pos;
        uint32_t stored_crc = png_read_be32_mem(data + pos + chunk_len);
        uint32_t actual_crc = png_update_crc(0xffffffffu, chunk_type, 4);
        actual_crc = png_update_crc(actual_crc, chunk_data, chunk_len) ^ 0xffffffffu;
        if (actual_crc != stored_crc) goto fail;

        if (memcmp(chunk_type, "IHDR", 4) == 0) {
            if (seen_ihdr || seen_idat || chunk_len != 13) goto fail;
            uint32_t parsed_width = png_read_be32_mem(chunk_data);
            uint32_t parsed_height = png_read_be32_mem(chunk_data + 4);
            if (parsed_width == 0 || parsed_height == 0 ||
                parsed_width > PNG_MAX_DIMENSION ||
                parsed_height > PNG_MAX_DIMENSION ||
                (size_t)parsed_width * (size_t)parsed_height > PNG_MAX_PIXELS) {
                goto fail;
            }
            bit_depth = chunk_data[8];
            if (chunk_data[10] != 0 || chunk_data[11] != 0 ||
                chunk_data[12] != 0) {
                goto fail;
            }
            width = (int)parsed_width;
            height = (int)parsed_height;
            color_type = chunk_data[9];
            switch (color_type) {
                case 0: source_channels = 1; break;
                case 2: source_channels = 3; break;
                case 3: source_channels = 1; break;
                case 4: source_channels = 2; break;
                case 6: source_channels = 4; break;
                default: goto fail;
            }
            if (color_type == 3) {
                if (bit_depth != 1 && bit_depth != 2 &&
                    bit_depth != 4 && bit_depth != 8) goto fail;
            } else if (bit_depth != 8) {
                goto fail;
            }
            seen_ihdr = 1;
        } else if (memcmp(chunk_type, "IDAT", 4) == 0) {
            /* Accumulate IDAT chunks */
            if (!seen_ihdr || seen_iend ||
                (size_t)chunk_len > PNG_MAX_INPUT_BYTES - idat_len) goto fail;
            uint8_t *grown = (uint8_t *)realloc(idat_data, idat_len + chunk_len);
            if (!grown && chunk_len != 0) goto fail;
            idat_data = grown;
            memcpy(idat_data + idat_len, chunk_data, chunk_len);
            idat_len += chunk_len;
            seen_idat = 1;
        } else if (memcmp(chunk_type, "PLTE", 4) == 0) {
            if (!seen_ihdr || seen_idat || chunk_len == 0 ||
                chunk_len > sizeof(palette) || chunk_len % 3 != 0) goto fail;
            memcpy(palette, chunk_data, chunk_len);
            palette_entries = chunk_len / 3;
        } else if (memcmp(chunk_type, "tRNS", 4) == 0) {
            if (!seen_ihdr || seen_idat) goto fail;
            if (color_type == 3) {
                if (chunk_len > sizeof(palette_alpha)) goto fail;
                memcpy(palette_alpha, chunk_data, chunk_len);
                palette_alpha_entries = chunk_len;
            } else if (!((color_type == 0 && chunk_len == 2) ||
                         (color_type == 2 && chunk_len == 6))) {
                goto fail;
            }
        } else if (memcmp(chunk_type, "IEND", 4) == 0) {
            if (!seen_ihdr || !seen_idat || chunk_len != 0) goto fail;
            seen_iend = 1;
            break;
        } else if ((chunk_type[0] & 0x20) == 0) {
            /* Unknown critical chunks change pixel interpretation. */
            goto fail;
        }
        pos += (size_t)chunk_len + 4;
    }

    if (!seen_ihdr || !seen_idat || !seen_iend || !idat_data) goto fail;
    if (color_type == 3 && palette_entries == 0) goto fail;

    /* Determine channels from color type */
    int channels;
    switch (color_type) {
        case 0: channels = 1; break;  /* Grayscale */
        case 2: channels = 3; break;  /* RGB */
        case 3: channels = palette_alpha_entries ? 4 : 3; break; /* Palette */
        case 4: channels = 2; break;  /* Grayscale + Alpha */
        case 6: channels = 4; break;  /* RGBA */
        default: goto fail;
    }

    /* Decompress */
    size_t row_payload = color_type == 3
        ? ((size_t)width * (size_t)bit_depth + 7) / 8
        : (size_t)width * (size_t)source_channels;
    size_t row_bytes = 1 + row_payload;
    if ((size_t)height > SIZE_MAX / row_bytes) goto fail;
    size_t raw_len = (size_t)height * row_bytes;
    uint8_t *raw = png_inflate_zlib(idat_data, idat_len, raw_len);
    free(idat_data);
    idat_data = NULL;

    if (!raw) return NULL;

    /* Create image and apply filters */
    png_image *img = png_create(width, height, channels);
    if (!img) {
        free(raw);
        return NULL;
    }

    uint8_t *prev_row = NULL;

    for (int y = 0; y < height; y++) {
        uint8_t *row_data = raw + (size_t)y * row_bytes;
        int filter = row_data[0];
        uint8_t *row = row_data + 1;

        int filter_width = color_type == 3 ? (int)row_payload : width;
        int filter_channels = color_type == 3 ? 1 : source_channels;
        if (!png_unfilter_row(row, prev_row, filter,
                              filter_width, filter_channels)) {
            png_free(img);
            free(raw);
            return NULL;
        }

        uint8_t *dst = img->data + (size_t)y * (size_t)width * (size_t)channels;
        if (color_type == 3) {
            for (int x = 0; x < width; x++) {
                size_t bit = (size_t)x * (size_t)bit_depth;
                unsigned shift = 8u - (unsigned)bit_depth -
                                 (unsigned)(bit & 7u);
                size_t index = (row[bit >> 3] >> shift) &
                               ((1u << bit_depth) - 1u);
                if (index >= palette_entries) {
                    png_free(img);
                    free(raw);
                    return NULL;
                }
                dst[(size_t)x * channels + 0] = palette[index * 3 + 0];
                dst[(size_t)x * channels + 1] = palette[index * 3 + 1];
                dst[(size_t)x * channels + 2] = palette[index * 3 + 2];
                if (channels == 4) dst[(size_t)x * 4 + 3] = palette_alpha[index];
            }
        } else {
            memcpy(dst, row, (size_t)width * (size_t)channels);
        }
        prev_row = row;
    }

    free(raw);
    return img;

fail:
    free(idat_data);
    return NULL;
}

png_image *png_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long end = ftell(f);
    if (end < 0 || (unsigned long)end > PNG_MAX_INPUT_BYTES ||
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

    png_image *img = png_load_mem(file_data, file_size);
    free(file_data);
    return img;
}

/* Clean up internal macros */
#undef PNG_MAXBITS

#endif /* PNG_IMPLEMENTATION */
