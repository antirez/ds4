#ifndef PULSAR_SHA1_HPP
#define PULSAR_SHA1_HPP

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Minimal SHA-1 (C++ port of the kvstore-internal implementation). Used for
 * content-addressed checkpoint names, not for security. */

namespace pulsar {

class Sha1 {
public:
    Sha1() { reset(); }

    void reset() {
        h_[0] = 0x67452301u;
        h_[1] = 0xefcdab89u;
        h_[2] = 0x98badcfeu;
        h_[3] = 0x10325476u;
        h_[4] = 0xc3d2e1f0u;
        bytes_ = 0;
        used_ = 0;
    }

    void update(const void *ptr, size_t len) {
        const uint8_t *p = static_cast<const uint8_t *>(ptr);
        bytes_ += len;
        while (len != 0) {
            size_t n = 64 - used_;
            if (n > len) n = len;
            memcpy(block_ + used_, p, n);
            used_ += n;
            p += n;
            len -= n;
            if (used_ == 64) {
                transform(block_);
                used_ = 0;
            }
        }
    }

    void final(uint8_t out[20]) {
        uint64_t bits = bytes_ * 8;
        uint8_t one = 0x80;
        uint8_t zero = 0;
        update(&one, 1);
        while (used_ != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
        update(len, sizeof(len));
        for (int i = 0; i < 5; i++) {
            out[i * 4] = (uint8_t)(h_[i] >> 24);
            out[i * 4 + 1] = (uint8_t)(h_[i] >> 16);
            out[i * 4 + 2] = (uint8_t)(h_[i] >> 8);
            out[i * 4 + 3] = (uint8_t)h_[i];
        }
    }

    /* One-shot digest rendered as 40 lowercase hex chars + NUL. */
    static void bytes_hex(const void *ptr, size_t len, char out[41]) {
        Sha1 c;
        c.update(ptr, len);
        uint8_t digest[20];
        c.final(digest);
        hex20(digest, out);
    }

    static void hex20(const uint8_t in[20], char out[41]) {
        static const char hex[] = "0123456789abcdef";
        for (int i = 0; i < 20; i++) {
            out[i * 2] = hex[in[i] >> 4];
            out[i * 2 + 1] = hex[in[i] & 15];
        }
        out[40] = '\0';
    }

private:
    static uint32_t rol32(uint32_t v, int n) {
        return (v << n) | (v >> (32 - n));
    }

    void transform(const uint8_t block[64]) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)block[i * 4] << 24) |
                   ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8) |
                   (uint32_t)block[i * 4 + 3];
        }
        for (int i = 16; i < 80; i++)
            w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        uint32_t a = h_[0], b = h_[1], d = h_[3], e = h_[4];
        uint32_t cc = h_[2];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) {
                f = (b & cc) | ((~b) & d);
                k = 0x5a827999u;
            } else if (i < 40) {
                f = b ^ cc ^ d;
                k = 0x6ed9eba1u;
            } else if (i < 60) {
                f = (b & cc) | (b & d) | (cc & d);
                k = 0x8f1bbcdcu;
            } else {
                f = b ^ cc ^ d;
                k = 0xca62c1d6u;
            }
            uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
            e = d;
            d = cc;
            cc = rol32(b, 30);
            b = a;
            a = tmp;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += cc;
        h_[3] += d;
        h_[4] += e;
    }

    uint32_t h_[5];
    uint64_t bytes_;
    uint8_t block_[64];
    size_t used_;
};

} // namespace pulsar

#endif /* PULSAR_SHA1_HPP */
