/* CPU codec tests for the Qwen3.8 TurboQuant KV cache.  No GPU, no weights.
 * Golden vectors pinned by tests/gen_qwen4_tq_tables.py (mlx-vlm reference,
 * pins in docs/superpowers/plans/2026-09-23-qwen4-turboquant-all-widths.md).
 * Build: make test-qwen4-tq */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds4_qwen4_tq.h"

static int failures;

static void check(const char *what, int ok) {
    printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

/* The pinned golden row: x[d] = (((d*2654435761) % 2001) - 1000) / 250 */
static void golden_row(float *x) {
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) {
        const uint32_t h = d * 2654435761u;
        x[d] = ((float)(h % 2001u) - 1000.0f) / 250.0f;
    }
}

/* Table integrity: every width's codebook is strictly increasing and its
 * midpoints are the neighbour averages the quantizer binary-searches. */
static void test_tables(void) {
    for (uint32_t bits = DS4_QWEN4_TQ_MIN_BITS; bits <= DS4_QWEN4_TQ_MAX_BITS; bits++) {
        const float *cb = ds4_qwen4_tq_cb(bits), *mid = ds4_qwen4_tq_mid(bits);
        const uint32_t levels = 1u << bits;
        char what[96];
        int sorted = 1, mids = 1;
        for (uint32_t i = 1u; i < levels; i++) {
            if (!(cb[i - 1u] < cb[i])) sorted = 0;
            if (fabsf(mid[i - 1u] - 0.5f * (cb[i - 1u] + cb[i])) > 1e-7f) mids = 0;
        }
        snprintf(what, sizeof(what), "codebook%u sorted + midpoints pinned", bits);
        check(what, sorted && mids);
    }
}

static void test_bits_supported(void) {
    int ok = 1, junk = 1;
    for (uint32_t bits = DS4_QWEN4_TQ_MIN_BITS; bits <= DS4_QWEN4_TQ_MAX_BITS; bits++)
        if (!ds4_qwen4_tq_bits_supported(bits)) ok = 0;
    /* 1 bit is deliberately out: two centroids are no KV cache. */
    for (uint32_t bits = 0u; bits < DS4_QWEN4_TQ_MIN_BITS; bits++)
        if (ds4_qwen4_tq_bits_supported(bits)) junk = 0;
    for (uint32_t bits = DS4_QWEN4_TQ_MAX_BITS + 1u; bits <= 32u; bits++)
        if (ds4_qwen4_tq_bits_supported(bits)) junk = 0;
    check("bits_supported covers 2..8 only", ok && junk);
}

/* oMLX layer policy on the production shape (49 layers, 1 nextn, interval 4:
 * attention at il % 4 == 3; TQ on 3..43, il 47 and nextn 48 stay f16). */
static void test_layer_policy(void) {
    for (uint32_t il = 0; il < 49u; il++) {
        const bool want = (il % 4u == 3u) && il <= 43u;
        char what[64];
        snprintf(what, sizeof(what), "policy il=%u (production)", il);
        check(what, ds4_qwen4_tq_layer_want(il, 49u, 1u, 4u) == want);
    }
    check("policy no-nextn last excluded",
          !ds4_qwen4_tq_layer_want(47u, 48u, 0u, 4u) &&
           ds4_qwen4_tq_layer_want(43u, 48u, 0u, 4u));
    check("policy shallow quantizes all attention",
          ds4_qwen4_tq_layer_want(1u, 2u, 0u, 1u) &&
          ds4_qwen4_tq_layer_want(0u, 2u, 0u, 1u));
    check("policy nextn never quantized",
          !ds4_qwen4_tq_layer_want(2u, 3u, 1u, 1u));
    check("policy linear never quantized",
          !ds4_qwen4_tq_layer_want(0u, 49u, 1u, 4u) &&
          !ds4_qwen4_tq_layer_want(6u, 49u, 1u, 4u));
}

int main(void) {
    static float x[DS4_QWEN4_TQ_DIM], y[DS4_QWEN4_TQ_DIM], z[DS4_QWEN4_TQ_DIM];
    golden_row(x);

    /* RHT: forward/inverse pair is the identity */
    ds4_qwen4_tq_rht(x, y);
    ds4_qwen4_tq_rht_inverse(y, z);
    double worst = 0;
    for (uint32_t i = 0; i < DS4_QWEN4_TQ_DIM; i++) {
        const double d = fabs((double)z[i] - (double)x[i]);
        if (d > worst) worst = d;
    }
    check("rht roundtrip identity", worst < 1e-5);

    /* in-place safety */
    memcpy(z, x, sizeof(z));
    ds4_qwen4_tq_rht(z, z);
    double worst2 = 0;
    for (uint32_t i = 0; i < DS4_QWEN4_TQ_DIM; i++) {
        const double d = fabs((double)z[i] - (double)y[i]);
        if (d > worst2) worst2 = d;
    }
    check("rht in-place matches two-buffer", worst2 == 0.0);

    /* sizes: a row is bits*8 words of codes plus the f16 norms, 16-aligned */
    int sizes_ok = 1, align_ok = 1, blob_ok = 1;
    for (uint32_t bits = DS4_QWEN4_TQ_MIN_BITS; bits <= DS4_QWEN4_TQ_MAX_BITS; bits++) {
        if (ds4_qwen4_tq_packed_width(bits) != bits * 8u) sizes_ok = 0;
        if ((ds4_qwen4_tq_norms_offset(1000u, bits) % 16u) != 0u) align_ok = 0;
        if (ds4_qwen4_tq_blob_bytes(4u, bits) !=
                ds4_qwen4_tq_norms_offset(4u, bits) + 4u * 2u * 2u) blob_ok = 0;
    }
    check("packed_width == bits*8 for every width", sizes_ok);
    check("norms_offset 16-aligned for every width", align_ok);
    check("blob_bytes = norms_offset + cap*Hkv*2", blob_ok);

    /* Pinned per width by tests/gen_qwen4_tq_tables.py (golden row above). */
    struct { uint32_t bits; uint16_t norm_bits; uint32_t w[4]; double cos_floor; } golden[] = {
        { 2, 0x5098u, { 0x57ab695au, 0x9724e5dcu, 0x1405259au, 0x4ea47648u }, 0.9475 },
        { 3, 0x5098u, { 0x6e5226e4u, 0xbcb14d59u, 0x896312f5u, 0x5211aae5u }, 0.9838 },
        { 4, 0x5098u, { 0x58856799u, 0x564b8abcu, 0xea77c5c2u, 0x854c3954u }, 0.9959 },
        { 5, 0x5098u, { 0x22a63e94u, 0x88571a4cu, 0xa7254b0fu, 0x27ed5cfcu }, 0.9982 },
        { 6, 0x5098u, { 0xd359daebu, 0xecf73e18u, 0x3d633486u, 0x1fd4fd87u }, 0.9995 },
        { 7, 0x5098u, { 0x658eeb56u, 0xf13b0a3au, 0xbec877b4u, 0x778b3ab0u }, 0.9994 },
        { 8, 0x5098u, { 0x5976acadu, 0x3b848e4cu, 0x86bcd2e5u, 0x3b582dd9u }, 0.9999 },
    };
    const uint32_t n_golden = sizeof(golden) / sizeof(golden[0]);
    double prev_cos = 0.0;
    for (uint32_t g = 0; g < n_golden; g++) {
        uint32_t codes[DS4_QWEN4_TQ_MAX_BITS * 8u];
        uint16_t nh;
        char what[96];
        ds4_qwen4_tq_quantize_row(x, golden[g].bits, codes, &nh);
        snprintf(what, sizeof(what), "golden norm f16 bits=%u", golden[g].bits);
        check(what, nh == golden[g].norm_bits);
        snprintf(what, sizeof(what), "golden packed[0..3] bits=%u", golden[g].bits);
        check(what, memcmp(codes, golden[g].w, 16) == 0);

        /* the dequantized row keeps at least the pinned cosine against x */
        ds4_qwen4_tq_dequantize_row(codes, nh, golden[g].bits, y);
        double dot = 0, nx = 0, ny = 0;
        for (uint32_t i = 0; i < DS4_QWEN4_TQ_DIM; i++) {
            dot += (double)x[i] * y[i];
            nx += (double)x[i] * x[i];
            ny += (double)y[i] * y[i];
        }
        const double cos = dot / (sqrt(nx) * sqrt(ny));
        snprintf(what, sizeof(what), "roundtrip cosine bits=%u (%.6f)", golden[g].bits, cos);
        check(what, cos >= golden[g].cos_floor);
        /* Wider codebooks can only help; a dip means a width read the wrong
         * slice of the concatenated table. */
        snprintf(what, sizeof(what), "cosine improves with bits=%u", golden[g].bits);
        check(what, g == 0u || cos > prev_cos);
        prev_cos = cos;

        /* quantize is deterministic: repacking the same row gives the same
         * words, so checkpoint save/load and the GPU prep kernel agree */
        uint32_t again[DS4_QWEN4_TQ_MAX_BITS * 8u];
        uint16_t nh2;
        ds4_qwen4_tq_quantize_row(x, golden[g].bits, again, &nh2);
        snprintf(what, sizeof(what), "quantize deterministic bits=%u", golden[g].bits);
        check(what, nh2 == nh && memcmp(again, codes,
              ds4_qwen4_tq_packed_width(golden[g].bits) * 4u) == 0);
    }

    test_bits_supported();
    test_tables();
    test_layer_policy();

    if (failures) {
        fprintf(stderr, "%d codec test(s) failed\n", failures);
        return 1;
    }
    printf("qwen4 tq codec: all ok\n");
    return 0;
}
