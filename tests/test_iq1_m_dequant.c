/* Check the IQ1_M superblock layout against values derived by hand.
 *
 * The point of deriving the expectations by hand is that there is no second
 * IQ1_M implementation in the tree to compare against: gguf-tools marks the
 * type as having no dequantiser, and the only other reader is the GPU device
 * function this code was written from. Comparing against that would just
 * restate the same arithmetic. So instead each case sets exactly one field of
 * the block and asserts a value worked out from the format description:
 *
 *   value = d * (2*k + 1) * (grid[j] + delta)
 *
 * The layout being checked, from vec_dot_iq1_m_q8_1:
 *
 *   - scales[8], read as uint16 sc[4]. The low 12 bits of each word hold four
 *     3-bit sub-scales; the top nibble of each is one nibble of the f16
 *     superblock scale, reassembled as
 *       (sc[0]>>12) | ((sc[1]>>8)&0x00F0) | ((sc[2]>>4)&0x0F00) | (sc[3]&0xF000)
 *   - qs[32], one byte per 8 values: low 8 bits of the codebook index.
 *   - qh[16], one nibble per 8 values: 3 index high bits plus a delta sign bit.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QK_K 256
#define IQ1M_DELTA 0.125f

/* Mirrors block_iq1_m in ds4.c: no leading scale field. */
typedef struct {
    uint8_t qs[QK_K / 8];
    uint8_t qh[QK_K / 16];
    uint8_t scales[QK_K / 32];
} test_block_iq1_m;

/* Mirrors block_q8_K in ds4.c. bsums is unused by the IQ1_M dot product but
 * has to be present so the struct size and field offsets agree. */
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} test_block_q8_K;

void ds4_test_dequantize_iq1_m_block(const void *block, float *out);
float ds4_test_vec_dot_iq1_m_q8_K(int n, const void *x, const void *y);

static int failures;

/* f16 1.0 is 0x3C00. Spread across the four top nibbles, with every sub-scale
 * left at zero so that 2*k+1 == 1. */
static void set_unit_scale(test_block_iq1_m *b) {
    uint16_t sc[4] = { 0x0000, 0x0000, 0xC000, 0x3000 };
    memcpy(b->scales, sc, sizeof(sc));
}

static void check(const char *what, float got, float want) {
    if (fabsf(got - want) > 1e-6f) {
        printf("  FAIL %-46s got %.6f want %.6f\n", what, got, want);
        failures++;
    } else {
        printf("  ok   %-46s %.6f\n", what, got);
    }
}

/* Codebook entry 0 is 0xffffffffffffffff: eight bytes of -1. Entry 1 is
 * 0xffffffffffffff01, so on a little-endian host its first value is +1 and the
 * remaining seven are -1. Both are the opening entries of the table and can be
 * read straight out of ggml-common.h. */
int main(void) {
    test_block_iq1_m b;
    float out[QK_K];

    /* 1. d = 1, k = 0, grid entry 0, delta bit clear.
     *    value = 1 * 1 * (-1 + 0.125) = -0.875 everywhere. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    ds4_test_dequantize_iq1_m_block(&b, out);
    int uniform = 1;
    for (int j = 0; j < QK_K; j++) {
        if (fabsf(out[j] - (-0.875f)) > 1e-6f) uniform = 0;
    }
    if (!uniform) {
        printf("  FAIL baseline block is not uniformly -0.875 "
               "(out[0]=%.6f out[255]=%.6f)\n", out[0], out[255]);
        failures++;
    } else {
        printf("  ok   baseline: all 256 values -0.875\n");
    }

    /* 2. Delta sign bit set in every qh nibble. 0x8 & 0x07 == 0, so the
     *    codebook index is unchanged and only delta moves:
     *    value = 1 * 1 * (-1 - 0.125) = -1.125. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    memset(b.qh, 0x88, sizeof(b.qh));
    ds4_test_dequantize_iq1_m_block(&b, out);
    check("delta bit set -> out[0]", out[0], -1.125f);
    check("delta bit set -> out[255]", out[255], -1.125f);

    /* 3. Sub-scales for ib == 0 set to k = 7 via the low 6 bits of sc[0].
     *    Those six bits cover both halves of the first 32 values, giving
     *    2*7+1 = 15, so value = 1 * 15 * -0.875 = -13.125. sc[0] bits 6..11
     *    cover ib == 1 and stay zero, so the next 32 values are untouched. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    b.scales[0] = 0x3F;
    ds4_test_dequantize_iq1_m_block(&b, out);
    check("k=7 on ib0 -> out[0]", out[0], -13.125f);
    check("k=7 on ib0 -> out[31]", out[31], -13.125f);
    check("k=7 on ib0 leaves ib1 -> out[32]", out[32], -0.875f);

    /* 4. Only the first sub-scale of ib == 0 set (bits 0..2), so the first 16
     *    values scale by 15 and the next 16 by 1. This is the granularity that
     *    the tiled MMQ path cannot express with one half2 per 32 values. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    b.scales[0] = 0x07;
    ds4_test_dequantize_iq1_m_block(&b, out);
    check("k per 16: out[15] scaled", out[15], -13.125f);
    check("k per 16: out[16] unscaled", out[16], -0.875f);

    /* 5. Codebook index 1 for the first group of 8. Entry 1 is
     *    0xffffffffffffff01, so the first value is +1 and the rest -1:
     *    out[0] = 1 * 1 * (+1 + 0.125) = +1.125, out[1] = -0.875. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    b.qs[0] = 1;
    ds4_test_dequantize_iq1_m_block(&b, out);
    check("grid entry 1 -> out[0]", out[0], 1.125f);
    check("grid entry 1 -> out[1]", out[1], -0.875f);

    /* 6. Delta sign is per 8 values, not per 32: set it for the second group
     *    of 8 only (qh[0] high nibble) and the first group must not move. */
    memset(&b, 0, sizeof(b));
    set_unit_scale(&b);
    b.qh[0] = 0x80;
    ds4_test_dequantize_iq1_m_block(&b, out);
    check("delta per 8: out[0] unchanged", out[0], -0.875f);
    check("delta per 8: out[8] flipped", out[8], -1.125f);

    /* 7. The q8_K dot product against the dequantiser, which the cases above
     *    have already pinned down. This is not circular: the dot product
     *    restructures the arithmetic to keep the codebook contribution in
     *    integers and apply each 16-value sub-scale once,
     *
     *      sum_j (grid[j]+delta)*q8[j] == sum_j grid[j]*q8[j] + delta*sum_j q8[j]
     *
     *    so agreeing with a plain sum over dequantised values is a real check
     *    on that rearrangement, on the per-8 delta and per-16 sub-scale
     *    indexing, and on the q8_K scale being applied exactly once.
     *
     *    Fields are filled from a fixed LCG so the case is reproducible and
     *    exercises many codebook entries, sub-scales and delta signs at once
     *    rather than the single-field blocks above. */
    uint32_t rng = 12345u;
#define NEXT (rng = rng * 1103515245u + 12345u, (rng >> 16))
    test_block_iq1_m blocks[3];
    test_block_q8_K act[3];
    for (int b = 0; b < 3; b++) {
        for (int j = 0; j < QK_K / 8; j++)  blocks[b].qs[j] = (uint8_t)(NEXT & 0xFF);
        for (int j = 0; j < QK_K / 16; j++) blocks[b].qh[j] = (uint8_t)(NEXT & 0xFF);
        /* Keep the top nibble of each scales word as a real f16: reuse the
         * unit-scale pattern and vary only the sub-scale bits, so d stays 1.0
         * and a stray NaN cannot mask a wrong sum. */
        uint16_t sc[4] = { 0x0000, 0x0000, 0xC000, 0x3000 };
        for (int w = 0; w < 4; w++) sc[w] |= (uint16_t)(NEXT & 0x0FFF);
        memcpy(blocks[b].scales, sc, sizeof(sc));

        act[b].d = 0.5f + (float)(NEXT % 100u) / 100.0f;
        for (int j = 0; j < QK_K; j++) act[b].qs[j] = (int8_t)(NEXT & 0xFF);
        memset(act[b].bsums, 0, sizeof(act[b].bsums));
    }
#undef NEXT

    float reference = 0.0f;
    for (int b = 0; b < 3; b++) {
        float deq[QK_K];
        ds4_test_dequantize_iq1_m_block(&blocks[b], deq);
        for (int j = 0; j < QK_K; j++) {
            reference += deq[j] * act[b].d * (float)act[b].qs[j];
        }
    }
    const float got = ds4_test_vec_dot_iq1_m_q8_K(3 * QK_K, blocks, act);
    const float tol = fabsf(reference) * 1e-4f + 1e-3f;
    if (fabsf(got - reference) > tol) {
        printf("  FAIL %-46s got %.4f want %.4f (tol %.4f)\n",
               "q8_K dot matches dequantised sum", got, reference, tol);
        failures++;
    } else {
        printf("  ok   %-46s %.4f vs %.4f\n",
               "q8_K dot matches dequantised sum", got, reference);
    }

    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall IQ1_M layout checks passed\n");
    return 0;
}
