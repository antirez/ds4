/* Numerical check for the wave64-native
 * moe_gate_up_mid_decode_lut_hwarp32_kernel (16-lane groups) against both a
 * CPU double-precision reference and the original, untouched
 * moe_gate_up_mid_decode_lut_qwarp32_kernel (8-lane groups).
 *
 * This is the decode-path MoE gate/up/mid kernel profiled this session at
 * 13.7% of GPU decode time: for this project's iq2_xxs decode models,
 * xq_blocks == 16 (confirmed live via a temporary instrumented run), so
 * widening each row's cooperating-lane group from 8 to 16 lanes lets every
 * lane do exactly one block instead of two, halving the serial per-row loop.
 * Both kernels compute the same IQ2_XXS x Q8_K dot products in different
 * lane groupings/summation order, so only FP summation-order noise should
 * separate their outputs from each other or from the CPU reference -- not
 * the ~100% a lane-mapping or block-index bug would produce.
 *
 * The CPU reference below reuses the *exact* grid/sign lookup tables baked
 * into ds4_iq2_tables_cuda.inc (copied verbatim, not re-derived), and
 * replicates dev_iq2_i8x8_lut's bit-level negate-per-grid-element decode as
 * a plain scalar loop -- not a re-implementation of a different algorithm,
 * just the same one written without SIMD-in-a-register tricks.
 *
 * Uses the test-only ds4_gpu_test_moe_gate_up_mid_decode_lut_tensor() entry
 * point (rocm/ds4_rocm_moe_launch.cuh), which bypasses the routed-MoE
 * expert-selection/sorting machinery to drive either kernel directly.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define XQ_BLOCKS       16u   /* matches this project's live decode shape */
#define QK_K            256u
#define EXPERT_MID_DIM  777u  /* deliberately not a multiple of 16/32/64 */
#define N_EXPERT        3u
#define N_TOK           2u
#define PAIR_COUNT      (N_TOK * N_EXPERT)

#define Q8K_BYTES  (4u + QK_K + 2u * (QK_K / 16u))      /* d + qs[256] + bsums[16] = 292 */
#define IQ2_BYTES  (2u + 2u * (QK_K / 8u))              /* d(f16) + qs[32] = 66 */

static uint32_t g_rng = 0xc2b2ae35u;
static uint32_t next_rand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
static float next_uniform(void) { return (float)(next_rand() & 0xffffu) / 32768.0f - 1.0f; }

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t half = 1u << (shift - 1);
        uint32_t r = mant + half - 1u + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (r >> shift));
    }
    uint32_t r = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (r & 0x800000u) { r = 0; exp++; if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (r >> 13));
}
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) { x = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) { x = sign | 0x7f800000u | (mant << 13); }
    else { x = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
    float f; memcpy(&f, &x, 4); return f;
}

/* Verbatim from ds4_iq2_tables_cuda.inc (cuda_ksigns_iq2xs / cuda_iq2xxs_grid),
 * copied as plain data so the CPU reference decodes with byte-identical
 * lookup tables to the GPU kernels. */
static const uint8_t g_ksigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t g_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* dev_iq2_i8x8_lut, as a scalar loop: element i of the 8-int8 grid vector is
 * negated iff bit i of the (already parity-corrected) sign byte is set. */
static int32_t dot_grid_sign_q8_8(uint64_t grid_word, uint8_t sbyte, const int8_t *q8_8) {
    int8_t g[8];
    memcpy(g, &grid_word, 8);
    int32_t acc = 0;
    for (int i = 0; i < 8; i++) {
        const int8_t gv = ((sbyte >> i) & 1u) ? (int8_t)(-(int32_t)g[i]) : g[i];
        acc += (int32_t)gv * (int32_t)q8_8[i];
    }
    return acc;
}

/* dev_dot_iq2_xxs_q8_K_block_lut, as scalar CPU double-precision math. */
static double cpu_dot_iq2xxs_q8k_block(const unsigned char *iq2_blk, const unsigned char *q8k_blk) {
    uint16_t dbits; memcpy(&dbits, iq2_blk, 2);
    const float xd = f16_to_f32(dbits);
    const uint16_t *q2 = (const uint16_t *)(iq2_blk + 2);
    float qd; memcpy(&qd, q8k_blk, 4);
    const int8_t *q8 = (const int8_t *)(q8k_blk + 4);
    int32_t bsum = 0;
    for (uint32_t ib32 = 0; ib32 < QK_K / 32u; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t sumi = 0;
        sumi += dot_grid_sign_q8_8(g_grid[aux0 & 0xffu],         g_ksigns[(aux1 >> 0)  & 127u], q8 + 0);
        sumi += dot_grid_sign_q8_8(g_grid[(aux0 >> 8) & 0xffu],  g_ksigns[(aux1 >> 7)  & 127u], q8 + 8);
        sumi += dot_grid_sign_q8_8(g_grid[(aux0 >> 16) & 0xffu], g_ksigns[(aux1 >> 14) & 127u], q8 + 16);
        sumi += dot_grid_sign_q8_8(g_grid[(aux0 >> 24) & 0xffu], g_ksigns[(aux1 >> 21) & 127u], q8 + 24);
        bsum += sumi * ls;
        q8 += 32;
    }
    return 0.125 * (double)xd * (double)qd * (double)bsum;
}

static void make_iq2_row(unsigned char *row, uint32_t n_blocks) {
    for (uint32_t b = 0; b < n_blocks; b++) {
        unsigned char *blk = row + (uint64_t)b * IQ2_BYTES;
        const uint16_t d = f32_to_f16(0.01f + 0.02f * fabsf(next_uniform()));
        memcpy(blk, &d, 2);
        uint16_t *q2 = (uint16_t *)(blk + 2);
        for (uint32_t ib32 = 0; ib32 < QK_K / 32u; ib32++) {
            const uint32_t g0 = next_rand() & 0xffu, g1 = next_rand() & 0xffu;
            const uint32_t g2 = next_rand() & 0xffu, g3 = next_rand() & 0xffu;
            const uint32_t s0 = next_rand() & 0x7fu, s1 = next_rand() & 0x7fu;
            const uint32_t s2 = next_rand() & 0x7fu, s3 = next_rand() & 0x7fu;
            const uint32_t ls_nibble = next_rand() & 0xfu;
            const uint32_t aux0 = g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
            const uint32_t aux1 = s0 | (s1 << 7) | (s2 << 14) | (s3 << 21) | (ls_nibble << 28);
            q2[ib32 * 4 + 0] = (uint16_t)(aux0 & 0xffffu);
            q2[ib32 * 4 + 1] = (uint16_t)(aux0 >> 16);
            q2[ib32 * 4 + 2] = (uint16_t)(aux1 & 0xffffu);
            q2[ib32 * 4 + 3] = (uint16_t)(aux1 >> 16);
        }
    }
}

static void make_q8k_block(unsigned char *blk) {
    const float d = 0.05f + 0.05f * fabsf(next_uniform());
    memcpy(blk, &d, 4);
    int8_t *qs = (int8_t *)(blk + 4);
    for (uint32_t j = 0; j < QK_K; j++) qs[j] = (int8_t)(int)(next_uniform() * 100.0f);
    memset(blk + 4 + QK_K, 0, 2u * (QK_K / 16u));
}

static double cpu_ref_mid(const unsigned char *gate_row, const unsigned char *up_row,
                           const unsigned char *xq_tok, uint32_t n_blocks,
                           float route_weight) {
    double gate = 0.0, up = 0.0;
    for (uint32_t b = 0; b < n_blocks; b++) {
        gate += cpu_dot_iq2xxs_q8k_block(gate_row + (uint64_t)b * IQ2_BYTES, xq_tok + (uint64_t)b * Q8K_BYTES);
        up   += cpu_dot_iq2xxs_q8k_block(up_row   + (uint64_t)b * IQ2_BYTES, xq_tok + (uint64_t)b * Q8K_BYTES);
    }
    const double silu = gate / (1.0 + exp(-gate));
    return silu * up * (double)route_weight;
}

static int run_once(ds4_gpu_tensor *gate_out, ds4_gpu_tensor *up_out, ds4_gpu_tensor *mid_out,
                     ds4_gpu_tensor *gate_base, ds4_gpu_tensor *up_base,
                     ds4_gpu_tensor *xq, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                     uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                     int force_kernel, float *out_gate, float *out_up, float *out_mid) {
    if (!ds4_gpu_test_moe_gate_up_mid_decode_lut_tensor(
            gate_out, up_out, mid_out, gate_base, up_base, xq, selected, weights,
            gate_expert_bytes, gate_row_bytes, XQ_BLOCKS, EXPERT_MID_DIM, N_EXPERT, N_TOK,
            1u, 0.0f, force_kernel)) {
        fprintf(stderr, "kernel launch failed (force_kernel=%d)\n", force_kernel);
        return 0;
    }
    const uint64_t n = (uint64_t)PAIR_COUNT * EXPERT_MID_DIM;
    if (!ds4_gpu_tensor_read(gate_out, 0, out_gate, n * sizeof(float)) ||
        !ds4_gpu_tensor_read(up_out, 0, out_up, n * sizeof(float)) ||
        !ds4_gpu_tensor_read(mid_out, 0, out_mid, n * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 0;
    }
    return 1;
}

int main(void) {
    const uint64_t gate_row_bytes = (uint64_t)XQ_BLOCKS * IQ2_BYTES;
    const uint64_t gate_expert_bytes = (uint64_t)EXPERT_MID_DIM * gate_row_bytes;
    const uint64_t gate_total = gate_expert_bytes * N_EXPERT;
    const uint64_t xq_total = (uint64_t)N_TOK * XQ_BLOCKS * Q8K_BYTES;
    const uint64_t n_pairs_rows = (uint64_t)PAIR_COUNT * EXPERT_MID_DIM;

    unsigned char *gate_host = malloc(gate_total);
    unsigned char *up_host = malloc(gate_total);
    unsigned char *xq_host = malloc(xq_total);
    int32_t *selected_host = malloc((uint64_t)PAIR_COUNT * sizeof(int32_t));
    float *weights_host = malloc((uint64_t)PAIR_COUNT * sizeof(float));
    float *hw_gate = malloc(n_pairs_rows * sizeof(float));
    float *hw_up = malloc(n_pairs_rows * sizeof(float));
    float *hw_mid = malloc(n_pairs_rows * sizeof(float));
    float *qw_gate = malloc(n_pairs_rows * sizeof(float));
    float *qw_up = malloc(n_pairs_rows * sizeof(float));
    float *qw_mid = malloc(n_pairs_rows * sizeof(float));
    if (!gate_host || !up_host || !xq_host || !selected_host || !weights_host ||
        !hw_gate || !hw_up || !hw_mid || !qw_gate || !qw_up || !qw_mid) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    for (uint32_t e = 0; e < N_EXPERT; e++) {
        make_iq2_row(gate_host + (uint64_t)e * gate_expert_bytes, EXPERT_MID_DIM * XQ_BLOCKS);
        make_iq2_row(up_host + (uint64_t)e * gate_expert_bytes, EXPERT_MID_DIM * XQ_BLOCKS);
    }
    for (uint32_t t = 0; t < N_TOK; t++) {
        for (uint32_t b = 0; b < XQ_BLOCKS; b++) {
            make_q8k_block(xq_host + ((uint64_t)t * XQ_BLOCKS + b) * Q8K_BYTES);
        }
    }
    for (uint32_t p = 0; p < PAIR_COUNT; p++) {
        selected_host[p] = (int32_t)(p % N_EXPERT);
        weights_host[p] = 0.5f + 0.5f * fabsf(next_uniform());
    }

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }

    ds4_gpu_tensor *gate_base = ds4_gpu_tensor_alloc(gate_total);
    ds4_gpu_tensor *up_base = ds4_gpu_tensor_alloc(gate_total);
    ds4_gpu_tensor *xq = ds4_gpu_tensor_alloc(xq_total);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)PAIR_COUNT * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)PAIR_COUNT * sizeof(float));
    ds4_gpu_tensor *gate_out = ds4_gpu_tensor_alloc(n_pairs_rows * sizeof(float));
    ds4_gpu_tensor *up_out = ds4_gpu_tensor_alloc(n_pairs_rows * sizeof(float));
    ds4_gpu_tensor *mid_out = ds4_gpu_tensor_alloc(n_pairs_rows * sizeof(float));
    if (!gate_base || !up_base || !xq || !selected || !weights || !gate_out || !up_out || !mid_out) {
        fprintf(stderr, "tensor alloc failed\n");
        return 1;
    }
    if (!ds4_gpu_tensor_write(gate_base, 0, gate_host, gate_total) ||
        !ds4_gpu_tensor_write(up_base, 0, up_host, gate_total) ||
        !ds4_gpu_tensor_write(xq, 0, xq_host, xq_total) ||
        !ds4_gpu_tensor_write(selected, 0, selected_host, (uint64_t)PAIR_COUNT * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host, (uint64_t)PAIR_COUNT * sizeof(float))) {
        fprintf(stderr, "tensor write failed\n");
        return 1;
    }

    if (!run_once(gate_out, up_out, mid_out, gate_base, up_base, xq, selected, weights,
                  gate_expert_bytes, gate_row_bytes, 2 /* hwarp32 */, hw_gate, hw_up, hw_mid))
        return 1;
    if (!run_once(gate_out, up_out, mid_out, gate_base, up_base, xq, selected, weights,
                  gate_expert_bytes, gate_row_bytes, 1 /* qwarp32 */, qw_gate, qw_up, qw_mid))
        return 1;

    double rms = 0.0;
    for (uint64_t i = 0; i < n_pairs_rows; i++) rms += (double)hw_mid[i] * hw_mid[i];
    rms = sqrt(rms / (double)n_pairs_rows);
    if (!(rms > 0.0)) { fprintf(stderr, "FAIL: mid output is all zeros\n"); return 1; }

    double worst_ref = 0.0, worst_qw = 0.0, worst_gate = 0.0, worst_up = 0.0;
    for (uint32_t p = 0; p < PAIR_COUNT; p++) {
        const uint32_t tok = p / N_EXPERT, slot = p - tok * N_EXPERT;
        (void)slot;
        const uint32_t expert = (uint32_t)selected_host[p];
        for (uint32_t row = 0; row < EXPERT_MID_DIM; row++) {
            const uint64_t off = (uint64_t)p * EXPERT_MID_DIM + row;
            const unsigned char *gr = gate_host + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
            const unsigned char *ur = up_host + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
            const unsigned char *xqb = xq_host + (uint64_t)tok * XQ_BLOCKS * Q8K_BYTES;
            const double ref = cpu_ref_mid(gr, ur, xqb, XQ_BLOCKS, weights_host[p]);
            const double d_ref = fabs((double)hw_mid[off] - ref);
            if (d_ref > worst_ref) worst_ref = d_ref;
            const double d_qw = fabs((double)hw_mid[off] - (double)qw_mid[off]);
            if (d_qw > worst_qw) worst_qw = d_qw;
            const double d_gate = fabs((double)hw_gate[off] - (double)qw_gate[off]);
            if (d_gate > worst_gate) worst_gate = d_gate;
            const double d_up = fabs((double)hw_up[off] - (double)qw_up[off]);
            if (d_up > worst_up) worst_up = d_up;
        }
    }

    printf("output rms                    : %.6f\n", rms);
    printf("max |hwarp32 - cpu ref|        : %.6f  (%.4f%% of rms)\n", worst_ref, 100.0 * worst_ref / rms);
    printf("max |hwarp32 - qwarp32| (mid)  : %.6f  (%.4f%% of rms)\n", worst_qw, 100.0 * worst_qw / rms);
    printf("max |hwarp32 - qwarp32| (gate) : %.6f\n", worst_gate);
    printf("max |hwarp32 - qwarp32| (up)   : %.6f\n", worst_up);

    int ok = 1;
    if (!(worst_ref < 1.0e-3 * rms)) { fprintf(stderr, "FAIL: hwarp32 disagrees with the CPU reference\n"); ok = 0; }
    if (!(worst_qw < 1.0e-3 * rms)) { fprintf(stderr, "FAIL: hwarp32 disagrees with qwarp32 (mid)\n"); ok = 0; }

    ds4_gpu_tensor_free(gate_base); ds4_gpu_tensor_free(up_base); ds4_gpu_tensor_free(xq);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(gate_out); ds4_gpu_tensor_free(up_out); ds4_gpu_tensor_free(mid_out);
    ds4_gpu_cleanup();
    free(gate_host); free(up_host); free(xq_host); free(selected_host); free(weights_host);
    free(hw_gate); free(hw_up); free(hw_mid); free(qw_gate); free(qw_up); free(qw_mid);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
