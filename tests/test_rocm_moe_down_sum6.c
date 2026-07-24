/* Numerical check for the wave64-native moe_down_sum6_hwarp32_kernel.
 *
 * moe_down_sum6_qwarp32_kernel is the routed-MoE decode-time down-projection
 * kernel (~4.4% of decode-phase GPU kernel time in this session's profiling,
 * never previously attempted). It uses 8-lane ("quarter warp") cooperation
 * per output row -- structurally identical in shape to
 * moe_gate_up_mid_decode_lut_qwarp32_kernel before its own wave64 fix (also
 * 8-lane originally), which won 24.5% after widening to 16-lane cooperation
 * plus halving rows-per-block/doubling the grid to preserve occupancy. This
 * kernel has a trivial single-scalar-store epilogue (no O(n^2) trap like
 * hc_expand hit) and, per source review, does NOT already use both wavefront
 * halves for two independent full-width rows the way matmul_q8_0_f32 does --
 * so it doesn't match either regression pattern from this session.
 *
 * moe_down_sum6_hwarp32_kernel applies the same fix: 16-lane cooperation
 * (half_warp_sum_f32), rows-per-block halved 32->16, grid doubled to match.
 *
 * midq_blocks is deliberately NOT a multiple of 16 (22) to stress the tail
 * of the strided per-lane accumulation loop, since unlike the gate/up/mid
 * fix this kernel has no "% 16 == 0" dispatch gate -- it must be correct for
 * any block count.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define OUT_DIM      777u   /* deliberately not a multiple of 16/32 */
#define N_EXPERT     6u     /* matches this session's test model's routed count */
#define MIDQ_BLOCKS  22u    /* deliberately not a multiple of 16 */

#define Q2K_SCALES 16u
#define Q2K_QS     64u
#define Q2K_BLOCK_BYTES (Q2K_SCALES + Q2K_QS + 2u + 2u)   /* 84 */

#define Q8K_QS     256u
#define Q8K_BSUMS  16u
#define Q8K_BLOCK_BYTES (4u + Q8K_QS + Q8K_BSUMS * 2u)     /* 292 */

static uint32_t g_rng = 0x9e3779b9u;

static uint32_t next_rand(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

static float next_uniform(void) {
    return (float)((next_rand() >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
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
    } else if (exp == 0x1f) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* --- synthetic q2_K / q8_K block generation --- */

static void make_q2_K_block(unsigned char *blk) {
    unsigned char *scales = blk;
    unsigned char *qs = blk + Q2K_SCALES;
    for (uint32_t i = 0; i < Q2K_SCALES; i++) scales[i] = (unsigned char)(next_rand() & 0xffu);
    for (uint32_t i = 0; i < Q2K_QS; i++) qs[i] = (unsigned char)(next_rand() & 0xffu);
    const uint16_t d = f32_to_f16(0.01f + 0.02f * fabsf(next_uniform()));
    const uint16_t dmin = f32_to_f16(0.005f + 0.01f * fabsf(next_uniform()));
    memcpy(blk + Q2K_SCALES + Q2K_QS, &d, 2);
    memcpy(blk + Q2K_SCALES + Q2K_QS + 2, &dmin, 2);
}

static void make_q8_K_block(unsigned char *blk) {
    const float d = 0.001f + 0.002f * fabsf(next_uniform());
    memcpy(blk, &d, 4);
    int8_t *qs = (int8_t *)(blk + 4);
    for (uint32_t i = 0; i < Q8K_QS; i++) qs[i] = (int8_t)(int)(next_uniform() * 100.0f);
    int16_t *bsums = (int16_t *)(blk + 4 + Q8K_QS);
    for (uint32_t i = 0; i < Q8K_BSUMS; i++) bsums[i] = (int16_t)(int)(next_uniform() * 4000.0f);
}

/* --- CPU reference, mirrors dev_dot_q2_K_q8_K_block bit-for-bit --- */

static int32_t cpu_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (int i = 0; i < 16; i += 4) {
        for (int j = 0; j < 4; j++) {
            int32_t v = (q2[i + j] >> shift) & 0x03;
            sum += v * (int32_t)q8[i + j];
        }
    }
    return sum;
}

static double cpu_dot_q2_K_q8_K_block(const unsigned char *x_blk, const unsigned char *y_blk) {
    const uint8_t *scales = x_blk;
    const uint8_t *q2 = x_blk + Q2K_SCALES;
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, x_blk + Q2K_SCALES + Q2K_QS, 2);
    memcpy(&dmin_bits, x_blk + Q2K_SCALES + Q2K_QS + 2, 2);
    const float x_d = f16_to_f32(d_bits);
    const float x_dmin = f16_to_f32(dmin_bits);

    float y_d;
    memcpy(&y_d, y_blk, 4);
    const int8_t *q8 = (const int8_t *)(y_blk + 4);
    const int16_t *bsums = (const int16_t *)(y_blk + 4 + Q8K_QS);

    int32_t summs = 0;
    for (int j = 0; j < 16; j++) summs += bsums[j] * (int32_t)(scales[j] >> 4);
    const double dall = (double)y_d * (double)x_d;
    const double dmin = (double)y_d * (double)x_dmin;

    int32_t isum = 0;
    int is = 0;
    const uint8_t *q2p = q2;
    const int8_t *q8p = q8;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int32_t d0 = scales[is++] & 0x0f;
            isum += d0 * cpu_dot_q2_16(q2p, q8p, shift);
            int32_t d1 = scales[is++] & 0x0f;
            isum += d1 * cpu_dot_q2_16(q2p + 16, q8p + 16, shift);
            shift += 2;
            q8p += 32;
        }
        q2p += 32;
    }
    return dall * (double)isum - dmin * (double)summs;
}

static double cpu_ref_row(const unsigned char *down_base, const unsigned char *midq,
                           const int32_t *selected, uint32_t row) {
    double total = 0.0;
    for (uint32_t slot = 0; slot < N_EXPERT; slot++) {
        const int32_t expert_i = selected[slot];
        const unsigned char *wr = down_base +
            (uint64_t)expert_i * OUT_DIM * MIDQ_BLOCKS * Q2K_BLOCK_BYTES +
            (uint64_t)row * MIDQ_BLOCKS * Q2K_BLOCK_BYTES;
        const unsigned char *xq = midq + (uint64_t)slot * MIDQ_BLOCKS * Q8K_BLOCK_BYTES;
        double acc = 0.0;
        for (uint32_t b = 0; b < MIDQ_BLOCKS; b++) {
            acc += cpu_dot_q2_K_q8_K_block(wr + (uint64_t)b * Q2K_BLOCK_BYTES,
                                            xq + (uint64_t)b * Q8K_BLOCK_BYTES);
        }
        total += acc;
    }
    return total;
}

static int run_once(ds4_gpu_tensor *out_t, ds4_gpu_tensor *down_t, ds4_gpu_tensor *midq_t,
                     ds4_gpu_tensor *selected_t, uint64_t down_expert_bytes,
                     uint64_t down_row_bytes, float *out_host, int force_kernel) {
    if (!ds4_gpu_test_moe_down_sum6_tensor(out_t, down_t, midq_t, selected_t,
                                            down_expert_bytes, down_row_bytes,
                                            MIDQ_BLOCKS, OUT_DIM, N_EXPERT, force_kernel)) {
        fprintf(stderr, "moe_down_sum6 launch failed (force_kernel=%d)\n", force_kernel);
        return 0;
    }
    if (!ds4_gpu_tensor_read(out_t, 0, out_host, (uint64_t)OUT_DIM * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 0;
    }
    return 1;
}

int main(void) {
    const uint64_t down_row_bytes = (uint64_t)MIDQ_BLOCKS * Q2K_BLOCK_BYTES;
    const uint64_t down_expert_bytes = (uint64_t)OUT_DIM * down_row_bytes;
    const uint64_t down_bytes = (uint64_t)N_EXPERT * down_expert_bytes;
    const uint64_t midq_bytes = (uint64_t)N_EXPERT * MIDQ_BLOCKS * Q8K_BLOCK_BYTES;

    unsigned char *down_base = malloc(down_bytes);
    unsigned char *midq = malloc(midq_bytes);
    int32_t selected[N_EXPERT];
    float *out_hwarp32 = malloc((uint64_t)OUT_DIM * sizeof(float));
    float *out_qwarp32 = malloc((uint64_t)OUT_DIM * sizeof(float));
    if (!down_base || !midq || !out_hwarp32 || !out_qwarp32) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    for (uint32_t e = 0; e < N_EXPERT; e++) {
        unsigned char *eb = down_base + (uint64_t)e * down_expert_bytes;
        for (uint32_t r = 0; r < OUT_DIM; r++) {
            for (uint32_t b = 0; b < MIDQ_BLOCKS; b++) {
                make_q2_K_block(eb + (uint64_t)r * down_row_bytes + (uint64_t)b * Q2K_BLOCK_BYTES);
            }
        }
        selected[e] = (int32_t)e;
        for (uint32_t b = 0; b < MIDQ_BLOCKS; b++) {
            make_q8_K_block(midq + (uint64_t)e * MIDQ_BLOCKS * Q8K_BLOCK_BYTES + (uint64_t)b * Q8K_BLOCK_BYTES);
        }
    }

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }

    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc((uint64_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *down_t = ds4_gpu_tensor_alloc(down_bytes);
    ds4_gpu_tensor *midq_t = ds4_gpu_tensor_alloc(midq_bytes);
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc((uint64_t)N_EXPERT * sizeof(int32_t));
    if (!out_t || !down_t || !midq_t || !selected_t) { fprintf(stderr, "tensor alloc failed\n"); return 1; }

    if (!ds4_gpu_tensor_write(down_t, 0, down_base, down_bytes) ||
        !ds4_gpu_tensor_write(midq_t, 0, midq, midq_bytes) ||
        !ds4_gpu_tensor_write(selected_t, 0, selected, (uint64_t)N_EXPERT * sizeof(int32_t))) {
        fprintf(stderr, "tensor write failed\n");
        return 1;
    }

    if (!run_once(out_t, down_t, midq_t, selected_t, down_expert_bytes, down_row_bytes, out_hwarp32, 2)) return 1;
    if (!run_once(out_t, down_t, midq_t, selected_t, down_expert_bytes, down_row_bytes, out_qwarp32, 1)) return 1;

    double rms = 0.0;
    for (uint32_t i = 0; i < OUT_DIM; i++) rms += (double)out_hwarp32[i] * out_hwarp32[i];
    rms = sqrt(rms / (double)OUT_DIM);
    if (!(rms > 0.0)) { fprintf(stderr, "FAIL: output is all zeros\n"); return 1; }

    double worst_ref = 0.0, worst_qwarp32 = 0.0;
    for (uint32_t r = 0; r < OUT_DIM; r++) {
        const double ref = cpu_ref_row(down_base, midq, selected, r);
        const double d_ref = fabs((double)out_hwarp32[r] - ref);
        if (d_ref > worst_ref) worst_ref = d_ref;
        const double d_qw = fabs((double)out_hwarp32[r] - (double)out_qwarp32[r]);
        if (d_qw > worst_qwarp32) worst_qwarp32 = d_qw;
    }

    printf("output rms                       : %.4f\n", rms);
    printf("max |hwarp32 - cpu f64ref|       : %.6f  (%.4f%% of rms)\n",
           worst_ref, 100.0 * worst_ref / rms);
    printf("max |hwarp32 - qwarp32|          : %.6f  (%.4f%% of rms)\n",
           worst_qwarp32, 100.0 * worst_qwarp32 / rms);

    int ok = 1;
    if (!(worst_ref < 1.0e-3 * rms)) {
        fprintf(stderr, "FAIL: hwarp32 kernel disagrees with the CPU reference\n");
        ok = 0;
    }
    if (!(worst_qwarp32 < 1.0e-3 * rms)) {
        fprintf(stderr, "FAIL: hwarp32 kernel disagrees with the qwarp32 kernel\n");
        ok = 0;
    }

    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(down_t);
    ds4_gpu_tensor_free(midq_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_cleanup();
    free(down_base); free(midq); free(out_hwarp32); free(out_qwarp32);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
