// DS4 ROCm routed-MoE quantization/device helpers and kernels.
//
// Included from the ROCm backend translation unit before
// ds4_rocm_moe_launch.cuh so host policy/glue can keep using these static kernels directly.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_pack_half2_bits(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ static uint32_t dev_load_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ static int32_t dev_packed_i8_lane(int32_t v, uint32_t lane) {
    return (int32_t)(int8_t)((uint32_t)v >> (lane * 8u));
}

__device__ __forceinline__ static int32_t dev_dp4a_i8(int32_t a, int32_t b, int32_t acc) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        acc += dev_packed_i8_lane(a, i) * dev_packed_i8_lane(b, i);
    }
    return acc;
#else
    return __dp4a(a, b, acc);
#endif
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_signed_grid_i8x4(
        uint64_t grid,
        uint32_t signs,
        uint32_t first) {
    uint32_t packed = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t idx = first + i;
        int32_t v = (int32_t)((grid >> (idx * 8u)) & 0xffu);
        if (signs & (1u << idx)) v = -v;
        packed |= ((uint32_t)(uint8_t)(int8_t)v) << (i * 8u);
    }
    return (int32_t)packed;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t g0 = dev_iq2_signed_grid_i8x4(grid, signs, 0u);
    const int32_t g1 = dev_iq2_signed_grid_i8x4(grid, signs, 4u);
    acc = dev_dp4a_i8(g0, (int32_t)dev_load_u32_le((const uint8_t *)(q8 + 0)), acc);
    acc = dev_dp4a_i8(g1, (int32_t)dev_load_u32_le((const uint8_t *)(q8 + 4)), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (int32_t)((dev_load_u32_le(q2 + i) >> shift) & 0x03030303);
        const int32_t y = (int32_t)dev_load_u32_le((const uint8_t *)(q8 + i));
        sum = dev_dp4a_i8(v, y, sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const uint64_t g = grid[grid_idx];
    *w0 = dev_iq2_signed_grid_i8x4(g, s, 0u);
    *w1 = dev_iq2_signed_grid_i8x4(g, s, 4u);
}

__device__ __forceinline__ static void dev_iq2_load_shared_lut(
        uint64_t *grid,
        uint8_t *signs) {
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) signs[i] = cuda_ksigns_iq2xs[i];
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = dev_dp4a_i8(w[0], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 0)),  sumi);
        sumi = dev_dp4a_i8(w[1], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 4)),  sumi);
        sumi = dev_dp4a_i8(w[2], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 8)),  sumi);
        sumi = dev_dp4a_i8(w[3], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 12)), sumi);
        sumi = dev_dp4a_i8(w[4], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 16)), sumi);
        sumi = dev_dp4a_i8(w[5], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 20)), sumi);
        sumi = dev_dp4a_i8(w[6], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 24)), sumi);
        sumi = dev_dp4a_i8(w[7], (int32_t)dev_load_u32_le((const uint8_t *)(q8 + ib32 * 32u + 28)), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = dev_dp4a_i8(w[0], (int32_t)dev_load_u32_le((const uint8_t *)(q + 0)),  sumi);
            sumi = dev_dp4a_i8(w[1], (int32_t)dev_load_u32_le((const uint8_t *)(q + 4)),  sumi);
            sumi = dev_dp4a_i8(w[2], (int32_t)dev_load_u32_le((const uint8_t *)(q + 8)),  sumi);
            sumi = dev_dp4a_i8(w[3], (int32_t)dev_load_u32_le((const uint8_t *)(q + 12)), sumi);
            sumi = dev_dp4a_i8(w[4], (int32_t)dev_load_u32_le((const uint8_t *)(q + 16)), sumi);
            sumi = dev_dp4a_i8(w[5], (int32_t)dev_load_u32_le((const uint8_t *)(q + 20)), sumi);
            sumi = dev_dp4a_i8(w[6], (int32_t)dev_load_u32_le((const uint8_t *)(q + 24)), sumi);
            sumi = dev_dp4a_i8(w[7], (int32_t)dev_load_u32_le((const uint8_t *)(q + 28)), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_ROCM_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (int32_t)((dev_load_u32_le(qs + i) >> shift) & 0x0f0f0f0f);
        const int32_t y = (int32_t)dev_load_u32_le((const uint8_t *)(q8 + i));
        sum = dev_dp4a_i8(v, y, sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ static void dev_dot_q4_K_q8_K_block4(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}


__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 16);
#else
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 16);
#endif
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 8);
#else
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 8);
#endif
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t slot_i = selected[(uint64_t)tok * n_expert + slot];
    if (slot_i < 0) slot_i = 0;
    const uint32_t compact_slot = (uint32_t)slot_i;
    const char *gate_base = gate_slots[compact_slot];
    const char *up_base = up_slots[compact_slot];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_ptrs_split_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const uint8_t *pair_missing,
        uint32_t want_missing,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    if (!pair_missing || ((uint32_t)(pair_missing[pair] != 0) != want_missing)) return;
    int32_t slot_i = selected[(uint64_t)tok * n_expert + slot];
    if (slot_i < 0) slot_i = 0;
    const uint32_t compact_slot = (uint32_t)slot_i;
    const char *gate_base = gate_slots[compact_slot];
    const char *up_base = up_slots[compact_slot];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    dev_iq2_load_shared_lut(s_iq2_grid, s_iq2_signs);
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
    }
    __syncthreads();
    if (xq_blocks <= 16u) xqb = sxq;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t slot_i = selected[(uint64_t)tok * n_expert + slot];
    if (slot_i < 0) slot_i = 0;
    const uint32_t compact_slot = (uint32_t)slot_i;
    const char *gate_base = gate_slots[compact_slot];
    const char *up_base = up_slots[compact_slot];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    dev_iq2_load_shared_lut(s_iq2_grid, s_iq2_signs);
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
    }
    __syncthreads();
    if (xq_blocks <= 16u) xqb = sxq;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_total_expert] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

/* Keep pair order stable inside each expert bucket.  The MoE WMMA kernels are
 * row-position sensitive enough that atomic append order changes logits. */
__global__ static void moe_scatter_sorted_pairs_deterministic_kernel(
        uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    const uint32_t expert = (uint32_t)blockIdx.x;
    if (expert >= n_total_expert || threadIdx.x != 0u) return;
    uint32_t pos = offsets[expert];
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        int32_t expert_i = selected[pair];
        if (expert_i < 0) expert_i = 0;
        if ((uint32_t)expert_i == expert) sorted_pairs[pos++] = pair;
    }
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[n_total_expert] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total_expert) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    dev_iq2_load_shared_lut(s_iq2_grid, s_iq2_signs);
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    dev_iq2_load_shared_lut(s_iq2_grid, s_iq2_signs);
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    dev_iq2_load_shared_lut(s_iq2_grid, s_iq2_signs);
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static DS4_ROCM_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_ROCM_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_ROCM_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    __shared__ float s_gate[256];
    __shared__ float s_up[256];
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const char *gate_base = gate_slots[(uint32_t)expert_i];
    const char *up_base = up_slots[(uint32_t)expert_i];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        float gate = 0.0f;
        float up = 0.0f;
        if (row < expert_mid_dim) {
            const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)row * gate_row_bytes);
            const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)row * gate_row_bytes);
            for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
                up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
            }
        }
        s_gate[threadIdx.x] = gate;
        s_up[threadIdx.x] = up;
        __syncthreads();
        if (row < expert_mid_dim && lane == 0) {
            #pragma unroll
            for (uint32_t i = 1u; i < 8u; i++) {
                gate += s_gate[threadIdx.x + i];
                up += s_up[threadIdx.x + i];
            }
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
        __syncthreads();
    }
}

__global__ static void moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 16u; rr++) {
        const uint32_t row = blockIdx.x * 256u + row_lane + rr * 16u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 16u) {
            gate += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        gate = half_warp_sum_f32(gate, lane);
        up = half_warp_sum_f32(up, lane);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sum6_qwarp32_ptrs_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const char *down_base = down_slots[slot];
        if (!down_base) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sum6_qwarp32_ptrs_batch_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t compact_i = selected[(uint64_t)tok * n_expert + slot];
        if (compact_i < 0) compact_i = 0;
        const char *down_base = down_slots[(uint32_t)compact_i];
        if (!down_base) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[(uint64_t)tok * out_dim + row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_ptrs_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    __shared__ float s_acc[256];
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const char *down_base = down_slots[(uint32_t)expert_i];
        float acc = 0.0f;
        if (row < out_dim && down_base) {
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)row * down_row_bytes);
            const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        s_acc[threadIdx.x] = acc;
        __syncthreads();
        if (row < out_dim && lane == 0) {
            #pragma unroll
            for (uint32_t i = 1u; i < 8u; i++) acc += s_acc[threadIdx.x + i];
            total += acc;
        }
        __syncthreads();
    }
    if (row < out_dim && lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_qwarp32_ptrs_kernel(
        float *down_out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    __shared__ float s_acc[256];
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const char *down_base = down_slots[(uint32_t)expert_i];
    float acc = 0.0f;
    if (row < out_dim && down_base) {
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    }
    s_acc[threadIdx.x] = acc;
    __syncthreads();
    if (row < out_dim && lane == 0) {
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) acc += s_acc[threadIdx.x + i];
        down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_q4K_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_q4K_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_ROCM_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__global__ static void moe_sum_f16_kernel(float *out, const half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += __half2float(down_h[((uint64_t)tok * n_expert + e) * out_dim + row]);
    out[gid] = acc;
}

__global__ static void moe_sum_f16x2_kernel(float *out, const half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t out_dim2 = out_dim >> 1u;
    const uint64_t n2 = (uint64_t)n_tokens * out_dim2;
    if (gid >= n2) return;
    const uint32_t tok = gid / out_dim2;
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim2) << 1u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint64_t off = ((uint64_t)tok * n_expert + e) * out_dim + row;
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(down_h + off));
        acc0 += v.x;
        acc1 += v.y;
    }
    const uint64_t out_off = (uint64_t)tok * out_dim + row;
    out[out_off] = acc0;
    out[out_off + 1u] = acc1;
}

__device__ __forceinline__ static void q2_K_scale_broadcast_w32(const unsigned char *blk, float *d, float *dmin) {
    float vd = 0.0f;
    float vm = 0.0f;
    if ((threadIdx.x & 31u) == 0u) {
        const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
        const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
        vd = dev_f16_to_f32(d_bits);
        vm = dev_f16_to_f32(dmin_bits);
    }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    *d = __shfl(vd, 0, 32);
    *dmin = __shfl(vm, 0, 32);
#else
    *d = __shfl_sync(FULL_WARP_MASK, vd, 0, 32);
    *dmin = __shfl_sync(FULL_WARP_MASK, vm, 0, 32);
#endif
}

__device__ __forceinline__ static float q2_K_dequant_256_scaled_w32(
        const unsigned char *blk,
        uint32_t lane,
        uint32_t kk,
        float d,
        float dmin) {
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = (lane >> 4u) + (kk << 1u);
    const uint32_t within = g & 7u;
    const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (lane & 15u);
    const uint32_t shift = (within >> 1u) * 2u;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4u);
    return d * scale * q - dmin * mn;
}


__device__ __forceinline__ static float q2_K_dequant_256_direct(const unsigned char *blk, uint32_t i) {
    const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
    const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = i >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (i & 15u);
    const uint32_t shift = (within >> 1u) * 2u;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4u);
    return dev_f16_to_f32(d_bits) * scale * q - dev_f16_to_f32(dmin_bits) * mn;
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_tile_half_rowwise(
        half *shB,
        const unsigned char *base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 2u;
    for (uint32_t j = tid; j < (uint32_t)(BN * (BK / KG)); j += blockDim.x) {
        const uint32_t nn = j / (uint32_t)(BK / KG);
        const uint32_t kk0 = (j - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + nn;
        if (row < out_dim) {
            const unsigned char *blk = base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d = dev_f16_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
            const float dm = dev_f16_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
            const float s = (float)(blk[g] & 0x0fu);
            const float m = (float)(blk[g] >> 4u);
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                const float q = (float)((blk[16u + qbase + kk] >> shift) & 3u);
                shB[kk * (uint32_t)BN + nn] = __float2half(d * s * q - dm * m);
            }
        } else {
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                shB[kk * (uint32_t)BN + nn] = __float2half(0.0f);
            }
        }
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_dual_tile_half_rowwise(
        half *shB0,
        half *shB1,
        const unsigned char *base0,
        const unsigned char *base1,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 2u;
    for (uint32_t j = tid; j < (uint32_t)(BN * (BK / KG)); j += blockDim.x) {
        const uint32_t nn = j / (uint32_t)(BK / KG);
        const uint32_t kk0 = (j - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + nn;
        if (row < out_dim) {
            const unsigned char *blk0 = base0 + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const unsigned char *blk1 = base1 + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d0 = dev_f16_to_f32((uint16_t)blk0[80] | ((uint16_t)blk0[81] << 8));
            const float dm0 = dev_f16_to_f32((uint16_t)blk0[82] | ((uint16_t)blk0[83] << 8));
            const float d1 = dev_f16_to_f32((uint16_t)blk1[80] | ((uint16_t)blk1[81] << 8));
            const float dm1 = dev_f16_to_f32((uint16_t)blk1[82] | ((uint16_t)blk1[83] << 8));
            const float s0 = (float)(blk0[g] & 0x0fu);
            const float m0 = (float)(blk0[g] >> 4u);
            const float s1 = (float)(blk1[g] & 0x0fu);
            const float m1 = (float)(blk1[g] >> 4u);
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                const float q0 = (float)((blk0[16u + qbase + kk] >> shift) & 3u);
                const float q1 = (float)((blk1[16u + qbase + kk] >> shift) & 3u);
                const uint32_t sj = kk * (uint32_t)BN + nn;
                shB0[sj] = __float2half(d0 * s0 * q0 - dm0 * m0);
                shB1[sj] = __float2half(d1 * s1 * q1 - dm1 * m1);
            }
        } else {
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                const uint32_t sj = kk * (uint32_t)BN + nn;
                shB0[sj] = __float2half(0.0f);
                shB1[sj] = __float2half(0.0f);
            }
        }
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_pair_tile_half_rowwise(
        half *shB0,
        half *shB1,
        const unsigned char *base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        half *shB = tile == 0u ? shB0 : shB1;
        uint32_t v0 = 0u;
        uint32_t v1 = 0u;
        if (row < out_dim) {
            const unsigned char *blk = base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d = dev_f16_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
            const float dm = dev_f16_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
            const float s = (float)(blk[g] & 0x0fu);
            const float m = (float)(blk[g] >> 4u);
            const uint32_t qbits = *reinterpret_cast<const uint32_t *>(blk + 16u + qbase + kk0);
            const uint32_t q0 = (qbits >> shift) & 3u;
            const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
            const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
            const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
            const float ds = d * s;
            const float dmm = dm * m;
            v0 = dev_pack_half2_bits(ds * (float)q0 - dmm,
                                     ds * (float)q1 - dmm);
            v1 = dev_pack_half2_bits(ds * (float)q2 - dmm,
                                     ds * (float)q3 - dmm);
        }
        half *dst = shB + nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(dst) = v0;
        *reinterpret_cast<uint32_t *>(dst + 2u) = v1;
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_dual_pair_tile_half_rowwise(
        half *shB0g,
        half *shB0u,
        half *shB1g,
        half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        half *shBg = tile == 0u ? shB0g : shB1g;
        half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u;
        uint32_t gv1 = 0u;
        uint32_t uv0 = 0u;
        uint32_t uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float gd = dev_f16_to_f32((uint16_t)gb[80] | ((uint16_t)gb[81] << 8));
            const float gm = dev_f16_to_f32((uint16_t)gb[82] | ((uint16_t)gb[83] << 8));
            const float ud = dev_f16_to_f32((uint16_t)ub[80] | ((uint16_t)ub[81] << 8));
            const float um = dev_f16_to_f32((uint16_t)ub[82] | ((uint16_t)ub[83] << 8));
            const float gs = (float)(gb[g] & 0x0fu);
            const float gmn = (float)(gb[g] >> 4u);
            const float us = (float)(ub[g] & 0x0fu);
            const float umn = (float)(ub[g] >> 4u);
            const uint32_t gqbits = *reinterpret_cast<const uint32_t *>(gb + 16u + qbase + kk0);
            const uint32_t uqbits = *reinterpret_cast<const uint32_t *>(ub + 16u + qbase + kk0);
            const uint32_t gq0 = (gqbits >> shift) & 3u;
            const uint32_t gq1 = (gqbits >> (8u + shift)) & 3u;
            const uint32_t gq2 = (gqbits >> (16u + shift)) & 3u;
            const uint32_t gq3 = (gqbits >> (24u + shift)) & 3u;
            const uint32_t uq0 = (uqbits >> shift) & 3u;
            const uint32_t uq1 = (uqbits >> (8u + shift)) & 3u;
            const uint32_t uq2 = (uqbits >> (16u + shift)) & 3u;
            const uint32_t uq3 = (uqbits >> (24u + shift)) & 3u;
            const float gds = gd * gs;
            const float gmm = gm * gmn;
            const float uds = ud * us;
            const float umm = um * umn;
            gv0 = dev_pack_half2_bits(gds * (float)gq0 - gmm,
                                      gds * (float)gq1 - gmm);
            gv1 = dev_pack_half2_bits(gds * (float)gq2 - gmm,
                                      gds * (float)gq3 - gmm);
            uv0 = dev_pack_half2_bits(uds * (float)uq0 - umm,
                                      uds * (float)uq1 - umm);
            uv1 = dev_pack_half2_bits(uds * (float)uq2 - umm,
                                      uds * (float)uq3 - umm);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(shBg + sj) = gv0;
        *reinterpret_cast<uint32_t *>(shBg + sj + 2u) = gv1;
        *reinterpret_cast<uint32_t *>(shBu + sj) = uv0;
        *reinterpret_cast<uint32_t *>(shBu + sj + 2u) = uv1;
    }
}

__device__ __forceinline__ static float moe_silu_oldhip(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ __launch_bounds__(32) static void moe_gate_up_mid_q2K_rows_rpb1_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim || lane >= 32u) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_rows_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_q2K_sum_rows_w32_kernel(
        float *out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    float acc = 0.0f;
    const uint32_t nb = expert_mid_dim >> 8u;
#pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[(uint64_t)tok * 6u + slot];
        if (expert_i < 0) expert_i = 0;
        const unsigned char *dr = (const unsigned char *)down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes;
        const float *mr = mid + ((uint64_t)tok * 6u + slot) * expert_mid_dim;
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *db = dr + (uint64_t)b * 84u;
            float d, dmin;
            q2_K_scale_broadcast_w32(db, &d, &dmin);
            const uint64_t mbase = (uint64_t)b * 256u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5u);
                acc += q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin) * mr[mbase + i];
            }
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

template <uint32_t PAIR_TILE, bool OUT_F16=false>
__global__ static void moe_gate_up_mid_q2K_expert_batch_sharedx_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    const bool row_valid = row < expert_mid_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *grow = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const unsigned char *urow = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t xbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint32_t tok = pair[u] / 6u;
                    shx[j] = x[(uint64_t)tok * expert_in_dim + xbase + k];
                } else {
                    shx[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *gb = grow + (uint64_t)b * 84u;
                const unsigned char *ub = urow + (uint64_t)b * 84u;
                float gd, gdmin, ud, udmin;
                q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
                q2_K_scale_broadcast_w32(ub, &ud, &udmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float gwv = q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin);
                    const float uwv = q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) {
                        const float xv = shx[(u << 8u) + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            g_acc[u] = warp_sum_f32(g_acc[u]);
            u_acc[u] = warp_sum_f32(u_acc[u]);
        }
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    float g = g_acc[u];
                    float upv = u_acc[u];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (upv > clamp) upv = clamp;
                        if (upv < -clamp) upv = -clamp;
                    }
                    const float v = moe_silu_oldhip(g) * upv * weights[pair[u]];
                    if (OUT_F16) mid_out_h[(uint64_t)pair[u] * expert_mid_dim + row] = __float2half(v);
                    else mid_out[(uint64_t)pair[u] * expert_mid_dim + row] = v;
                }
            }
        }
    }
}

template <uint32_t PAIR_TILE, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_expert_batch_sharedmid_kernel(
        float *down_out,
        half *down_out_h,
        const char *down_base,
        const float *mid,
        const half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_tokens = 0u) {
    extern __shared__ float shmid[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    const bool row_valid = row < out_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *drow = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)(row_valid ? row : 0u) * down_row_bytes;
    const uint32_t nb = expert_mid_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t mbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint64_t moff = (uint64_t)pair[u] * expert_mid_dim + mbase + k;
                    shmid[j] = MID_F16 ? __half2float(mid_h[moff]) : mid[moff];
                } else {
                    shmid[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *db = drow + (uint64_t)b * 84u;
                float d, dmin;
                q2_K_scale_broadcast_w32(db, &d, &dmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float wv = q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] += wv * shmid[(u << 8u) + i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    if (OUT_F16) {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / 6u;
                            const uint32_t slot = pair[u] - tok * 6u;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out_h[dst] = __float2half(acc[u]);
                    } else {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / 6u;
                            const uint32_t slot = pair[u] - tok * 6u;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out[dst] = acc[u];
                    }
                }
            }
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
__device__ __forceinline__ static float iq2_xxs_dequant_256_direct(const unsigned char *blk, uint32_t i) {
    const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const uint16_t *q2 = reinterpret_cast<const uint16_t *>(blk + 2u);
    const uint32_t ib32 = i >> 5u;
    const uint32_t j = i & 31u;
    const uint32_t aux_g = (uint32_t)q2[ib32 * 4u + 0u] | ((uint32_t)q2[ib32 * 4u + 1u] << 16);
    const uint32_t aux_s = (uint32_t)q2[ib32 * 4u + 2u] | ((uint32_t)q2[ib32 * 4u + 3u] << 16);
    const uint32_t half = j >> 4u;
    const uint32_t g = (j >> 3u) & 1u;
    const uint32_t ii = j & 7u;
    const uint32_t grid_idx = (aux_g >> (8u * ((half << 1u) + g))) & 0xffu;
    const uint32_t sign_idx = (aux_s >> (14u * half + 7u * g)) & 127u;
    const uint64_t grid = cuda_iq2xxs_grid[grid_idx];
    const uint32_t signs = dev_unpack_iq2_signs(cuda_ksigns_iq2xs[sign_idx]);
    float w = (float)((grid >> (8u * ii)) & 0xffu);
    if (signs & (1u << ii)) w = -w;
    return dev_f16_to_f32(d_bits) * (0.125f + 0.25f * (float)(aux_s >> 28u)) * w;
}

template <int BN, int BK>
__device__ __forceinline__ static void iq2_xxs_dequant_dual_pair_tile_half_rowwise(
        half *shB0g,
        half *shB0u,
        half *shB1g,
        half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        half *shBg = tile == 0u ? shB0g : shB1g;
        half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u, gv1 = 0u, uv0 = 0u, uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const uint32_t kk_base = k0 & 255u;
            const float g0 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 0u);
            const float g1 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 1u);
            const float g2 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 2u);
            const float g3 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 3u);
            const float u0 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 0u);
            const float u1 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 1u);
            const float u2 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 2u);
            const float u3 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 3u);
            gv0 = dev_pack_half2_bits(g0, g1);
            gv1 = dev_pack_half2_bits(g2, g3);
            uv0 = dev_pack_half2_bits(u0, u1);
            uv1 = dev_pack_half2_bits(u2, u3);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(shBg + sj) = gv0;
        *reinterpret_cast<uint32_t *>(shBg + sj + 2u) = gv1;
        *reinterpret_cast<uint32_t *>(shBu + sj) = uv0;
        *reinterpret_cast<uint32_t *>(shBu + sj + 2u) = uv1;
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shBg0 = shA + MTILES * BM * BK;
    half *shBu0 = shBg0 + BK * BN;
    half *shBg1 = shBu0 + BK * BN;
    half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0, bu0, bg1, bu1;
    frag_c accg0, accu0, accg1, accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t pair_row = j / BK;
                const uint32_t kk = j - pair_row * BK;
                const uint32_t pair = shPair[pair_row];
                shA[j] = pair != UINT32_MAX
                    ? __float2half(x[(uint64_t)(pair / 6u) * expert_in_dim + k0 + kk])
                    : __float2half(0.0f);
            }
        }
        iq2_xxs_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j], u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j], u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shBg0 = shA + MTILES * BM * BK;
    half *shBu0 = shBg0 + BK * BN;
    half *shBg1 = shBu0 + BK * BN;
    half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0;
    frag_b bu0;
    frag_b bg1;
    frag_b bu1;
    frag_c accg0;
    frag_c accu0;
    frag_c accg1;
    frag_c accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t mt = j / (BM * BK);
                const uint32_t rem = j - mt * BM * BK;
                const uint32_t mm = rem / BK;
                const uint32_t kk = rem - mm * BK;
                const uint32_t pair = shPair[mt * BM + mm];
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    shA[j] = __float2half(x[(uint64_t)token * expert_in_dim + k0 + kk]);
                } else {
                    shA[j] = __float2half(0.0f);
                }
            }
        }
        q2_K_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j];
                float u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j];
                float u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16>
__global__ static void moe_down_q2K_hotlist_wmma_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB = shA + MTILES * BM * BK;
    float *shC = reinterpret_cast<float *>(shB + BK * BN);
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_mid_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const uint32_t mt = j / (BM * BK);
            const uint32_t rem = j - mt * BM * BK;
            const uint32_t mm = rem / BK;
            const uint32_t kk = rem - mm * BK;
            const uint32_t bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < count) {
                const uint32_t pair = pairs[first + bucket_row];
                shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        q2_K_dequant_tile_half_rowwise<BN, BK>(
                shB, dew, down_row_bytes, n0, k0, out_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < MTILES) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t bucket_row = m_group0 + mt * BM + mm;
        const uint32_t row = n0 + nn;
        if (bucket_row < count && row < out_dim) {
            const uint32_t pair = pairs[first + bucket_row];
            down_out[(uint64_t)pair * out_dim + row] = shC[j];
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_hotlist_wmma_n2_kernel(
        float *down_out,
        half *down_out_h,
        const char *down_base,
        const float *mid,
        const half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_tokens = 0u) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB0 = shA + MTILES * BM * BK;
    half *shB1 = shB0 + BK * BN;
    float *shC0 = reinterpret_cast<float *>(shB1 + BK * BN);
    float *shC1 = shC0 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b0;
    frag_b b1;
    frag_c acc0;
    frag_c acc1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(acc0, 0.0f);
        rocwmma::fill_fragment(acc1, 0.0f);
    }

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_mid_dim; k0 += BK) {
        if (MID_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint64_t moff = (uint64_t)pair * expert_mid_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(mid_h + moff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t mt = j / (BM * BK);
                const uint32_t rem = j - mt * BM * BK;
                const uint32_t mm = rem / BK;
                const uint32_t kk = rem - mm * BK;
                const uint32_t pair = shPair[mt * BM + mm];
                if (pair != UINT32_MAX) {
                    shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
                } else {
                    shA[j] = __float2half(0.0f);
                }
            }
        }
        q2_K_dequant_pair_tile_half_rowwise<BN, BK>(
                shB0, shB1, dew, down_row_bytes, n0, k0, out_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b0, shB0, BN);
            rocwmma::load_matrix_sync(b1, shB1, BN);
            rocwmma::mma_sync(acc0, a, b0, acc0);
            rocwmma::mma_sync(acc1, a, b1, acc1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC0 + wave * BM * BN, acc0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shC1 + wave * BM * BN, acc1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const uint32_t tok = pair / 6u;
            const uint32_t slot = pair - tok * 6u;
            if (row0 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out_h[dst] = __float2half(shC0[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out[dst] = shC0[j];
                }
            }
            if (row1 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out_h[dst] = __float2half(shC1[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out[dst] = shC1[j];
                }
            }
        }
    }
}

#endif

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_q2_K_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_q2_K_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}
