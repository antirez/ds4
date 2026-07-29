
__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
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
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
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
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
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
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
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

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
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
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
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

/* Vector-load variant of dev_dot_q4_K_q8_K_block: loads the whole 144-byte
 * Q4_K block with nine 16B loads (requires a 16B-aligned tensor base; block
 * stride 144 and row strides are 16B multiples), then computes the exact same
 * integer sums and float finish. Same values in the same order, so results
 * are bit-identical; the wide loads just improve DRAM/memory-level
 * parallelism for the bandwidth-bound decode matvecs. */
__device__ __forceinline__ static void dev_dot_q4_K_q8_K_block_vec(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y,
        float *out_acc) {
    const uint4 hdr = *(const uint4 *)x; /* d, dmin, scales[12] */
    uint4 qv[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) qv[i] = ((const uint4 *)(x->qs))[i];
    const uint16_t xd_u = (uint16_t)(hdr.x & 0xffffu);
    const uint16_t xmin_u = (uint16_t)(hdr.x >> 16u);
    uint8_t scales[12];
    scales[0] = (uint8_t)(hdr.y);
    scales[1] = (uint8_t)(hdr.y >> 8);
    scales[2] = (uint8_t)(hdr.y >> 16);
    scales[3] = (uint8_t)(hdr.y >> 24);
    scales[4] = (uint8_t)(hdr.z);
    scales[5] = (uint8_t)(hdr.z >> 8);
    scales[6] = (uint8_t)(hdr.z >> 16);
    scales[7] = (uint8_t)(hdr.z >> 24);
    scales[8] = (uint8_t)(hdr.w);
    scales[9] = (uint8_t)(hdr.w >> 8);
    scales[10] = (uint8_t)(hdr.w >> 16);
    scales[11] = (uint8_t)(hdr.w >> 24);
    const float xd = dev_f16_to_f32(xd_u);
    const float xmin = dev_f16_to_f32(xmin_u);
    int isum = 0;
    int summs = 0;
    const int32_t *qw = (const int32_t *)qv;
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t word_off = (j >> 1u) * 8u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t sum = 0;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const int32_t v = (qw[word_off + i] >> shift) & 0x0f0f0f0f;
            sum = __dp4a(v, *(const int32_t *)(y->qs + j * 32u + i * 4u), sum);
        }
        isum += (int)sc * sum;
    }
    *out_acc += y->d * xd * (float)isum - y->d * xmin * (float)summs;
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
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }

    for (uint32_t p = 0; p < n; p++) {
        acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
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
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
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

/* Decode-only dual quantizer.  The Q8_0 half mirrors
 * quantize_q8_0_f32_kernel's 32-thread reduction and expression order, while
 * the Q8_K half remains byte-for-byte the ordinary routed-MoE quantizer. */
__global__ static void q8_K_q8_0_quantize_kernel(
        cuda_block_q8_K *out,
        int8_t *q8_0,
        float *q8_0_scale,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim +
                      (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out +
        (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;

    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            abs_part[tid] = fmaxf(abs_part[tid], abs_part[tid + stride]);
        }
        __syncthreads();
    }
    const uint32_t q8_blocks = in_dim / 32u;
    const uint32_t q8_block = b * 8u + warp;
    const float d = abs_part[warp * 32u] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) {
        q8_0_scale[(uint64_t)row * q8_blocks + q8_block] = d;
    }
    int qv = (int)lrintf(v * id);
    qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
    q8_0[((uint64_t)row * q8_blocks + q8_block) * 32u + lane] =
        (int8_t)qv;
    __syncthreads();

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
    const float amax = abs_part[0];
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
        int kv = (int)lrintf(iscale_s * xr[tid]);
        if (kv > 127) kv = 127;
        if (kv < -128) kv = -128;
        yb->qs[tid] = (int8_t)kv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__device__ __forceinline__ static bool moe_owned_local_expert(
        int32_t expert,
        uint32_t expert_base,
        uint32_t expert_count,
        uint32_t *local_expert) {
    if (expert < 0) return false;
    const uint32_t e = (uint32_t)expert;
    if (e < expert_base || e - expert_base >= expert_count) return false;
    if (local_expert) *local_expert = e - expert_base;
    return true;
}

/* Quantize only selected slots owned by this expert-parallel rank.  Rows keep
 * their original slot index so the final rank-local reduction can visit slots
 * in canonical order without compaction or a host synchronization. */
__global__ static void q8_K_quantize_owned_kernel(
        cuda_block_q8_K *out,
        const float *x,
        const int32_t *selected,
        uint32_t in_dim,
        uint32_t n_rows,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    if (!moe_owned_local_expert(selected[row], expert_base, expert_count, NULL)) return;

    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
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
    const float amax = abs_part[0];
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

__global__ static void moe_filter_owned_pairs_kernel(
        int32_t *selected,
        float *weights,
        uint64_t pair_count,
        uint32_t n_total_expert,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;
    const int32_t expert_i = selected[pair];
    if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert &&
        (uint32_t)expert_i >= expert_base &&
        (uint32_t)expert_i - expert_base < expert_count) {
        selected[pair] = expert_i - (int32_t)expert_base;
    } else {
        selected[pair] = -1;
        weights[pair] = 0.0f;
    }
}

__global__ static void q8_K_quantize_sidecar_kernel(
        cuda_block_q8_K *out,
        const float *x,
        const float *amax_sidecar,
        uint32_t in_dim,
        uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    const uint32_t blocks = in_dim / CUDA_QK_K;
    if (row >= n_rows || b >= blocks) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    const float *sc = amax_sidecar + ((uint64_t)row * blocks + b) * 32u;
    cuda_block_q8_K *yb = out + (uint64_t)row * blocks + b;
    __shared__ float abs_part[32];
    __shared__ float val_part[32];
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    if (tid < 32u) {
        const float v = sc[tid];
        abs_part[tid] = fabsf(v);
        val_part[tid] = v;
    }
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0u) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16u) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0u) iscale_s = -127.0f / val_part[0];
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16u) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16u + (uint32_t)i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
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

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
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

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
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

// perf-04: launch-geometry tuning for the routed-MoE gate/up decode kernels
// (moe_gate_up_mid_qwarp32 / _decode_lut_qwarp32 / _decode_q4K_qwarp32). Each
// block processes MOE_DECODE_ROW_TILES tiles of 32 rows (row_lane in [0,32)).
// The historical value was 4 (128 rows/block -> ~96 blocks, occupancy ~16%,
// "grid too small to fill the device"). Lowering it issues correspondingly more
// blocks (e.g. 1 tile -> 32 rows/block -> ~4x more blocks -> ~384) to fill the
// SMs. The per-row arithmetic is identical regardless of this value, so output
// is bit-identical; only the qgrid.x divisor must match MOE_DECODE_ROWS_PER_BLOCK.
#ifndef MOE_DECODE_ROW_TILES
#define MOE_DECODE_ROW_TILES 1u
#endif
#define MOE_DECODE_ROWS_PER_BLOCK (32u * MOE_DECODE_ROW_TILES)

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
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
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
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
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

__global__ static void moe_gate_up_mid_decode_lut_owned_qwarp32_kernel(
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
        uint32_t expert_base,
        uint32_t expert_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t expert = 0u;
    if (!moe_owned_local_expert(selected[pair], expert_base, expert_count,
                                &expert)) return;
    const cuda_block_q8_K *xqb = xq;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
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
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
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
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
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
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
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

/* Decode-sized routed batches spend more host time launching metadata kernels
 * than doing the <= 96 pair / 128 expert setup. Build both tile lists in one
 * deterministic block; the expensive expert kernels remain unchanged. */
__global__ static void moe_prepare_sorted_tiles_small_kernel(
        uint32_t *counts,
        uint32_t *offsets,
        uint32_t *cursors,
        uint32_t *sorted_pairs,
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        uint32_t *tile16_offsets,
        uint32_t *tile16_total,
        uint32_t *tile16_experts,
        uint32_t *tile16_starts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert,
        uint32_t block_m,
        bool build_tile16) {
    if (blockIdx.x != 0) return;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t local_counts[128];
    __shared__ int32_t local_selected[96];

    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        local_counts[e] = 0u;
    }
    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        local_selected[pair] = selected[pair];
    }
    __syncthreads();

    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        const int32_t expert_i = local_selected[pair];
        if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert) {
            atomicAdd(local_counts + (uint32_t)expert_i, 1u);
        }
    }
    __syncthreads();

    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        counts[e] = local_counts[e];
    }
    if (tid == 0u) {
        uint32_t pair_sum = 0u;
        uint32_t tile_sum = 0u;
        uint32_t tile16_sum = 0u;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t count = local_counts[e];
            offsets[e] = pair_sum;
            pair_sum += count;
            cursors[e] = pair_sum;
            tile_offsets[e] = tile_sum;
            tile_sum += (count + block_m - 1u) / block_m;
            if (build_tile16) {
                tile16_offsets[e] = tile16_sum;
                tile16_sum += (count + 15u) / 16u;
            }
        }
        offsets[n_total_expert] = pair_sum;
        tile_offsets[n_total_expert] = tile_sum;
        *tile_total = tile_sum;
        if (build_tile16) {
            tile16_offsets[n_total_expert] = tile16_sum;
            *tile16_total = tile16_sum;
        }
    }
    __syncthreads();

    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        const int32_t expert_i = local_selected[pair];
        if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert) {
            uint32_t rank = 0u;
            for (uint32_t prev = 0; prev < pair; prev++) {
                rank += local_selected[prev] == expert_i;
            }
            sorted_pairs[offsets[(uint32_t)expert_i] + rank] = pair;
        }
    }
    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        const uint32_t count = local_counts[e];
        const uint32_t ntiles = (count + block_m - 1u) / block_m;
        const uint32_t tile_off = tile_offsets[e];
        for (uint32_t t = 0; t < ntiles; t++) {
            tile_experts[tile_off + t] = e;
            tile_starts[tile_off + t] = t * block_m;
        }
        if (build_tile16) {
            const uint32_t ntiles16 = (count + 15u) / 16u;
            const uint32_t tile16_off = tile16_offsets[e];
            for (uint32_t t = 0; t < ntiles16; t++) {
                tile16_experts[tile16_off + t] = e;
                tile16_starts[tile16_off + t] = t * 16u;
            }
        }
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

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
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
        if (local_pair >= counts[expert]) break;
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
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
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
        if (local_pair >= counts[expert]) break;
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
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
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
        if (local_pair >= counts[expert]) break;
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
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
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

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
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

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
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

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
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
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
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

__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_kernel(
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
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_row8_kernel(
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
    uint32_t lane = threadIdx.x & 15u;
    uint32_t group = threadIdx.x >> 4u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    if (group >= 8u) return;
    uint32_t row = blockIdx.x * 8u + group;
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_warp32_kernel(
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
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_kernel(
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
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        /* Word-wise cooperative staging copy (same bytes, all lanes busy). */
        const uint32_t words = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t *srcw = (const uint32_t *)xqb;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = srcw[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    const bool vec_ok = ((((uintptr_t)gate_base | (uintptr_t)up_base |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u);
    if (vec_ok) {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            dev_dot_q4_K_q8_K_block_vec(gr + b, xqb + b, &gate);
            dev_dot_q4_K_q8_K_block_vec(ur + b, xqb + b, &up);
        }
    } else {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_owned_warp32_noaux_kernel(
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
        uint32_t expert_base,
        uint32_t expert_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t expert = 0u;
    if (!moe_owned_local_expert(selected[pair], expert_base, expert_count,
                                &expert)) return;
    const cuda_block_q8_K *xqb = xq;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        const uint32_t words = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t *srcw = (const uint32_t *)xqb;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = srcw[i];
        __syncthreads();
        xqb = sxq;
    }
    const bool vec_ok = ((((uintptr_t)gate_base | (uintptr_t)up_base |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u);
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    if (vec_ok) {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            dev_dot_q4_K_q8_K_block_vec(gr + b, xqb + b, &gate);
            dev_dot_q4_K_q8_K_block_vec(ur + b, xqb + b, &up);
        }
    } else {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_sidecar_kernel(
        float *mid_out,
        float *amax_sidecar,
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
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * 8u + warp;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ float tile_vals[8];
    __shared__ float tile_abs[8];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }

    float midv = 0.0f;
    const bool valid = row < expert_mid_dim;
    if (valid) {
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
            midv = (gate / (1.0f + expf(-gate))) * up *
                   weights[(uint64_t)tok * n_expert + slot];
            mid_out[off] = midv;
        }
    }
    if (lane == 0u) {
        tile_vals[warp] = midv;
        tile_abs[warp] = valid ? fabsf(midv) : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float best_abs = tile_abs[0];
        float best_val = tile_vals[0];
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) {
            if (tile_abs[i] > best_abs) {
                best_abs = tile_abs[i];
                best_val = tile_vals[i];
            }
        }
        const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
        const uint32_t qblock = blockIdx.x / 32u;
        const uint32_t tile = blockIdx.x & 31u;
        if (qblock < midq_blocks) {
            amax_sidecar[((uint64_t)pair * midq_blocks + qblock) * 32u + tile] = best_val;
        }
    }
}

__global__ static void moe_gate_up_mid_decode_q4K_warp32_row16_kernel(
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
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 16u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
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
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_midq_decode_q4K_qwarp32_kernel(
        float *mid_out,
        cuda_block_q8_K *midq,
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
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t qblock = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ float vals[CUDA_QK_K];
    __shared__ float abs_part[CUDA_QK_K];
    __shared__ float val_part[CUDA_QK_K];
    __shared__ float iscale_s;

    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }

    const float w = weights[(uint64_t)tok * n_expert + slot];
    #pragma unroll
    for (uint32_t rr = 0; rr < 8u; rr++) {
        const uint32_t row_in_block = row_lane + rr * 32u;
        const uint32_t row = qblock * CUDA_QK_K + row_in_block;
        float midv = 0.0f;
        if (row < expert_mid_dim) {
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
            if (lane == 0u) {
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                midv = (gate / (1.0f + expf(-gate))) * up * w;
                const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                mid_out[off] = midv;
            }
        }
        if (lane == 0u) vals[row_in_block] = midv;
    }
    __syncthreads();

    cuda_block_q8_K *yb = midq + (uint64_t)pair * (expert_mid_dim / CUDA_QK_K) + qblock;
    const uint32_t tid = threadIdx.x;
    const float v = vals[tid];
    abs_part[tid] = fabsf(v);
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0u) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16u) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0u) {
        iscale_s = -127.0f / val_part[0];
    }
    __syncthreads();
    int qv = (int)lrintf(iscale_s * v);
    if (qv > 127) qv = 127;
    if (qv < -128) qv = -128;
    yb->qs[tid] = (int8_t)qv;
    __syncthreads();
    if (tid < CUDA_QK_K / 16u) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16u + (uint32_t)i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) yb->d = 1.0f / iscale_s;
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
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
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
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

__global__ static void moe_down_owned_slots_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        return;
    }
    const cuda_block_q2_K *wr =
        (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)slot * out_dim + row] = acc;
}

/* Map one of two packed operands for a three-slot reduction group. The only
 * multi-slot operand is the peer-owned prefix (slots 0+1 within the group),
 * which can be pre-added exactly because the reference reduction starts from
 * +0. Every other peer slot remains a distinct operand in original order. */
__device__ __forceinline__ static int moe_owned_packed_component(
        const int32_t *selected,
        uint32_t group,
        uint32_t component,
        uint32_t expert_base,
        uint32_t expert_count,
        bool *prefix_pair) {
    const uint32_t slot0 = group * 3u;
    uint32_t mask = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if (moe_owned_local_expert(selected[slot0 + i], expert_base,
                                   expert_count, NULL)) {
            mask |= 1u << i;
        }
    }
    *prefix_pair = false;
    if ((mask & 3u) == 3u) {
        if (component == 0u) {
            *prefix_pair = true;
            return (int)slot0;
        }
        return (mask & 4u) != 0u ? (int)(slot0 + 2u) : -1;
    }
    uint32_t ordinal = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if ((mask & (1u << i)) == 0u) continue;
        if (ordinal++ == component) return (int)(slot0 + i);
    }
    return -1;
}

__global__ static void moe_down_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q2_K *wr =
            (const cuda_block_q2_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}

__global__ static void moe_down_sum3_qwarp32_kernel(
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
    for (uint32_t slot = 0; slot < 3u; slot++) {
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
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_owned_slots_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        return;
    }
    const cuda_block_q4_K *wr =
        (const cuda_block_q4_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float acc = 0.0f;
    if (vec_ok) {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        }
    } else {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)slot * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
            }
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
            }
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}

__global__ static void moe_owned_slots_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_slots,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t col =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t row = blockIdx.y;
    if (col >= out_dim) return;
    out += (uint64_t)row * out_dim;
    home_slots += (uint64_t)row * 6u * out_dim;
    peer_slots += (uint64_t)row * 6u * out_dim;
    selected += (uint64_t)row * 6u;
    float slotv[6];
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const int32_t expert = selected[slot];
        if (expert < 0 ||
            (uint32_t)expert >= 2u * expert_split) {
            slotv[slot] = 0.0f;
        } else {
            const bool on_home = (uint32_t)expert < expert_split;
            const float *src = on_home ? home_slots : peer_slots;
            slotv[slot] = src[(uint64_t)slot * out_dim + col];
        }
    }
    float home = __fadd_rn(0.0f, slotv[0]);
    home = __fadd_rn(home, slotv[1]);
    home = __fadd_rn(home, slotv[2]);
    float peer = __fadd_rn(0.0f, slotv[3]);
    peer = __fadd_rn(peer, slotv[4]);
    peer = __fadd_rn(peer, slotv[5]);
    out[col] = __fadd_rn(home, peer);
}

__device__ static float moe_owned_packed_combine_row(
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t row,
        uint32_t out_dim,
        uint32_t expert_split) {
    float groups[2];
    #pragma unroll
    for (uint32_t group = 0; group < 2u; group++) {
        const uint32_t slot0 = group * 3u;
        uint32_t peer_mask = 0u;
        uint32_t valid_mask = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 3u; i++) {
            const int32_t expert = selected[slot0 + i];
            if (expert >= 0 && (uint32_t)expert < 2u * expert_split) {
                valid_mask |= 1u << i;
            }
            if (expert >= 0 && (uint32_t)expert >= expert_split &&
                (uint32_t)expert < 2u * expert_split) {
                peer_mask |= 1u << i;
            }
        }
        const float *packed = peer_packed +
            (uint64_t)group * 2u * out_dim + row;
        float acc;
        if ((peer_mask & 3u) == 3u) {
            /* packed[0] is already (+0 + slot0) + slot1. */
            acc = packed[0];
            float slot2 = 0.0f;
            if ((peer_mask & 4u) != 0u) {
                slot2 = packed[out_dim];
            } else if ((valid_mask & 4u) != 0u) {
                slot2 = home_slots[(uint64_t)(slot0 + 2u) * out_dim + row];
            }
            acc = __fadd_rn(acc, slot2);
        } else {
            acc = 0.0f;
            uint32_t peer_operand = 0u;
            #pragma unroll
            for (uint32_t i = 0; i < 3u; i++) {
                float value;
                if ((peer_mask & (1u << i)) != 0u) {
                    value = packed[(uint64_t)peer_operand * out_dim];
                    peer_operand++;
                } else if ((valid_mask & (1u << i)) != 0u) {
                    value = home_slots[(uint64_t)(slot0 + i) * out_dim + row];
                } else {
                    value = 0.0f;
                }
                acc = __fadd_rn(acc, value);
            }
        }
        groups[group] = acc;
    }
    return __fadd_rn(groups[0], groups[1]);
}

__global__ static void moe_owned_packed_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t row =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= out_dim) return;
    out[row] = moe_owned_packed_combine_row(
            home_slots, peer_packed, selected, row, out_dim, expert_split);
}

__global__ static void moe_down_q4K_sum3_qwarp32_kernel(
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
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 3u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum3_slotwarp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slot = lane >> 3u;
    const uint32_t qlane = lane & 7u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= out_dim) return;

    float acc = 0.0f;
    if (slot < 3u) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                                      (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                                      (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        for (uint32_t b = qlane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, qlane);
    }

    const float s1 = __shfl_sync(0xffffffffu, acc, 8);
    const float s2 = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0u) {
        const float s0 = acc;
        out[row] = (s0 + s1) + s2;
    }
}

static void routed_moe_decode_graph_destroy_one(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return;
    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    if (c->exec) (void)cudaGraphExecDestroy(c->exec);
    if (c->graph) (void)cudaGraphDestroy(c->graph);
    memset(c, 0, sizeof(*c));
}

static int routed_moe_decode_q4_graph_launch(
        int logical_tier,
        float *out,
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_w,
        const char *up_w,
        const char *down_w,
        cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *x) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return 0;
    if (n_expert != 3u && n_expert != 6u) return 0;
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    if (xq_blocks == 0u || midq_blocks == 0u) return 0;

    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    const bool shape_match =
        c->valid &&
        c->n_expert == n_expert &&
        c->expert_in_dim == expert_in_dim &&
        c->expert_mid_dim == expert_mid_dim &&
        c->out_dim == out_dim;
    if (c->valid && !shape_match) {
        routed_moe_decode_graph_destroy_one(logical_tier);
        c = &g_moe_decode_graph[logical_tier];
    }

    uint32_t x_rows = 1u;
    uint32_t mid_rows = n_expert;
    dim3 xq_grid(xq_blocks, 1, 1);
    dim3 gate_grid((expert_mid_dim + 7u) / 8u, n_expert, 1);
    dim3 midq_grid(midq_blocks, n_expert, 1);
    dim3 down_grid((out_dim + 31u) / 32u, 1, 1);
    dim3 block(256, 1, 1);

    void *xq_args[] = { &xq, &x, &expert_in_dim, &x_rows };
    cudaKernelNodeParams xq_params;
    memset(&xq_params, 0, sizeof(xq_params));
    xq_params.func = (void *)q8_K_quantize_kernel;
    xq_params.gridDim = xq_grid;
    xq_params.blockDim = block;
    xq_params.kernelParams = xq_args;

    void *gate_args[] = {
        &gate_out, &up_out, &mid_out, &gate_w, &up_w, &xq, &selected,
        &weights, &gate_expert_bytes, &gate_row_bytes, &xq_blocks,
        &expert_mid_dim, &n_expert, &write_aux, &clamp
    };
    cudaKernelNodeParams gate_params;
    memset(&gate_params, 0, sizeof(gate_params));
    gate_params.func = (void *)moe_gate_up_mid_decode_q4K_warp32_kernel;
    gate_params.gridDim = gate_grid;
    gate_params.blockDim = block;
    gate_params.kernelParams = gate_args;

    void *midq_args[] = { &midq, &mid_out, &expert_mid_dim, &mid_rows };
    cudaKernelNodeParams midq_params;
    memset(&midq_params, 0, sizeof(midq_params));
    midq_params.func = (void *)q8_K_quantize_kernel;
    midq_params.gridDim = midq_grid;
    midq_params.blockDim = block;
    midq_params.kernelParams = midq_args;

    void *down_args[] = {
        &out, &down_w, &midq, &selected, &down_expert_bytes,
        &down_row_bytes, &midq_blocks, &out_dim
    };
    cudaKernelNodeParams down_params;
    memset(&down_params, 0, sizeof(down_params));
    down_params.func = n_expert == 6u
        ? (void *)moe_down_q4K_sum6_qwarp32_kernel
        : (void *)moe_down_q4K_sum3_qwarp32_kernel;
    down_params.gridDim = down_grid;
    down_params.blockDim = block;
    down_params.kernelParams = down_args;

    if (!c->valid) {
        cudaError_t err = cudaGraphCreate(&c->graph, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph create failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        err = cudaGraphAddKernelNode(&c->xq_node, c->graph, NULL, 0,
                                     &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->gate_node, c->graph,
                                         &c->xq_node, 1, &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->midq_node, c->graph,
                                         &c->gate_node, 1, &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->down_node, c->graph,
                                         &c->midq_node, 1, &down_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphInstantiate(&c->exec, c->graph, NULL, NULL, 0);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph instantiate failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        c->n_expert = n_expert;
        c->expert_in_dim = expert_in_dim;
        c->expert_mid_dim = expert_mid_dim;
        c->out_dim = out_dim;
        c->valid = 1;
    } else {
        cudaError_t err =
            cudaGraphExecKernelNodeSetParams(c->exec, c->xq_node,
                                             &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->gate_node,
                                                   &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->midq_node,
                                                   &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->down_node,
                                                   &down_params);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph update failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
    }

    cudaError_t err = cudaGraphLaunch(c->exec, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: routed MoE decode graph launch failed: %s\n",
                cudaGetErrorString(err));
        routed_moe_decode_graph_destroy_one(logical_tier);
        return -1;
    }
    return 1;
}

/* Q4_K prefill (n_tokens > 1) down kernel. Mirrors moe_down_qwarp32_kernel
 * geometry exactly; only the weight block type and dot helper differ. The
 * pair = blockIdx.y indexing means the same grid shape (out_dim/32, n_tokens*n_expert)
 * used by the IQ2 path applies here. The downstream moe_sum_kernel is
 * weight-type-agnostic and sums these per-pair outputs into the final output. */
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

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
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
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
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
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
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
            if (lane == 0) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}


/* INT8 tensor-core (m8n8k16) exact MoE prefill tile kernels.
 *
 * Each warp computes an 8-token x 8-row tile. The Q4_K x Q8_K superblock dot
 * keeps its integer sums (order-invariant, exact) but computes the 32-wide
 * group dots on tensor cores; every output element keeps 8 float slot
 * accumulators (slot[b & 7] += term_b, b ascending) and reduces them with the
 * exact quarter_warp_sum_f32 grouping, so results are bit-identical to the
 * scalar expert-tile kernels (fuzz-verified). Requires sm_75+, 16B-aligned
 * expert tensors, and the staged activation-block counts (<=16 gate/up,
 * <=8 down). Rollback: DS4_CUDA_MOE_NO_Q4_MMA=1. */
__device__ __forceinline__ static void mma_m8n8k16_s8(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)a; (void)b; (void)c0; (void)c1;
#endif
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile8_mma_kernel(
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_tok[8];
    __shared__ uint32_t s_slot[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            uint32_t pr = sorted_pairs[offsets[expert] + local_pair];
            s_pair[np] = pr;
            s_tok[np] = pr / n_expert;
            s_slot[np] = pr - s_tok[np] * n_expert;
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u); i += blockDim.x) {
            const uint32_t words_per_tok = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;      /* token row of this thread's C elems */
    const uint32_t n0 = (lane & 3u) * 2u;  /* first C column (weight row) */
    /* 8 warps x 8 rows = 64 rows per pass */
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        const char *grow = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        const char *urow = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        /* per-element slot accumulators (2 elements x 8 slots) */
        float sg0[8] = {0,0,0,0,0,0,0,0}, sg1[8] = {0,0,0,0,0,0,0,0};
        float su0[8] = {0,0,0,0,0,0,0,0}, su1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < xq_blocks; b++) {
            /* headers for this thread's two C columns */
            const uint4 ghdr0 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 ghdr1 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            const uint4 uhdr0 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 uhdr1 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            /* B-fragment source rows for loads: n_load = lane>>2.
             * Batch all global loads for this superblock upfront so the
             * memory system sees independent requests instead of a
             * load->mma dependency chain. */
            const uint32_t *gqw = (const uint32_t *)(((const cuda_block_q4_K *)(grow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const uint32_t *uqw = (const uint32_t *)(((const cuda_block_q4_K *)(urow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t gw8[8], uw8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                gw8[k] = gqw[k * 4u + (lane & 3u)];
                uw8[k] = uqw[k * 4u + (lane & 3u)];
            }
            int gi0 = 0, gi1 = 0, ui0 = 0, ui1 = 0;
            int gs0 = 0, gs1 = 0, us0 = 0, us1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                /* dot32 via two chained k16 mmas, per matrix */
                int32_t gc0 = 0, gc1 = 0, uc0 = 0, uc1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t gw = (gw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    const uint32_t uw = (uw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(gc0, gc1, a, gw);
                    mma_m8n8k16_s8(uc0, uc1, a, uw);
                }
                /* integer scale application for this thread's two columns */
                uint8_t sc, m;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr0.y, &sc, &m);
                gi0 += (int)sc * gc0;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                gs0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr1.y, &sc, &m);
                gi1 += (int)sc * gc1;
                gs1 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr0.y, &sc, &m);
                ui0 += (int)sc * uc0;
                us0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr1.y, &sc, &m);
                ui1 += (int)sc * uc1;
                us1 += (int)m * bs;
            }
            /* float finish, exact dev_dot_q4_K_q8_K_block8 expression */
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            sg0[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr0.x & 0xffffu)) * (float)gi0 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr0.x >> 16u)) * (float)gs0;
            sg1[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr1.x & 0xffffu)) * (float)gi1 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr1.x >> 16u)) * (float)gs1;
            su0[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr0.x & 0xffffu)) * (float)ui0 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr0.x >> 16u)) * (float)us0;
            su1[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr1.x & 0xffffu)) * (float)ui1 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr1.x >> 16u)) * (float)us1;
        }
        /* quarter_warp_sum_f32 order: ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)) */
        const uint32_t p = mtok;
        if (p < np) {
            const uint32_t rowa = row0 + n0;
            const uint32_t rowb = row0 + n0 + 1u;
            float gate2[2], up2[2];
            {
                float a0 = sg0[0] + sg0[4], a1 = sg0[1] + sg0[5], a2 = sg0[2] + sg0[6], a3 = sg0[3] + sg0[7];
                gate2[0] = (a0 + a2) + (a1 + a3);
                a0 = sg1[0] + sg1[4]; a1 = sg1[1] + sg1[5]; a2 = sg1[2] + sg1[6]; a3 = sg1[3] + sg1[7];
                gate2[1] = (a0 + a2) + (a1 + a3);
                a0 = su0[0] + su0[4]; a1 = su0[1] + su0[5]; a2 = su0[2] + su0[6]; a3 = su0[3] + su0[7];
                up2[0] = (a0 + a2) + (a1 + a3);
                a0 = su1[0] + su1[4]; a1 = su1[1] + su1[5]; a2 = su1[2] + su1[6]; a3 = su1[3] + su1[7];
                up2[1] = (a0 + a2) + (a1 + a3);
            }
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t row = e ? rowb : rowa;
                if (row >= expert_mid_dim) continue;
                float gate = gate2[e];
                float up = up2[e];
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                const uint64_t off = (uint64_t)s_pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate;
                    up_out[off] = up;
                }
                mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile8_mma_kernel(
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
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (midq_blocks <= 8u) {
        const uint32_t words_per_tok = midq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(midq + (uint64_t)s_pair[p] * midq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        const char *wrow = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes;
        float s0[8] = {0,0,0,0,0,0,0,0}, s1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < midq_blocks; b++) {
            const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * down_row_bytes) + b);
            const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * down_row_bytes) + b);
            const uint32_t *wqw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * down_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t w8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) w8[k] = wqw[k * 4u + (lane & 3u)];
            int i0 = 0, i1 = 0, m0 = 0, m1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                int32_t c0 = 0, c1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t w = (w8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(c0, c1, a, w);
                }
                uint8_t sc, m;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc, &m);
                i0 += (int)sc * c0;
                m0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc, &m);
                i1 += (int)sc * c1;
                m1 += (int)m * bs;
            }
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            s0[sl] += yd * dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu)) * (float)i0 -
                      yd * dev_f16_to_f32((uint16_t)(hdr0.x >> 16u)) * (float)m0;
            s1[sl] += yd * dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu)) * (float)i1 -
                      yd * dev_f16_to_f32((uint16_t)(hdr1.x >> 16u)) * (float)m1;
        }
        const uint32_t p = mtok;
        if (p < np) {
            float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
            const float r0 = (a0 + a2) + (a1 + a3);
            a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
            const float r1 = (a0 + a2) + (a1 + a3);
            if (row0 + n0 < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0] = r0;
            if (row0 + n0 + 1u < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0 + 1u] = r1;
        }
    }
}

/* 16-pair MoE expert tile kernels on sm_80+ m16n8k32 INT8 tensor cores.
 *
 * Same per-output math and reduction order as the 8-pair expert tile
 * kernels (slot[b & 7] += term_b with b ascending, then the exact
 * quarter_warp_sum_f32 grouping), so results are bit-identical; grouping
 * 16 pairs per tile just halves how often each expert's weights are
 * streamed from DRAM. Gate and up run as two passes over the superblocks
 * to keep register pressure at the 8-pair kernel's level. */

__device__ __forceinline__ static void mma16_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)c0;(void)c1;(void)c2;(void)c3;
#endif
}

/* One matrix pass over all superblocks for this thread's 4 C elements
 * (tokens mtokA/mtokB x rows n0/n0+1). Returns the quarter-tree-reduced
 * values in r[4] with the exact reference ordering. */
__device__ __forceinline__ static void moe_tile16_mma_pass(
        const char *wrow,            /* row0 base of this matrix */
        uint64_t row_bytes,
        const cuda_block_q8_K (*sxq)[16],
        uint32_t xq_blocks,
        uint32_t lane,
        float r[4]) {
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    float s0[8] = {0,0,0,0,0,0,0,0};
    float s1[8] = {0,0,0,0,0,0,0,0};
    float s2[8] = {0,0,0,0,0,0,0,0};
    float s3[8] = {0,0,0,0,0,0,0,0};
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * row_bytes) + b);
        const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * row_bytes) + b);
        const uint32_t *qw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * row_bytes) + b)->qs);
        uint32_t w8[8];
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) w8[k] = qw[k * 4u + (lane & 3u)];
        const int8_t *aqsA = sxq[mtokA][b].qs;
        const int8_t *aqsB = sxq[mtokB][b].qs;
        int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const int shift = (j & 1u) ? 4 : 0;
            const uint32_t koff = (lane & 3u) * 4u;
            const uint32_t a0 = *(const uint32_t *)(aqsA + j * 32u + koff);
            const uint32_t a1 = *(const uint32_t *)(aqsB + j * 32u + koff);
            const uint32_t a2 = *(const uint32_t *)(aqsA + j * 32u + 16u + koff);
            const uint32_t a3 = *(const uint32_t *)(aqsB + j * 32u + 16u + koff);
            const uint32_t b0 = (w8[(j >> 1u) * 2u + 0u] >> shift) & 0x0f0f0f0fu;
            const uint32_t b1 = (w8[(j >> 1u) * 2u + 1u] >> shift) & 0x0f0f0f0fu;
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma16_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            uint8_t sc0, sm0, sc1, sm1;
            dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc0, &sm0);
            dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc1, &sm1);
            const int bsA = (int)sxq[mtokA][b].bsums[2u * j] + (int)sxq[mtokA][b].bsums[2u * j + 1u];
            const int bsB = (int)sxq[mtokB][b].bsums[2u * j] + (int)sxq[mtokB][b].bsums[2u * j + 1u];
            i0 += (int)sc0 * c0;
            i1 += (int)sc1 * c1;
            i2 += (int)sc0 * c2;
            i3 += (int)sc1 * c3;
            m0 += (int)sm0 * bsA;
            m1 += (int)sm1 * bsA;
            m2 += (int)sm0 * bsB;
            m3 += (int)sm1 * bsB;
        }
        const float ydA = sxq[mtokA][b].d;
        const float ydB = sxq[mtokB][b].d;
        const float xd0 = dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu));
        const float xmin0 = dev_f16_to_f32((uint16_t)(hdr0.x >> 16u));
        const float xd1 = dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu));
        const float xmin1 = dev_f16_to_f32((uint16_t)(hdr1.x >> 16u));
        const uint32_t sl = b & 7u;
        s0[sl] += ydA * xd0 * (float)i0 - ydA * xmin0 * (float)m0;
        s1[sl] += ydA * xd1 * (float)i1 - ydA * xmin1 * (float)m1;
        s2[sl] += ydB * xd0 * (float)i2 - ydB * xmin0 * (float)m2;
        s3[sl] += ydB * xd1 * (float)i3 - ydB * xmin1 * (float)m3;
    }
    {
        float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
        r[0] = (a0 + a2) + (a1 + a3);
        a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
        r[1] = (a0 + a2) + (a1 + a3);
        a0 = s2[0] + s2[4]; a1 = s2[1] + s2[5]; a2 = s2[2] + s2[6]; a3 = s2[3] + s2[7];
        r[2] = (a0 + a2) + (a1 + a3);
        a0 = s3[0] + s3[4]; a1 = s3[1] + s3[5]; a2 = s3[2] + s3[6]; a3 = s3[3] + s3[7];
        r[3] = (a0 + a2) + (a1 + a3);
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile16_mma_kernel(
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    extern __shared__ unsigned char t16_sh[];
    cuda_block_q8_K (*sxq)[16] = (cuda_block_q8_K (*)[16])t16_sh; /* [16][16] */
    __shared__ uint32_t s_pair[16];
    __shared__ uint32_t s_tok[16];
    __shared__ uint32_t s_slot[16];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 16u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            uint32_t pr = sorted_pairs[offsets[expert] + local_pair];
            s_pair[np] = pr;
            s_tok[np] = pr / n_expert;
            s_slot[np] = pr - s_tok[np] * n_expert;
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (xq_blocks <= 16u) {
        const uint32_t words_per_tok = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w];
        }
        /* zero-fill missing pairs so the A fragments are defined */
        const uint32_t total_words = 16u * words_per_tok;
        for (uint32_t i = threadIdx.x + np * words_per_tok; i < total_words; i += blockDim.x) {
            ((uint32_t *)t16_sh)[i] = 0u;
        }
        __syncthreads();
    }
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        const char *grow = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        const char *urow = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        float gr[4], ur[4];
        moe_tile16_mma_pass(grow, gate_row_bytes, (const cuda_block_q8_K (*)[16])sxq, xq_blocks, lane, gr);
        moe_tile16_mma_pass(urow, gate_row_bytes, (const cuda_block_q8_K (*)[16])sxq, xq_blocks, lane, ur);
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = (e < 2u) ? mtokA : mtokB;
            const uint32_t row = row0 + n0 + (e & 1u);
            if (p >= np || row >= expert_mid_dim) continue;
            float gate = gr[e];
            float up = ur[e];
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)s_pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                           weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile16_mma_kernel(
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
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    extern __shared__ unsigned char t16_sh[];
    __shared__ uint32_t s_pair[16];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 16u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (midq_blocks <= 16u) {
        const uint32_t words_per_tok = midq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)t16_sh)[i] = ((const uint32_t *)(midq + (uint64_t)s_pair[p] * midq_blocks))[w];
        }
        const uint32_t total_words = 16u * words_per_tok;
        for (uint32_t i = threadIdx.x + np * words_per_tok; i < total_words; i += blockDim.x) {
            ((uint32_t *)t16_sh)[i] = 0u;
        }
        __syncthreads();
    }
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        const char *wrow = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes;
        float s0[8] = {0,0,0,0,0,0,0,0};
        float s1[8] = {0,0,0,0,0,0,0,0};
        float s2[8] = {0,0,0,0,0,0,0,0};
        float s3[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < midq_blocks; b++) {
            const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * down_row_bytes) + b);
            const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * down_row_bytes) + b);
            const uint32_t *qw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * down_row_bytes) + b)->qs);
            uint32_t w8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) w8[k] = qw[k * 4u + (lane & 3u)];
            /* activation rows: midq_blocks stride within the staged region */
            const int8_t *aqsA = ((const cuda_block_q8_K *)t16_sh + (uint64_t)mtokA * midq_blocks + b)->qs;
            const int8_t *aqsB = ((const cuda_block_q8_K *)t16_sh + (uint64_t)mtokB * midq_blocks + b)->qs;
            const cuda_block_q8_K *blkA = (const cuda_block_q8_K *)t16_sh + (uint64_t)mtokA * midq_blocks + b;
            const cuda_block_q8_K *blkB = (const cuda_block_q8_K *)t16_sh + (uint64_t)mtokB * midq_blocks + b;
            int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
            int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                const uint32_t koff = (lane & 3u) * 4u;
                const uint32_t a0 = *(const uint32_t *)(aqsA + j * 32u + koff);
                const uint32_t a1 = *(const uint32_t *)(aqsB + j * 32u + koff);
                const uint32_t a2 = *(const uint32_t *)(aqsA + j * 32u + 16u + koff);
                const uint32_t a3 = *(const uint32_t *)(aqsB + j * 32u + 16u + koff);
                const uint32_t b0 = (w8[(j >> 1u) * 2u + 0u] >> shift) & 0x0f0f0f0fu;
                const uint32_t b1 = (w8[(j >> 1u) * 2u + 1u] >> shift) & 0x0f0f0f0fu;
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma16_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                uint8_t sc0, sm0, sc1, sm1;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc0, &sm0);
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc1, &sm1);
                const int bsA = (int)blkA->bsums[2u * j] + (int)blkA->bsums[2u * j + 1u];
                const int bsB = (int)blkB->bsums[2u * j] + (int)blkB->bsums[2u * j + 1u];
                i0 += (int)sc0 * c0;
                i1 += (int)sc1 * c1;
                i2 += (int)sc0 * c2;
                i3 += (int)sc1 * c3;
                m0 += (int)sm0 * bsA;
                m1 += (int)sm1 * bsA;
                m2 += (int)sm0 * bsB;
                m3 += (int)sm1 * bsB;
            }
            const float ydA = blkA->d;
            const float ydB = blkB->d;
            const float xd0 = dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu));
            const float xmin0 = dev_f16_to_f32((uint16_t)(hdr0.x >> 16u));
            const float xd1 = dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu));
            const float xmin1 = dev_f16_to_f32((uint16_t)(hdr1.x >> 16u));
            const uint32_t sl = b & 7u;
            s0[sl] += ydA * xd0 * (float)i0 - ydA * xmin0 * (float)m0;
            s1[sl] += ydA * xd1 * (float)i1 - ydA * xmin1 * (float)m1;
            s2[sl] += ydB * xd0 * (float)i2 - ydB * xmin0 * (float)m2;
            s3[sl] += ydB * xd1 * (float)i3 - ydB * xmin1 * (float)m3;
        }
        float rr4[4];
        {
            float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
            rr4[0] = (a0 + a2) + (a1 + a3);
            a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
            rr4[1] = (a0 + a2) + (a1 + a3);
            a0 = s2[0] + s2[4]; a1 = s2[1] + s2[5]; a2 = s2[2] + s2[6]; a3 = s2[3] + s2[7];
            rr4[2] = (a0 + a2) + (a1 + a3);
            a0 = s3[0] + s3[4]; a1 = s3[1] + s3[5]; a2 = s3[2] + s3[6]; a3 = s3[3] + s3[7];
            rr4[3] = (a0 + a2) + (a1 + a3);
        }
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = (e < 2u) ? mtokA : mtokB;
            const uint32_t row = row0 + n0 + (e & 1u);
            if (p >= np || row >= out_dim) continue;
            down_out[(uint64_t)s_pair[p] * out_dim + row] = rr4[e];
        }
    }
}

static int cuda_q4_mma_tile16_shmem_ok(int which_down) {
    /* Opt the tile16 kernels into >48KB dynamic shared memory, per device. */
    static int ready[DS4_MAX_GPUS][2];
    static int failed = 0;
    if (failed) return 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_MAX_GPUS) return 0;
    if (ready[dev][which_down]) return 1;
    cudaFuncAttributes fn_attr;
    cudaError_t err = which_down
        ? cudaFuncGetAttributes(&fn_attr, moe_down_q4K_tile16_mma_kernel<512>)
        : cudaFuncGetAttributes(&fn_attr, moe_gate_up_mid_q4K_tile16_mma_kernel<512>);
    if (err != cudaSuccess || fn_attr.binaryVersion < 80) {
        failed = 1;
        return 0;
    }
    const int bytes = (int)(16u * 16u * sizeof(cuda_block_q8_K));
    if (which_down) {
        err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<512>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<1024>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<2048>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    } else {
        err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<512>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<1024>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<2048>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    if (err != cudaSuccess) {
        failed = 1;
        return 0;
    }
    ready[dev][which_down] = 1;
    return 1;
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

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
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

__global__ static void moe_sum_owned_kernel(
        float *out,
        const float *down,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    const uint32_t tok = (uint32_t)(gid / out_dim);
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        if (slot >= n_expert) break;
        const uint64_t pair = (uint64_t)tok * n_expert + slot;
        const float value = selected[pair] >= 0
            ? down[pair * out_dim + row] : 0.0f;
        acc = __fadd_rn(acc, value);
    }
    out[gid] = acc;
}

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

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        int allow_streaming,
        int owned_filtered) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
    /* Q4_K routed-MoE dispatch:
     *   n_tokens == 1 and n_expert == 6:
     *                  use_direct_down_sum + moe_gate_up_mid_decode_q4K_qwarp32
     *                  + moe_down_q4K_sum6_qwarp32.
     *   n_tokens == 1 and n_expert == 3:
     *                  use the same direct path with moe_down_q4K_sum3_qwarp32.
     *                  Decode TP relies on this for splitting the six selected
     *                  experts into two groups.
     *   n_tokens == 1 and other n_expert:
     *                  use the same per-pair gate/up kernel plus the generic
     *                  q4K down + sum path.
     *   n_tokens >  1: default sorted-pairs expert-tile path groups token/expert
     *                  pairs by expert and uses Q4_K tile8 gate/up + down kernels
     *                  (`DS4_CUDA_MOE_NO_Q4_SORTED=1` restores the older
     *                  token-indexed decode-style prefill kernels). */
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const uint64_t required_slot_count = (uint64_t)n_tokens * n_expert;
    const int logical_tier = ds4_tensor_device_idx(out);
    const int use_stream_selected_cache =
        allow_streaming &&
        g_ssd_streaming_mode &&
        g_stream_selected_cache.valid &&
        g_stream_selected_cache.logical_tier == logical_tier &&
        g_stream_selected_cache.model_map == model_map &&
        g_stream_selected_cache.layer == layer_index &&
        g_stream_selected_cache.n_total_expert == n_total_expert &&
        g_stream_selected_cache.slot_count >= required_slot_count &&
        g_stream_selected_cache.gate_offset == gate_offset &&
        g_stream_selected_cache.up_offset == up_offset &&
        g_stream_selected_cache.down_offset == down_offset &&
        g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
        g_stream_selected_cache.gate_ptr &&
        g_stream_selected_cache.up_ptr &&
        g_stream_selected_cache.down_ptr &&
        g_stream_selected_cache.slot_selected_tensor.ptr &&
        g_stream_selected_cache.slot_selected_tensor.bytes >=
            required_slot_count * sizeof(int32_t);
    if (g_ssd_streaming_mode && allow_streaming &&
        !use_stream_selected_cache) {
        fprintf(stderr,
                "ds4: CUDA streaming selected experts are unavailable for layer %u\n",
                layer_index);
        return 0;
    }
    if (use_stream_selected_cache) {
        selected = &g_stream_selected_cache.slot_selected_tensor;
    }
    const char *gate_w = use_stream_selected_cache ?
        g_stream_selected_cache.gate_ptr :
        cuda_resolve_weight_ptr(model_map, gate_offset, gate_bytes,
                                logical_tier, "moe_gate");
    const char *up_w = use_stream_selected_cache ?
        g_stream_selected_cache.up_ptr :
        cuda_resolve_weight_ptr(model_map, up_offset, gate_bytes,
                                logical_tier, "moe_up");
    const char *down_w = use_stream_selected_cache ?
        g_stream_selected_cache.down_ptr :
        cuda_resolve_weight_ptr(model_map, down_offset, down_bytes,
                                logical_tier, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_q4_sorted_pairs =
            q4k_path && n_tokens > 1u &&
            (owned_filtered ||
             (getenv("DS4_CUDA_MOE_NO_Q4_SORTED") == NULL &&
              getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL &&
              getenv("DS4_CUDA_MOE_TILE4") == NULL));
        const uint32_t use_sorted_pairs =
            n_tokens > 1u &&
            (owned_filtered || !q4k_path || use_q4_sorted_pairs);
        const uint32_t use_expert_tiles =
            use_sorted_pairs &&
            (owned_filtered || getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL);
        /* Small batches (DSpark stage chain / verify, n<=8) leave most of an
         * 8-slot expert tile empty (1-2 rows per expert): tile4 halves the
         * wasted dot-slots and measures ~2x faster there. Large prefill
         * keeps tile8. Env overrides both ways. */
        const uint32_t expert_tile_m =
            getenv("DS4_CUDA_MOE_TILE4") ? 4u :
            (getenv("DS4_CUDA_MOE_TILE8") ? 8u :
             (n_tokens <= 8u ? 4u : 8u));
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted =
            use_sorted_pairs && !owned_filtered &&
            getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = !q4k_path && use_expert_tiles &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL));
        const uint32_t use_owned_sparse_buffers = owned_filtered &&
            getenv("DS4_CUDA_MOE_NO_OWNED_SPARSE_BUFFERS") == NULL;
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_q4_mma_tiles16 = q4k_path && use_expert_tiles &&
            expert_tile_m == 8u && cuda_q4_mma_ok() &&
            getenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16") == NULL;
        const uint32_t use_down_tile16 = !q4k_path && use_atomic_down && expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL;
        const uint32_t use_small_sorted_prep =
            owned_filtered && q4k_path && n_tokens <= 16u && pair_count <= 96u &&
            n_total_expert <= 128u && use_sorted_pairs && use_expert_tiles &&
            getenv("DS4_CUDA_MOE_NO_SMALL_SORTED_PREP") == NULL;
        const uint32_t use_q4_down_rowspan = q4k_path && use_expert_tiles && expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_Q4_DOWN_ROWSPAN") == NULL;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u :
            getenv("DS4_CUDA_MOE_GATE_ROW1024") != NULL ? 1024u : 512u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ? 2048u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 512u;
        const uint32_t use_down_row2048 = !q4k_path && use_atomic_down && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum =
            n_tokens == 1u && (n_expert == 6u || n_expert == 3u) &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        const uint32_t use_direct_midq =
            q4k_path && use_direct_down_sum && !write_gate_up &&
            getenv("DS4_CUDA_MOE_DIRECT_MIDQ") != NULL &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_MIDQ") == NULL;
        const uint32_t use_q4_gate_h16r8 =
            q4k_path && !use_direct_midq &&
            getenv("DS4_CUDA_MOE_Q4_GATE_H16R8") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_H16R8") == NULL;
        const uint32_t use_q4_gate_h16 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 &&
            getenv("DS4_CUDA_MOE_Q4_GATE_H16") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_H16") == NULL;
        const uint32_t use_q4_gate_w32r16 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 && !use_q4_gate_h16 &&
            getenv("DS4_CUDA_MOE_Q4_GATE_W32R16") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32R16") == NULL;
        const uint32_t use_q4_gate_w32 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 && !use_q4_gate_h16 &&
            !use_q4_gate_w32r16 &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32") == NULL;
        const uint32_t use_q4_gate_w32_noaux =
            use_q4_gate_w32 && !write_gate_up &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32_NOAUX") == NULL;
        const uint32_t use_q4_down_slot3 =
            q4k_path && use_direct_down_sum && n_expert == 3u &&
            getenv("DS4_CUDA_MOE_Q4_DOWN_SLOT3") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_DOWN_SLOT3") == NULL;
        const uint32_t use_q4_midq_sidecar =
            q4k_path && use_direct_down_sum && use_q4_gate_w32_noaux &&
            !use_direct_midq && !write_gate_up &&
            (expert_mid_dim % CUDA_QK_K) == 0u &&
            getenv("DS4_CUDA_MOE_MIDQ_SIDECAR") != NULL &&
            getenv("DS4_CUDA_MOE_NO_MIDQ_SIDECAR") == NULL;
        float *midq_sidecar = use_q4_midq_sidecar ? (float *)up->ptr : NULL;
        if (g_cuda_moe_decode_graph &&
            !owned_filtered &&
            !profile_moe &&
            q4k_path &&
            n_tokens == 1u &&
            use_direct_down_sum &&
            use_q4_gate_w32 &&
            !use_q4_gate_w32r16 &&
            !use_q4_down_slot3 &&
            !use_direct_midq &&
            !use_q4_midq_sidecar &&
            (n_expert == 3u || n_expert == 6u)) {
            int grc = routed_moe_decode_q4_graph_launch(
                    logical_tier,
                    (float *)out->ptr,
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    down_w,
                    xq,
                    midq,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    down_expert_bytes,
                    down_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    out_dim,
                    n_expert,
                    write_gate_up,
                    clamp,
                    (const float *)x->ptr);
            if (grc == 1) return 1;
            if (grc < 0) return 0;
        }
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + n_total_expert;
            tile16_capacity = (use_down_tile16 || use_q4_mma_tiles16) ? ((pair_count + 15u) / 16u + n_total_expert) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = (use_down_tile16 || use_q4_mma_tiles16) ? (((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t)) : 0u;
            const uint64_t tile16_total_bytes = (use_down_tile16 || use_q4_mma_tiles16) ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc_on(logical_tier, scratch_bytes,
                                                             "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                if (use_small_sorted_prep) {
                    moe_prepare_sorted_tiles_small_kernel<<<1, 128>>>(
                        counts, offsets, cursors, sorted_pairs,
                        tile_offsets, tile_total, tile_experts, tile_starts,
                        tile16_offsets, tile16_total, tile16_experts, tile16_starts,
                        (const int32_t *)selected->ptr, pair_count, n_total_expert,
                        expert_tile_m, use_down_tile16 || use_q4_mma_tiles16);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe small sorted setup launch");
                } else {
                    ok = cuda_ok(cudaMemset(counts, 0, counts_bytes),
                                 "routed_moe sorted counts clear");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count,
                        n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        (const int32_t *)selected->ptr,
                        pair_count,
                        n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep) {
                    moe_build_expert_tiles_kernel<<<(n_total_expert + 255u) / 256u, 256>>>(
                        tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep &&
                    (use_down_tile16 || use_q4_mma_tiles16)) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep &&
                    (use_down_tile16 || use_q4_mma_tiles16)) {
                    moe_build_expert_tiles_kernel<<<(n_total_expert + 255u) / 256u, 256>>>(
                        tile16_experts, tile16_starts, tile16_offsets, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok && owned_filtered && use_sorted_pairs &&
            !use_owned_sparse_buffers) {
            const uint64_t mid_bytes =
                (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
            ok = cuda_ok(cudaMemset(mid->ptr, 0, (size_t)mid_bytes),
                         "owned routed_moe mid clear");
        }
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    const int use_q4_mma = cuda_q4_mma_ok() &&
                        ((((uintptr_t)gate_w | (uintptr_t)up_w |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u) &&
                        xq_blocks <= 16u && (expert_mid_dim & 7u) == 0u;
                    const int use_q4_mma_t16 = use_q4_mma && use_q4_mma_tiles16 &&
                        tile16_total && tile16_experts && tile16_starts &&
                        xq_blocks == 16u && cuda_q4_mma_tile16_shmem_ok(0);
                    if (use_q4_mma_t16 && use_gate_row2048) {
                        const unsigned t16cap = (unsigned)((pair_count + 15u) / 16u + n_total_expert);
                        const size_t t16sh = 16u * 16u * sizeof(cuda_block_q8_K);
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<512><<<tgrid, 256, t16sh>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<1024><<<tgrid, 256, t16sh>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<2048><<<tgrid, 256, t16sh>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else if (use_q4_mma && use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<512><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<1024><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<2048><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else if (use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else {
                        dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                        moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + MOE_DECODE_ROWS_PER_BLOCK - 1u) / MOE_DECODE_ROWS_PER_BLOCK, n_tokens * n_expert, 1);
                if (q4k_path) {
                    /* Q4_K gate/up: the decode kernel is token-indexed via
                     * pair = blockIdx.y; tok = pair / n_expert, so the same
                     * launch covers both n_tokens == 1 (decode) and n_tokens > 1
                     * (prefill). q4k_path is steered here by use_sorted_pairs = 0
                     * cascading the IQ2 sorted/expert-tile branches off. */
                    if (use_direct_midq) {
                        dim3 mqgrid(midq_blocks, n_tokens * n_expert, 1);
                        moe_gate_up_midq_decode_q4K_qwarp32_kernel<<<mqgrid, 256>>>(
                            (float *)mid->ptr,
                            midq,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            clamp);
                    } else if (use_q4_gate_h16r8) {
                        dim3 h8grid((expert_mid_dim + 7u) / 8u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_hwarp16_row8_kernel<<<h8grid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else if (use_q4_gate_w32r16) {
                        dim3 w16grid((expert_mid_dim + 15u) / 16u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_warp32_row16_kernel<<<w16grid, 512>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else if (use_q4_gate_w32) {
                        dim3 wgrid((expert_mid_dim + 7u) / 8u, n_tokens * n_expert, 1);
                        if (use_q4_gate_w32_noaux) {
                            if (use_q4_midq_sidecar) {
                                moe_gate_up_mid_decode_q4K_warp32_noaux_sidecar_kernel<<<wgrid, 256>>>(
                                    (float *)mid->ptr,
                                    midq_sidecar,
                                    gate_w,
                                    up_w,
                                    xq,
                                    (const int32_t *)selected->ptr,
                                    (const float *)weights->ptr,
                                    gate_expert_bytes,
                                    gate_row_bytes,
                                    xq_blocks,
                                    expert_mid_dim,
                                    n_expert,
                                    clamp);
                            } else {
                                moe_gate_up_mid_decode_q4K_warp32_noaux_kernel<<<wgrid, 256>>>(
                                    (float *)mid->ptr,
                                    gate_w,
                                    up_w,
                                    xq,
                                    (const int32_t *)selected->ptr,
                                    (const float *)weights->ptr,
                                    gate_expert_bytes,
                                    gate_row_bytes,
                                    xq_blocks,
                                    expert_mid_dim,
                                    n_expert,
                                    clamp);
                            }
                        } else {
                            moe_gate_up_mid_decode_q4K_warp32_kernel<<<wgrid, 256>>>(
                                (float *)gate->ptr,
                                (float *)up->ptr,
                                (float *)mid->ptr,
                                gate_w,
                                up_w,
                                xq,
                                (const int32_t *)selected->ptr,
                                (const float *)weights->ptr,
                                gate_expert_bytes,
                                gate_row_bytes,
                                xq_blocks,
                                expert_mid_dim,
                                n_expert,
                                write_gate_up,
                                clamp);
                        }
                    } else if (use_q4_gate_h16) {
                        dim3 hgrid((expert_mid_dim + 15u) / 16u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_hwarp16_kernel<<<hgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else {
                        moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    }
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok && !use_direct_midq) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            if (use_q4_midq_sidecar) {
                q8_K_quantize_sidecar_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        midq_sidecar,
                        expert_mid_dim,
                        n_tokens * n_expert);
                ok = cuda_ok(cudaGetLastError(), "routed_moe mid sidecar quantize launch");
            } else if (use_owned_sparse_buffers) {
                q8_K_quantize_owned_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        expert_mid_dim,
                        n_tokens * n_expert,
                        0u,
                        n_total_expert);
                ok = cuda_ok(cudaGetLastError(),
                             "owned routed_moe active mid quantize launch");
            } else {
                q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
                ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
            }
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok && owned_filtered && use_sorted_pairs && !use_atomic_down &&
            !use_owned_sparse_buffers) {
            const uint64_t down_clear_bytes =
                (uint64_t)n_tokens * n_expert * out_dim * sizeof(float);
            ok = cuda_ok(cudaMemset(down->ptr, 0, (size_t)down_clear_bytes),
                         "owned routed_moe down clear");
        }
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    if (n_expert == 6u) {
                        moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    } else {
                        if (use_q4_down_slot3) {
                            dim3 swgrid((out_dim + 7u) / 8u, 1, 1);
                            moe_down_q4K_sum3_slotwarp_kernel<<<swgrid, 256>>>(
                                (float *)out->ptr,
                                down_w,
                                midq,
                                (const int32_t *)selected->ptr,
                                down_expert_bytes,
                                down_row_bytes,
                                midq_blocks,
                                out_dim);
                        } else {
                            moe_down_q4K_sum3_qwarp32_kernel<<<sgrid, 256>>>(
                                (float *)out->ptr,
                                down_w,
                                midq,
                                (const int32_t *)selected->ptr,
                                down_expert_bytes,
                                down_row_bytes,
                                midq_blocks,
                                out_dim);
                        }
                    }
                } else {
                    if (n_expert == 6u) {
                        moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    } else {
                        moe_down_sum3_qwarp32_kernel<<<sgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    }
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    const int use_q4_down_mma = cuda_q4_mma_ok() &&
                        ((((uintptr_t)down_w | down_row_bytes | down_expert_bytes) & 15u) == 0u) &&
                        midq_blocks <= 8u && (out_dim & 7u) == 0u;
                    const int use_q4_down_t16 = use_q4_down_mma && use_q4_mma_tiles16 &&
                        tile16_total && tile16_experts && tile16_starts &&
                        midq_blocks <= 16u && cuda_q4_mma_tile16_shmem_ok(1);
                    if (use_q4_down_t16 && use_q4_down_rowspan) {
                        const unsigned t16cap = (unsigned)((pair_count + 15u) / 16u + n_total_expert);
                        const size_t dt16sh = 16u * (size_t)midq_blocks * sizeof(cuda_block_q8_K);
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<512><<<tgrid, 256, dt16sh>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<1024><<<tgrid, 256, dt16sh>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<2048><<<tgrid, 256, dt16sh>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else if (use_q4_down_mma && use_q4_down_rowspan) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<512><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<1024><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<2048><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else if (use_q4_down_rowspan) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else if (q4k_path) {
                /* Q4_K prefill down. New kernel mirrors moe_down_qwarp32_kernel
                 * grid/geometry, swapping the weight block type to cuda_block_q4_K
                 * and the dot helper to dev_dot_q4_K_q8_K_block. Writes per-pair
                 * outputs into down->ptr; moe_sum_kernel below sums them across
                 * experts into out->ptr. */
                moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            if (use_owned_sparse_buffers) {
                moe_sum_owned_kernel<<<(n + 255) / 256, 256>>>(
                        (float *)out->ptr,
                        (const float *)down->ptr,
                        (const int32_t *)selected->ptr,
                        out_dim,
                        n_expert,
                        n_tokens);
            } else {
                moe_sum_kernel<<<(n + 255) / 256, 256>>>(
                        (float *)out->ptr,
                        (const float *)down->ptr,
                        out_dim,
                        n_expert,
                        n_tokens);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t resident_expert_base,
        uint32_t resident_expert_count,
        float clamp,
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor *down_output,
        bool pack_fixed3,
        ds4_gpu_tensor *shared_prequant) {
    if (!out || !gate || !up || !mid || !down || !model_map ||
        !selected || !weights || !x || n_expert != 6u ||
        n_total_expert == 0u || resident_expert_count == 0u ||
        gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base ||
        expert_in_dim % CUDA_QK_K != 0u ||
        expert_mid_dim % CUDA_QK_K != 0u ||
        selected->bytes < 6u * sizeof(int32_t) ||
        weights->bytes < 6u * sizeof(float) ||
        x->bytes < (uint64_t)expert_in_dim * sizeof(float) ||
        mid->bytes < 6ull * expert_mid_dim * sizeof(float) ||
        out->bytes < (uint64_t)out_dim * sizeof(float)) {
        return 0;
    }
    if (pack_fixed3 && resident_expert_base == 0u) return 0;
    const bool q4k_path = gate_type == 12u && down_type == 12u;
    if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
    if (q4k_path && getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL) {
        fprintf(stderr, "ds4: CUDA owned Q4 decode does not support gate/up auxiliary output\n");
        return 0;
    }
    if (!q4k_path && getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") != NULL) {
        fprintf(stderr, "ds4: CUDA owned IQ2 decode requires the LUT gate path\n");
        return 0;
    }
    const bool write_aux =
        !q4k_path && getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;

    if (resident_expert_base > UINT64_MAX / gate_expert_bytes ||
        resident_expert_count > UINT64_MAX / gate_expert_bytes ||
        resident_expert_base > UINT64_MAX / down_expert_bytes ||
        resident_expert_count > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_shift = (uint64_t)resident_expert_base * gate_expert_bytes;
    const uint64_t down_shift = (uint64_t)resident_expert_base * down_expert_bytes;
    const uint64_t gate_bytes = (uint64_t)resident_expert_count * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)resident_expert_count * down_expert_bytes;
    if (gate_offset > model_size || gate_shift > model_size - gate_offset ||
        gate_bytes > model_size - gate_offset - gate_shift ||
        up_offset > model_size || gate_shift > model_size - up_offset ||
        gate_bytes > model_size - up_offset - gate_shift ||
        down_offset > model_size || down_shift > model_size - down_offset ||
        down_bytes > model_size - down_offset - down_shift) {
        return 0;
    }

    const int logical_tier = ds4_tensor_device_idx(out);
    const char *gate_w = (const char *)cuda_resolve_weight_ptr(
            model_map, gate_offset + gate_shift, gate_bytes,
            logical_tier, "moe_owned_gate");
    const char *up_w = (const char *)cuda_resolve_weight_ptr(
            model_map, up_offset + gate_shift, gate_bytes,
            logical_tier, "moe_owned_up");
    const char *down_w = (const char *)cuda_resolve_weight_ptr(
            model_map, down_offset + down_shift, down_bytes,
            logical_tier, "moe_owned_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_bytes = (uint64_t)xq_blocks * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = 6ull * midq_blocks * sizeof(cuda_block_q8_K);
    const uint64_t down_output_bytes =
        (uint64_t)(pack_fixed3 ? 4u : 6u) * out_dim * sizeof(float);
    const uint64_t aux_bytes = 6ull * expert_mid_dim * sizeof(float);
    const uint64_t shared_q8_blocks = expert_in_dim / 32u;
    const uint64_t shared_q8_bytes = shared_q8_blocks * 32u;
    const uint64_t shared_scale_offset =
        (shared_q8_bytes + 15u) & ~15ull;
    const uint64_t shared_prequant_bytes =
        shared_scale_offset + shared_q8_blocks * sizeof(float);
    if (down->bytes < xq_bytes || down->bytes < down_output_bytes ||
        (down_output && down_output->bytes < down_output_bytes) ||
        gate->bytes < midq_bytes ||
        (shared_prequant &&
         (shared_prequant->bytes < shared_prequant_bytes ||
          ds4_tensor_device_idx(shared_prequant) != logical_tier)) ||
        (write_aux && (gate->bytes < aux_bytes || up->bytes < aux_bytes))) {
        return 0;
    }
    float *down_dst = (float *)(down_output ? down_output->ptr : down->ptr);
    cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
    cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;

    dim3 xq_grid(xq_blocks, 1, 1);
    if (shared_prequant) {
        int8_t *shared_xq = (int8_t *)shared_prequant->ptr;
        float *shared_scale = (float *)((char *)shared_prequant->ptr +
                                        shared_scale_offset);
        q8_K_q8_0_quantize_kernel<<<xq_grid, 256>>>(
                xq, shared_xq, shared_scale, (const float *)x->ptr,
                expert_in_dim, 1u);
    } else {
        q8_K_quantize_kernel<<<xq_grid, 256>>>(
                xq, (const float *)x->ptr, expert_in_dim, 1u);
    }
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe x quantize launch")) return 0;

    if (q4k_path) {
        dim3 gate_grid((expert_mid_dim + 7u) / 8u, 6u, 1u);
        moe_gate_up_mid_decode_q4K_owned_warp32_noaux_kernel<<<gate_grid, 256>>>(
                (float *)mid->ptr,
                gate_w,
                up_w,
                xq,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                xq_blocks,
                expert_mid_dim,
                6u,
                resident_expert_base,
                resident_expert_count,
                clamp);
    } else {
        dim3 gate_grid((expert_mid_dim + 31u) / 32u, 6u, 1u);
        moe_gate_up_mid_decode_lut_owned_qwarp32_kernel<<<gate_grid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                xq,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                xq_blocks,
                expert_mid_dim,
                6u,
                resident_expert_base,
                resident_expert_count,
                write_aux,
                clamp);
    }
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe gate/up launch")) return 0;

    dim3 midq_grid(midq_blocks, 6u, 1u);
    q8_K_quantize_owned_kernel<<<midq_grid, 256>>>(
            midq,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            expert_mid_dim,
            6u,
            resident_expert_base,
            resident_expert_count);
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe mid quantize launch")) return 0;

    dim3 down_grid((out_dim + 31u) / 32u, pack_fixed3 ? 4u : 6u, 1u);
    if (q4k_path && pack_fixed3) {
        moe_down_q4K_owned_packed_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else if (q4k_path) {
        moe_down_q4K_owned_slots_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else if (pack_fixed3) {
        moe_down_owned_packed_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else {
        moe_down_owned_slots_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    }
    return cuda_ok(cudaGetLastError(), "owned routed_moe down launch");
}

extern "C" int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split,
        uint32_t rows) {
    if (!out || !home_slots || !peer_slots || !selected || out_dim == 0u ||
        rows == 0u || rows > 65535u) {
        return 0;
    }
    const uint64_t row_elems = (uint64_t)rows * out_dim;
    if (row_elems > UINT64_MAX / (6u * sizeof(float))) return 0;
    const uint64_t out_bytes = row_elems * sizeof(float);
    const uint64_t slots_bytes = row_elems * 6u * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)rows * 6u * sizeof(int32_t);
    if (out->bytes < out_bytes || home_slots->bytes < slots_bytes ||
        peer_slots->bytes < slots_bytes || selected->bytes < selected_bytes) {
        return 0;
    }
    const dim3 grid((out_dim + 255u) / 256u, rows, 1u);
    moe_owned_slots_combine_fixed3_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            (const float *)home_slots->ptr,
            (const float *)peer_slots->ptr,
            (const int32_t *)selected->ptr,
            out_dim,
            expert_split);
    return cuda_ok(cudaGetLastError(),
                   "owned routed_moe slot rows combine launch");
}

extern "C" int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    return ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
            out, home_slots, peer_slots, selected,
            out_dim, expert_split, 1u);
}

extern "C" int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint64_t home_bytes = 6ull * out_dim * sizeof(float);
    const uint64_t peer_bytes = 4ull * out_dim * sizeof(float);
    if (!out || !home_slots || !peer_packed || !selected || out_dim == 0u ||
        out->bytes < (uint64_t)out_dim * sizeof(float) ||
        home_slots->bytes < home_bytes || peer_packed->bytes < peer_bytes ||
        selected->bytes < 6u * sizeof(int32_t)) {
        return 0;
    }
    moe_owned_packed_combine_fixed3_kernel<<<
            (out_dim + 255u) / 256u, 256>>>(
            (float *)out->ptr,
            (const float *)home_slots->ptr,
            (const float *)peer_packed->ptr,
            (const int32_t *)selected->ptr,
            out_dim,
            expert_split);
    return cuda_ok(cudaGetLastError(),
                   "owned routed_moe packed combine launch");
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t layer_index,
        bool force_resident) {
    if (add_in) {
        if (!ds4_gpu_add_tensor(out, out, add_in,
                                (uint32_t)(out->bytes / sizeof(float)))) return 0;
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1, force_resident ? 0 : 1, 0);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    (void)force_resident;
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens, 1, 0);
}

extern "C" int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t resident_expert_base,
        uint32_t resident_expert_count,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!selected || !weights || n_tokens == 0u || n_expert == 0u ||
        n_total_expert == 0u || resident_expert_count == 0u ||
        gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base ||
        resident_expert_base > UINT64_MAX / gate_expert_bytes ||
        resident_expert_base > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (pair_count > UINT32_MAX ||
        selected->bytes < pair_count * sizeof(int32_t) ||
        weights->bytes < pair_count * sizeof(float)) {
        return 0;
    }
    const uint64_t gate_shift =
        (uint64_t)resident_expert_base * gate_expert_bytes;
    const uint64_t down_shift =
        (uint64_t)resident_expert_base * down_expert_bytes;
    if (gate_offset > model_size || gate_shift > model_size - gate_offset ||
        up_offset > model_size || gate_shift > model_size - up_offset ||
        down_offset > model_size || down_shift > model_size - down_offset) {
        return 0;
    }
    moe_filter_owned_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            pair_count,
            n_total_expert,
            resident_expert_base,
            resident_expert_count);
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe pair filter launch")) return 0;

    return routed_moe_launch(
            out, gate, up, mid, down, model_map, model_size,
            gate_offset + gate_shift,
            up_offset + gate_shift,
            down_offset + down_shift,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, resident_expert_count, n_expert,
            clamp, x, layer_index, n_tokens, 0, 1);
}
