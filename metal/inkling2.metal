/* Inkling 2 Mini (GGUF architecture inkling2_loop) kernels.  Math mirrors the CPU oracle in
 * ds4_inkling2.inc (ik_* functions); the graph is in ds4_inkling2_gpu.inc and the wrappers in
 * ds4_inkling2_metal.inc.  Activations are f32 end to end; K and V caches are IEEE half.
 *
 * Summation order: every output is summed by a fixed pattern (lane l takes units l, l+32, ...,
 * then simd_sum) that does not depend on how many tokens share the dispatch, so a logit
 * does not depend on the chunk size except through the router. */

#define IKM_E      2304u
#define IKM_H      24u
#define IKM_HK     8u
#define IKM_D      128u
#define IKM_R      16u
#define IKM_FF     1152u
#define IKM_TOPK   8u
#define IKM_NEXP   128u
#define IKM_EXTENT 1024u
#define IKM_EPS    1e-6f

/* Bit tests: the library is compiled with fast math, where isnan()/isfinite() may be folded to constants. */
static inline bool ikm_finite(float x) { return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u; }
static inline half4 ikm_finite_h4(half4 x) {
    const ushort4 b = as_type<ushort4>(x);
    return select(x, half4(0.0h), (b & 0x7c00u) == 0x7c00u);
}

static inline float ikm_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

constant float ikm_e2m1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

/* UE4M3 block scale of NVFP4 (0 and 0x7f read as 0). */
static inline float ikm_ue4m3(uint x) {
    x &= 0x7fu;
    if (x == 0u || x == 0x7fu) return 0.0f;
    const uint e = x >> 3, m = x & 7u;
    if (e == 0u) return (float)m * 0.001953125f;                       /* m * 2^-9 */
    return as_type<float>(((e - 7u + 127u) << 23) | (m << 20));       /* (1 + m/8) * 2^(e-7) */
}

/* ------------------------------------------------------------------------- */
/* Weight unit readers: UE elements per unit, UB bytes per unit              */
/* ------------------------------------------------------------------------- */

#define IKM_LOAD_AT static inline void load_at(device const uchar *row, uint u, thread float *w) { load(row + u * UB, w); }
struct ikm_w_f32 {
    static constant constexpr uint UE = 32, UB = 128;
    static inline void load(device const uchar *u, thread float *w) {
        device const float4 *p = (device const float4 *)u;
        for (uint i = 0; i < 8; i++) { const float4 v = p[i]; w[4*i] = v.x; w[4*i+1] = v.y; w[4*i+2] = v.z; w[4*i+3] = v.w; }
    }
    IKM_LOAD_AT
};
struct ikm_w_f16 {
    static constant constexpr uint UE = 32, UB = 64;
    static inline void load(device const uchar *u, thread float *w) {
        device const half *p = (device const half *)u;
        for (uint i = 0; i < 32; i++) w[i] = (float)p[i];
    }
    IKM_LOAD_AT
};
struct ikm_w_bf16 {
    static constant constexpr uint UE = 32, UB = 64;
    static inline void load(device const uchar *u, thread float *w) {
        device const ushort *p = (device const ushort *)u;
        for (uint i = 0; i < 32; i++) w[i] = as_type<float>((uint)p[i] << 16);
    }
    IKM_LOAD_AT
};
struct ikm_w_q8_0 {
    static constant constexpr uint UE = 32, UB = 34;
    static inline void load(device const uchar *u, thread float *w) {
        const float d = (float)(*(device const half *)u);
        device const packed_char4 *q = (device const packed_char4 *)(u + 2);
        for (uint i = 0; i < 8; i++) {
            const char4 c = char4(q[i]);
            w[4*i] = d * (float)c.x; w[4*i+1] = d * (float)c.y; w[4*i+2] = d * (float)c.z; w[4*i+3] = d * (float)c.w;
        }
    }
    IKM_LOAD_AT
};
struct ikm_w_mxfp4 {   /* 32 nibbles, one E8M0 scale; byte j holds elements j and j+16 */
    static constant constexpr uint UE = 32, UB = 17;
    static inline void load(device const uchar *u, thread float *w) {
        const uint eb = u[0];
        const float d = as_type<float>(eb == 0u ? 0x00400000u : (eb << 23));
        for (uint j = 0; j < 16; j++) {
            const uint q = u[1 + j];
            w[j] = d * ikm_e2m1[q & 15u];
            w[16 + j] = d * ikm_e2m1[q >> 4];
        }
    }
    IKM_LOAD_AT
};
/* NVFP4 block (64 elements, 36 bytes: four UE4M3 scales, then 8 quant bytes per 16 elements; byte j holds
 * elements j and j+8) is read as four 16-element units: unit u is block u/4, sub-block u%4. */
struct ikm_w_nvfp4 {
    static constant constexpr uint UE = 16;
    static inline void load_at(device const uchar *row, uint u, thread float *w) {
        device const uchar *blk = row + (u >> 2) * 36u;
        const uint s = u & 3u;
        const float d = ikm_ue4m3((*(device const uint *)blk) >> (8u * s));
        const uint2 qq = *(device const uint2 *)(blk + 4u + 8u * s);
        for (uint j = 0; j < 4; j++) {
            const uint b0 = (qq.x >> (8u * j)) & 255u, b1 = (qq.y >> (8u * j)) & 255u;
            w[j] = d * ikm_e2m1[b0 & 15u];
            w[8 + j] = d * ikm_e2m1[b0 >> 4];
            w[4 + j] = d * ikm_e2m1[b1 & 15u];
            w[12 + j] = d * ikm_e2m1[b1 >> 4];
        }
    }
};

/* ------------------------------------------------------------------------- */
/* Dense matmul: y[t][r] = dot(weight row r, x[t])                           */
/* ------------------------------------------------------------------------- */

struct ikm_mm_args { uint in_dim, out_rows, n_tokens, row_bytes; };

#define IKM_NT  8u    /* tokens per tile */
#define IKM_NR  2u    /* rows per simdgroup */
#define IKM_NSG 4u    /* simdgroups per threadgroup */

template<typename W>
kernel void ikm_mm(constant ikm_mm_args &a [[buffer(0)]],
                   device const uchar *wt [[buffer(1)]],
                   device const float *x  [[buffer(2)]],
                   device float *y        [[buffer(3)]],
                   uint3 tg   [[threadgroup_position_in_grid]],
                   uint  sg   [[simdgroup_index_in_threadgroup]],
                   uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * IKM_NSG + sg) * IKM_NR;
    if (row0 >= a.out_rows) return;
    const uint t0 = tg.y * IKM_NT;
    const uint nt = min(IKM_NT, a.n_tokens - t0);
    const uint nunit = a.in_dim / W::UE;
    float acc[IKM_NR][IKM_NT];
    for (uint r = 0; r < IKM_NR; r++) for (uint t = 0; t < IKM_NT; t++) acc[r][t] = 0.0f;
    device const float *xt = x + (ulong)t0 * a.in_dim;
    for (uint u = lane; u < nunit; u += 32) {
        float w[IKM_NR][W::UE];
        for (uint r = 0; r < IKM_NR; r++)
            if (row0 + r < a.out_rows) W::load_at(wt + (ulong)(row0 + r) * a.row_bytes, u, w[r]);
        for (uint t = 0; t < nt; t++) {
            device const float4 *xp = (device const float4 *)(xt + (ulong)t * a.in_dim + u * W::UE);
            for (uint r = 0; r < IKM_NR; r++) {
                float s = acc[r][t];
                for (uint i = 0; i < W::UE / 4; i++) {
                    const float4 xv = xp[i];
                    s = fma(w[r][4*i], xv.x, s); s = fma(w[r][4*i+1], xv.y, s);
                    s = fma(w[r][4*i+2], xv.z, s); s = fma(w[r][4*i+3], xv.w, s);
                }
                acc[r][t] = s;
            }
        }
    }
    for (uint r = 0; r < IKM_NR; r++)
        for (uint t = 0; t < nt; t++) {
            const float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_rows) y[(ulong)(t0 + t) * a.out_rows + row0 + r] = s;
        }
}

#define IKM_MM(NAME, W) typedef decltype(ikm_mm<W>) ikm_mm_##NAME##_t; \
    template [[host_name("kernel_ik_mm_" #NAME)]] kernel ikm_mm_##NAME##_t ikm_mm<W>;
IKM_MM(f32, ikm_w_f32)
IKM_MM(f16, ikm_w_f16)
IKM_MM(bf16, ikm_w_bf16)
IKM_MM(q8_0, ikm_w_q8_0)
IKM_MM(mxfp4, ikm_w_mxfp4)
IKM_MM(nvfp4, ikm_w_nvfp4)


/* ------------------------------------------------------------------------- */
/* Prefill matmul on simdgroup matrices: 32 rows x 32 tokens per threadgroup, K in steps of 64, f32 math.     */
/* Each output sums over k in one fixed order, so it does not depend on which tokens share the tile.          */
/* ------------------------------------------------------------------------- */
template<typename W>
kernel void ikm_mma(constant ikm_mm_args &a [[buffer(0)]],
                    device const uchar *wt [[buffer(1)]],
                    device const float *x  [[buffer(2)]],
                    device float *y        [[buffer(3)]],
                    uint3 tg   [[threadgroup_position_in_grid]],
                    uint  tid  [[thread_index_in_threadgroup]],
                    uint  sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sw[32 * 64];
    threadgroup float sx[32 * 64];
    const uint r0 = tg.x * 32u, t0 = tg.y * 32u;
    const uint rb = (sg & 1u) * 16u, tb = (sg >> 1) * 16u;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint k0 = 0; k0 < a.in_dim; k0 += 64u) {
        if (tid < 64u) {
            const uint row = tid & 31u, half_ = tid >> 5;
            threadgroup float *dst = sw + row * 64u + half_ * 32u;
            if (r0 + row < a.out_rows) {
                float w[32];
                W::load_at(wt + (ulong)(r0 + row) * a.row_bytes, (k0 >> 5) + half_, w);
                for (uint i = 0; i < 32; i++) dst[i] = w[i];
            } else {
                for (uint i = 0; i < 32; i++) dst[i] = 0.0f;
            }
        }
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + 128u * i, tok = idx >> 4, k4 = idx & 15u;
            float4 v = float4(0.0f);
            if (t0 + tok < a.n_tokens) v = ((device const float4 *)(x + (ulong)(t0 + tok) * a.in_dim + k0))[k4];
            ((threadgroup float4 *)(sx + tok * 64u))[k4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 8; kk++) {
            simdgroup_float8x8 am[2], bm[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(am[i], sx + (tb + 8u * i) * 64u + kk * 8u, 64);
            for (uint j = 0; j < 2; j++) simdgroup_load(bm[j], sw + (rb + 8u * j) * 64u + kk * 8u, 64, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++)
        simdgroup_store(acc[i][j], sw + (tb + 8u * i) * 32u + rb + 8u * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; i++) {
        const uint idx = tid + 128u * i, tok = idx >> 5, r = idx & 31u;
        if (t0 + tok < a.n_tokens && r0 + r < a.out_rows) y[(ulong)(t0 + tok) * a.out_rows + r0 + r] = sw[tok * 32u + r];
    }
}
#define IKM_MMA(NAME, W) typedef decltype(ikm_mma<W>) ikm_mma_##NAME##_t; \
    template [[host_name("kernel_ik_mma_" #NAME)]] kernel ikm_mma_##NAME##_t ikm_mma<W>;
IKM_MMA(f32, ikm_w_f32)
IKM_MMA(bf16, ikm_w_bf16)
IKM_MMA(q8_0, ikm_w_q8_0)


/* ------------------------------------------------------------------------- */
/* Decode matvec (n <= 8 tokens): y[t][r] = dot(row r, x[t]).  Lanes split each row (Q8_0: 4 lanes per block,   */
/* 8 weights each, so 8 blocks are in flight); two rows per simdgroup, the scale applied once per block.       */
/* ------------------------------------------------------------------------- */

template<uint NRR>
kernel void ikm_mv_q8(constant ikm_mm_args &a [[buffer(0)]],
                              device const uchar *wt [[buffer(1)]],
                              device const float *x  [[buffer(2)]],
                              device float *y        [[buffer(3)]],
                              uint3 tg   [[threadgroup_position_in_grid]],
                              uint  sg   [[simdgroup_index_in_threadgroup]],
                              uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * IKM_NSG + sg) * NRR;
    if (row0 >= a.out_rows) return;
    const uint nt = a.n_tokens;
    const uint nb = a.in_dim / 32u;
    const uint ix = lane >> 2, il = lane & 3u;
    float acc[NRR][8];
    for (uint r = 0; r < NRR; r++) for (uint t = 0; t < 8; t++) acc[r][t] = 0.0f;
    for (uint ib = ix; ib < nb; ib += 8) {
        float yl[8][8];
        for (uint t = 0; t < nt; t++) {
            device const float4 *xp = (device const float4 *)(x + (ulong)t * a.in_dim + ib * 32u + il * 8u);
            const float4 v0 = xp[0], v1 = xp[1];
            yl[t][0] = v0.x; yl[t][1] = v0.y; yl[t][2] = v0.z; yl[t][3] = v0.w;
            yl[t][4] = v1.x; yl[t][5] = v1.y; yl[t][6] = v1.z; yl[t][7] = v1.w;
        }
        for (uint r = 0; r < NRR; r++) {
            if (row0 + r >= a.out_rows) break;
            device const uchar *blk = wt + (ulong)(row0 + r) * a.row_bytes + (ulong)ib * 34u;
            const float d = (float)(*(device const half *)blk);
            device const char *qs = (device const char *)blk + 2 + il * 8u;
            float q[8];
            for (uint i = 0; i < 8; i++) q[i] = (float)qs[i];
            for (uint t = 0; t < nt; t++) {
                float s = 0.0f;
                for (uint i = 0; i < 8; i++) s = fma(q[i], yl[t][i], s);
                acc[r][t] = fma(s, d, acc[r][t]);
            }
        }
    }
    for (uint r = 0; r < NRR; r++)
        for (uint t = 0; t < nt; t++) {
            const float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_rows) y[(ulong)t * a.out_rows + row0 + r] = s;
        }
}

/* F32 and BF16 rows: lane l takes float4 groups l, l+32, ... */
template<bool BF, uint NRR>
kernel void ikm_mv_wide(constant ikm_mm_args &a [[buffer(0)]],
                        device const uchar *wt [[buffer(1)]],
                        device const float *x  [[buffer(2)]],
                        device float *y        [[buffer(3)]],
                        uint3 tg   [[threadgroup_position_in_grid]],
                        uint  sg   [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * IKM_NSG + sg) * NRR;
    if (row0 >= a.out_rows) return;
    const uint nt = a.n_tokens;
    const uint n4 = a.in_dim / 4u;
    float acc[NRR][8];
    for (uint r = 0; r < NRR; r++) for (uint t = 0; t < 8; t++) acc[r][t] = 0.0f;
    for (uint i = lane; i < n4; i += 32) {
        float4 w[NRR];
        for (uint r = 0; r < NRR; r++) {
            if (row0 + r >= a.out_rows) { w[r] = float4(0.0f); continue; }
            device const uchar *row = wt + (ulong)(row0 + r) * a.row_bytes;
            if (BF) {
                const ushort4 u = ((device const ushort4 *)row)[i];
                w[r] = float4(as_type<float>((uint)u.x << 16), as_type<float>((uint)u.y << 16),
                              as_type<float>((uint)u.z << 16), as_type<float>((uint)u.w << 16));
            } else {
                w[r] = ((device const float4 *)row)[i];
            }
        }
        for (uint t = 0; t < nt; t++) {
            const float4 xv = ((device const float4 *)(x + (ulong)t * a.in_dim))[i];
            for (uint r = 0; r < NRR; r++) acc[r][t] += dot(w[r], xv);
        }
    }
    for (uint r = 0; r < NRR; r++)
        for (uint t = 0; t < nt; t++) {
            const float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_rows) y[(ulong)t * a.out_rows + row0 + r] = s;
        }
}
typedef decltype(ikm_mv_wide<false, 2>) ikm_mv_wide_t;
typedef decltype(ikm_mv_q8<2>) ikm_mv_q8_t;
template [[host_name("kernel_ik_mv_f32")]] kernel ikm_mv_wide_t ikm_mv_wide<false, 2>;
template [[host_name("kernel_ik_mv_bf16")]] kernel ikm_mv_wide_t ikm_mv_wide<true, 2>;
template [[host_name("kernel_ik_mv_q8_0")]] kernel ikm_mv_q8_t ikm_mv_q8<2>;
template [[host_name("kernel_ik_mv1_f32")]] kernel ikm_mv_wide_t ikm_mv_wide<false, 1>;
template [[host_name("kernel_ik_mv1_bf16")]] kernel ikm_mv_wide_t ikm_mv_wide<true, 1>;
template [[host_name("kernel_ik_mv1_q8_0")]] kernel ikm_mv_q8_t ikm_mv_q8<1>;


/* Batched decode (2..8 sessions): the same matvec with the token count NT fixed at compile time, so the accumulators and the activation
 * rows stay in registers (the runtime-count kernels above keep them in memory: 2.6x per token at 8 tokens).  Per token the operation order
 * is the one of the kernels above. */
template<uint NRR, uint NT>
kernel void ikm_mvn_q8(constant ikm_mm_args &a [[buffer(0)]],
                       device const uchar *wt [[buffer(1)]],
                       device const float *x  [[buffer(2)]],
                       device float *y        [[buffer(3)]],
                       uint3 tg   [[threadgroup_position_in_grid]],
                       uint  sg   [[simdgroup_index_in_threadgroup]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * IKM_NSG + sg) * NRR;
    if (row0 >= a.out_rows) return;
    const uint nb = a.in_dim / 32u;
    const uint ix = lane >> 2, il = lane & 3u;
    float acc[NRR][NT];
    for (uint r = 0; r < NRR; r++) for (uint t = 0; t < NT; t++) acc[r][t] = 0.0f;
    for (uint ib = ix; ib < nb; ib += 8) {
        float yl[NT][8];
        for (uint t = 0; t < NT; t++) {
            device const float4 *xp = (device const float4 *)(x + (ulong)t * a.in_dim + ib * 32u + il * 8u);
            const float4 v0 = xp[0], v1 = xp[1];
            yl[t][0] = v0.x; yl[t][1] = v0.y; yl[t][2] = v0.z; yl[t][3] = v0.w;
            yl[t][4] = v1.x; yl[t][5] = v1.y; yl[t][6] = v1.z; yl[t][7] = v1.w;
        }
        for (uint r = 0; r < NRR; r++) {
            if (row0 + r >= a.out_rows) break;
            device const uchar *blk = wt + (ulong)(row0 + r) * a.row_bytes + (ulong)ib * 34u;
            const float d = (float)(*(device const half *)blk);
            device const char *qs = (device const char *)blk + 2 + il * 8u;
            float q[8];
            for (uint i = 0; i < 8; i++) q[i] = (float)qs[i];
            for (uint t = 0; t < NT; t++) {
                float s = 0.0f;
                for (uint i = 0; i < 8; i++) s = fma(q[i], yl[t][i], s);
                acc[r][t] = fma(s, d, acc[r][t]);
            }
        }
    }
    for (uint r = 0; r < NRR; r++)
        for (uint t = 0; t < NT; t++) {
            const float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_rows) y[(ulong)t * a.out_rows + row0 + r] = s;
        }
}

template<bool BF, uint NRR, uint NT>
kernel void ikm_mvn_wide(constant ikm_mm_args &a [[buffer(0)]],
                         device const uchar *wt [[buffer(1)]],
                         device const float *x  [[buffer(2)]],
                         device float *y        [[buffer(3)]],
                         uint3 tg   [[threadgroup_position_in_grid]],
                         uint  sg   [[simdgroup_index_in_threadgroup]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * IKM_NSG + sg) * NRR;
    if (row0 >= a.out_rows) return;
    const uint n4 = a.in_dim / 4u;
    float acc[NRR][NT];
    for (uint r = 0; r < NRR; r++) for (uint t = 0; t < NT; t++) acc[r][t] = 0.0f;
    for (uint i = lane; i < n4; i += 32) {
        float4 w[NRR];
        for (uint r = 0; r < NRR; r++) {
            if (row0 + r >= a.out_rows) { w[r] = float4(0.0f); continue; }
            device const uchar *row = wt + (ulong)(row0 + r) * a.row_bytes;
            if (BF) {
                const ushort4 u = ((device const ushort4 *)row)[i];
                w[r] = float4(as_type<float>((uint)u.x << 16), as_type<float>((uint)u.y << 16),
                              as_type<float>((uint)u.z << 16), as_type<float>((uint)u.w << 16));
            } else {
                w[r] = ((device const float4 *)row)[i];
            }
        }
        for (uint t = 0; t < NT; t++) {
            const float4 xv = ((device const float4 *)(x + (ulong)t * a.in_dim))[i];
            for (uint r = 0; r < NRR; r++) acc[r][t] += dot(w[r], xv);
        }
    }
    for (uint r = 0; r < NRR; r++)
        for (uint t = 0; t < NT; t++) {
            const float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_rows) y[(ulong)t * a.out_rows + row0 + r] = s;
        }
}
typedef decltype(ikm_mvn_wide<false, 2, 2>) ikm_mvn_wide_t;
typedef decltype(ikm_mvn_q8<2, 2>) ikm_mvn_q8_t;
#define IKM_MVN_INST(NT) \
    template [[host_name("kernel_ik_mvn_f32_t" #NT)]] kernel ikm_mvn_wide_t ikm_mvn_wide<false, 2, NT>; \
    template [[host_name("kernel_ik_mvn_bf16_t" #NT)]] kernel ikm_mvn_wide_t ikm_mvn_wide<true, 2, NT>; \
    template [[host_name("kernel_ik_mvn_q8_0_t" #NT)]] kernel ikm_mvn_q8_t ikm_mvn_q8<2, NT>; \
    template [[host_name("kernel_ik_mvn1_f32_t" #NT)]] kernel ikm_mvn_wide_t ikm_mvn_wide<false, 1, NT>; \
    template [[host_name("kernel_ik_mvn1_bf16_t" #NT)]] kernel ikm_mvn_wide_t ikm_mvn_wide<true, 1, NT>; \
    template [[host_name("kernel_ik_mvn1_q8_0_t" #NT)]] kernel ikm_mvn_q8_t ikm_mvn_q8<1, NT>;
IKM_MVN_INST(2)
IKM_MVN_INST(3)
IKM_MVN_INST(4)
IKM_MVN_INST(5)
IKM_MVN_INST(6)
IKM_MVN_INST(7)
IKM_MVN_INST(8)


/* Several Q8_0 matvecs (one token, same input width) in one dispatch.  Segments are consecutive row ranges; segment g reads
 * weight rows at wb + woff[g], input x[xsel[g]] and writes y[g].  One row per simdgroup, four simdgroups per threadgroup. */
struct ikm_mvm_args { uint nseg, in_dim, row_bytes, pad; uint row_end[8]; uint xsel[8]; long woff[8]; };
kernel void kernel_ik_mvm_q8(constant ikm_mvm_args &a [[buffer(0)]],
                             device const uchar *wb [[buffer(1)]],
                             device const float *x0 [[buffer(2)]], device const float *x1 [[buffer(3)]], device const float *x2 [[buffer(4)]],
                             device float *y0 [[buffer(5)]], device float *y1 [[buffer(6)]], device float *y2 [[buffer(7)]],
                             device float *y3 [[buffer(8)]], device float *y4 [[buffer(9)]], device float *y5 [[buffer(10)]],
                             uint3 tg   [[threadgroup_position_in_grid]],
                             uint  sg   [[simdgroup_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x * 4u + sg;
    uint g = 0;
    while (g < a.nseg && row >= a.row_end[g]) g++;
    if (g >= a.nseg) return;
    const uint lrow = row - (g ? a.row_end[g - 1] : 0u);
    device const float *x = a.xsel[g] == 0u ? x0 : (a.xsel[g] == 1u ? x1 : x2);
    device float *y = g == 0 ? y0 : g == 1 ? y1 : g == 2 ? y2 : g == 3 ? y3 : g == 4 ? y4 : y5;
    device const uchar *rp = wb + a.woff[g] + (ulong)lrow * a.row_bytes;
    const uint nb = a.in_dim / 32u;
    const uint ix = lane >> 2, il = lane & 3u;
    float acc = 0.0f;
    #pragma clang loop unroll_count(3)
    for (uint ib = ix; ib < nb; ib += 8u) {
        device const uchar *blk = rp + (ulong)ib * 34u;
        const float d = (float)(*(device const half *)blk);
        device const ushort *qs = (device const ushort *)(blk + 2u + il * 8u);
        device const float4 *xp = (device const float4 *)(x + ib * 32u + il * 8u);
        const float4 xa = xp[0], xb = xp[1];
        const char2 c0 = as_type<char2>(qs[0]), c1 = as_type<char2>(qs[1]), c2 = as_type<char2>(qs[2]), c3 = as_type<char2>(qs[3]);
        float sum = (float)c0.x * xa.x;
        sum = fma((float)c0.y, xa.y, sum); sum = fma((float)c1.x, xa.z, sum); sum = fma((float)c1.y, xa.w, sum);
        sum = fma((float)c2.x, xb.x, sum); sum = fma((float)c2.y, xb.y, sum); sum = fma((float)c3.x, xb.z, sum); sum = fma((float)c3.y, xb.w, sum);
        acc = fma(sum, d, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[lrow] = acc;
}

/* ------------------------------------------------------------------------- */
/* Routed experts                                                            */
/* ------------------------------------------------------------------------- */

struct ikm_moe_args { uint in_dim, out_rows, n_tokens, row_bytes; ulong expert_bytes; uint list_cap, has_scale; };

/* gate/up: hidden unit j of expert e for every token routed to it, 8 tokens at a time:
 * ac[pair][j] = SiTU(gate_j * s, up_j * s). */
template<typename W>
kernel void ikm_moe_gu(constant ikm_moe_args &a [[buffer(0)]],
                       device const uchar *wt   [[buffer(1)]],
                       device const float *h    [[buffer(2)]],   /* [n][E] */
                       device const int *lists  [[buffer(3)]],
                       device const int *counts [[buffer(4)]],
                       device const float *gs   [[buffer(5)]],
                       device float *ac         [[buffer(6)]],   /* [n*8][FF] */
                       uint3 tg   [[threadgroup_position_in_grid]],
                       uint  sg   [[simdgroup_index_in_threadgroup]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    const uint e = tg.y;
    const uint cnt = (uint)counts[e];
    const uint j0 = (tg.x * IKM_NSG + sg) * IKM_NR;
    if (cnt == 0 || j0 >= IKM_FF) return;
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    const float sc = a.has_scale ? gs[e] : 1.0f;
    const uint nunit = IKM_E / W::UE;
    for (uint t0 = 0; t0 < cnt; t0 += IKM_NT) {
        const uint nt = min(IKM_NT, cnt - t0);
        device const float *xs[IKM_NT];
        int pair[IKM_NT];
        for (uint t = 0; t < IKM_NT; t++) {
            pair[t] = lists[e * a.list_cap + t0 + (t < nt ? t : 0)];
            xs[t] = h + (ulong)(pair[t] >> 3) * IKM_E;
        }
        float acc[IKM_NR][2][IKM_NT];
        for (uint r = 0; r < IKM_NR; r++) for (uint t = 0; t < IKM_NT; t++) { acc[r][0][t] = 0.0f; acc[r][1][t] = 0.0f; }
        for (uint u = lane; u < nunit; u += 32) {
            float w[IKM_NR][2][W::UE];
            for (uint r = 0; r < IKM_NR; r++)
                if (j0 + r < IKM_FF) {
                    W::load_at(ew + (ulong)(j0 + r) * a.row_bytes, u, w[r][0]);
                    W::load_at(ew + (ulong)(IKM_FF + j0 + r) * a.row_bytes, u, w[r][1]);
                }
            for (uint t = 0; t < nt; t++) {
                device const float4 *xp = (device const float4 *)(xs[t] + u * W::UE);
                for (uint r = 0; r < IKM_NR; r++)
                    for (uint c = 0; c < 2; c++) {
                        float s = acc[r][c][t];
                        for (uint i = 0; i < W::UE / 4; i++) {
                            const float4 xv = xp[i];
                            s = fma(w[r][c][4*i], xv.x, s); s = fma(w[r][c][4*i+1], xv.y, s);
                            s = fma(w[r][c][4*i+2], xv.z, s); s = fma(w[r][c][4*i+3], xv.w, s);
                        }
                        acc[r][c][t] = s;
                    }
            }
        }
        for (uint r = 0; r < IKM_NR; r++)
            for (uint t = 0; t < nt; t++) {
                const float g = simd_sum(acc[r][0][t]) * sc, u = simd_sum(acc[r][1][t]) * sc;
                if (lane == 0 && j0 + r < IKM_FF)   /* SiTU: softcapped gate * sigmoid(gate) * softcapped up */
                    ac[(ulong)pair[t] * IKM_FF + j0 + r] =
                        4.0f * precise::tanh(g / 4.0f) * ikm_sigmoid(g) * (25.0f * precise::tanh(u / 25.0f));
            }
    }
}

#define IKM_GU(NAME, W) typedef decltype(ikm_moe_gu<W>) ikm_gu_##NAME##_t; \
    template [[host_name("kernel_ik_moe_gu_" #NAME)]] kernel ikm_gu_##NAME##_t ikm_moe_gu<W>;
IKM_GU(f16, ikm_w_f16)
IKM_GU(bf16, ikm_w_bf16)
IKM_GU(q8_0, ikm_w_q8_0)
IKM_GU(mxfp4, ikm_w_mxfp4)
IKM_GU(nvfp4, ikm_w_nvfp4)

/* down: yd[pair][r] = dot(down row r of expert e, ac[pair]) (the expert scale and router weight come later). */
template<typename W>
kernel void ikm_moe_down(constant ikm_moe_args &a [[buffer(0)]],
                         device const uchar *wt   [[buffer(1)]],
                         device const float *ac   [[buffer(2)]],   /* [n*8][FF] */
                         device const int *lists  [[buffer(3)]],
                         device const int *counts [[buffer(4)]],
                         device float *yd         [[buffer(5)]],   /* [n*8][E] */
                         uint3 tg   [[threadgroup_position_in_grid]],
                         uint  sg   [[simdgroup_index_in_threadgroup]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    const uint e = tg.y;
    const uint cnt = (uint)counts[e];
    const uint r0 = (tg.x * IKM_NSG + sg) * IKM_NR;
    if (cnt == 0 || r0 >= IKM_E) return;
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    const uint nunit = IKM_FF / W::UE;
    for (uint t0 = 0; t0 < cnt; t0 += IKM_NT) {
        const uint nt = min(IKM_NT, cnt - t0);
        device const float *xs[IKM_NT];
        int pair[IKM_NT];
        for (uint t = 0; t < IKM_NT; t++) {
            pair[t] = lists[e * a.list_cap + t0 + (t < nt ? t : 0)];
            xs[t] = ac + (ulong)pair[t] * IKM_FF;
        }
        float acc[IKM_NR][IKM_NT];
        for (uint r = 0; r < IKM_NR; r++) for (uint t = 0; t < IKM_NT; t++) acc[r][t] = 0.0f;
        for (uint u = lane; u < nunit; u += 32) {
            float w[IKM_NR][W::UE];
            for (uint r = 0; r < IKM_NR; r++)
                if (r0 + r < IKM_E) W::load_at(ew + (ulong)(r0 + r) * a.row_bytes, u, w[r]);
            for (uint t = 0; t < nt; t++) {
                device const float4 *xp = (device const float4 *)(xs[t] + u * W::UE);
                for (uint r = 0; r < IKM_NR; r++) {
                    float s = acc[r][t];
                    for (uint i = 0; i < W::UE / 4; i++) {
                        const float4 xv = xp[i];
                        s = fma(w[r][4*i], xv.x, s); s = fma(w[r][4*i+1], xv.y, s);
                        s = fma(w[r][4*i+2], xv.z, s); s = fma(w[r][4*i+3], xv.w, s);
                    }
                    acc[r][t] = s;
                }
            }
        }
        for (uint r = 0; r < IKM_NR; r++)
            for (uint t = 0; t < nt; t++) {
                const float s = simd_sum(acc[r][t]);
                if (lane == 0 && r0 + r < IKM_E) yd[(ulong)pair[t] * IKM_E + r0 + r] = s;
            }
    }
}

#define IKM_DN(NAME, W) typedef decltype(ikm_moe_down<W>) ikm_dn_##NAME##_t; \
    template [[host_name("kernel_ik_moe_down_" #NAME)]] kernel ikm_dn_##NAME##_t ikm_moe_down<W>;
IKM_DN(f16, ikm_w_f16)
IKM_DN(bf16, ikm_w_bf16)
IKM_DN(q8_0, ikm_w_q8_0)
IKM_DN(mxfp4, ikm_w_mxfp4)
IKM_DN(nvfp4, ikm_w_nvfp4)


/* ------------------------------------------------------------------------- */
/* Decode (one token) NVFP4 experts: the eight picked experts in one dispatch, no lists, byte-lookup decode. */
/* ------------------------------------------------------------------------- */

/* lut[byte] = (value of the low nibble, value of the high nibble); one entry per thread of a 128/256-thread group. */
#define IKM_FILL_LUT(lut, tid, nthreads) if ((tid) < 16u) lut[(tid)] = ikm_e2m1[(tid)];

/* One 16-element NVFP4 unit against x (four float4: elements 0-3, 4-7, 8-11, 12-15): byte j holds elements j (low) and j+8 (high). */
static inline float ikm_nv_dot16(uint2 q, float4 x0, float4 x1, float4 x2, float4 x3, threadgroup const float *lut) {
    float s = 0.0f;
    for (uint j = 0; j < 4; j++) {
        const uint b0 = (q.x >> (8u * j)) & 255u, b1 = (q.y >> (8u * j)) & 255u;
        s = fma(lut[b0 & 15u], x0[j], s); s = fma(lut[b0 >> 4], x2[j], s);
        s = fma(lut[b1 & 15u], x1[j], s); s = fma(lut[b1 >> 4], x3[j], s);
    }
    return s;
}

/* gate/up: tg.y = pick slot k (expert picks[k]); each simdgroup makes two hidden units j (gate and up row each). */
kernel void kernel_ik_moe_gu1_nvfp4(constant ikm_moe_args &a [[buffer(0)]],
                                    device const uchar *wt   [[buffer(1)]],
                                    device const float *h    [[buffer(2)]],   /* [E] */
                                    device const int *picks  [[buffer(3)]],   /* [8] */
                                    device const float *gs   [[buffer(4)]],
                                    device float *ac         [[buffer(5)]],   /* [8][FF] */
                                    uint3 tg   [[threadgroup_position_in_grid]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float lut[16];
    IKM_FILL_LUT(lut, sg * 32u + lane, 128u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k = tg.y;
    h += (ulong)tg.z * IKM_E;                 /* tg.z = token of a batched decode (0 for one token) */
    picks += tg.z * IKM_TOPK;
    ac += (ulong)tg.z * IKM_TOPK * IKM_FF;
    const uint e = (uint)picks[k];
    const uint j0 = (tg.x * 4u + sg) * 2u;
    if (j0 >= IKM_FF) return;
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    device const uchar *row[4] = { ew + (ulong)j0 * a.row_bytes, ew + (ulong)(j0 + 1) * a.row_bytes,
                                   ew + (ulong)(IKM_FF + j0) * a.row_bytes, ew + (ulong)(IKM_FF + j0 + 1) * a.row_bytes };
    const uint s = lane & 3u;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma clang loop unroll(full)
    for (uint blk = lane >> 2; blk < IKM_E / 64u; blk += 8u) {
        device const float4 *xp = (device const float4 *)(h + blk * 64u + s * 16u);
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        for (uint r = 0; r < 4; r++) {
            device const uchar *b = row[r] + blk * 36u;
            const float d = ikm_ue4m3((*(device const uint *)b) >> (8u * s));
            const uint2 q = *(device const uint2 *)(b + 4u + 8u * s);
            acc[r] = fma(d, ikm_nv_dot16(q, x0, x1, x2, x3, lut), acc[r]);
        }
    }
    const float sc = a.has_scale ? gs[e] : 1.0f;
    for (uint r = 0; r < 2; r++) {
        const float g = simd_sum(acc[r]) * sc, u = simd_sum(acc[2 + r]) * sc;
        if (lane == 0)
            ac[(ulong)k * IKM_FF + j0 + r] = 4.0f * precise::tanh(g / 4.0f) * ikm_sigmoid(g) * (25.0f * precise::tanh(u / 25.0f));
    }
}

/* down + combine: one threadgroup = eight simdgroups (one per pick slot) x two output rows; the rows' partial sums of the
 * eight experts are combined in slot order: out[i] = sum_k wts[k] * down_scale[picks[k]] * dot_k(i). */
kernel void kernel_ik_moe_dc1_nvfp4(constant ikm_moe_args &a [[buffer(0)]],
                                    device const uchar *wt   [[buffer(1)]],
                                    device const float *ac   [[buffer(2)]],   /* [8][FF] */
                                    device const int *picks  [[buffer(3)]],
                                    device const float *wts  [[buffer(4)]],
                                    device const float *ds   [[buffer(5)]],
                                    device float *out        [[buffer(6)]],   /* [E] */
                                    uint3 tg   [[threadgroup_position_in_grid]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float lut[16];
    threadgroup float part[8][2];
    const uint tid = sg * 32u + lane;
    IKM_FILL_LUT(lut, tid, 256u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k = sg;
    ac += (ulong)tg.z * IKM_TOPK * IKM_FF;    /* tg.z = token of a batched decode (0 for one token) */
    picks += tg.z * IKM_TOPK;
    wts += tg.z * IKM_TOPK;
    out += (ulong)tg.z * IKM_E;
    const uint e = (uint)picks[k];
    const uint r0 = tg.x * 2u;
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    device const uchar *row0 = ew + (ulong)r0 * a.row_bytes, *row1 = row0 + a.row_bytes;
    const uint s = lane & 3u;
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma clang loop unroll(full)
    for (uint blk = lane >> 2; blk < IKM_FF / 64u; blk += 8u) {
        device const float4 *xp = (device const float4 *)(ac + (ulong)k * IKM_FF + blk * 64u + s * 16u);
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        device const uchar *b0 = row0 + blk * 36u, *b1 = row1 + blk * 36u;
        const float d0 = ikm_ue4m3((*(device const uint *)b0) >> (8u * s)), d1 = ikm_ue4m3((*(device const uint *)b1) >> (8u * s));
        const uint2 q0 = *(device const uint2 *)(b0 + 4u + 8u * s), q1 = *(device const uint2 *)(b1 + 4u + 8u * s);
        acc0 = fma(d0, ikm_nv_dot16(q0, x0, x1, x2, x3, lut), acc0);
        acc1 = fma(d1, ikm_nv_dot16(q1, x0, x1, x2, x3, lut), acc1);
    }
    const float p0 = simd_sum(acc0), p1 = simd_sum(acc1);
    if (lane == 0) { part[k][0] = p0; part[k][1] = p1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2u) {
        float sum = 0.0f;
        for (uint kk = 0; kk < IKM_TOPK; kk++) {
            const float w = wts[kk] * (a.has_scale ? ds[picks[kk]] : 1.0f);
            sum = fma(w, part[kk][tid], sum);
        }
        out[r0 + tid] = sum;
    }
}


/* ------------------------------------------------------------------------- */
/* Prefill experts on simdgroup matrices, tokens grouped by expert: one threadgroup = one 32-pair tile of one expert x 32 */
/* output rows (f32 math, K in steps of 64).  tiles[0].x = tile count, tiles[1 + slot] = (expert, first list index).     */
/* ------------------------------------------------------------------------- */
template<typename W>
kernel void ikm_moe_gu_mma(constant ikm_moe_args &a [[buffer(0)]],
                           device const uchar *wt   [[buffer(1)]],
                           device const float *h    [[buffer(2)]],
                           device const int *lists  [[buffer(3)]],
                           device const int *counts [[buffer(4)]],
                           device const float *gs   [[buffer(5)]],
                           device float *ac         [[buffer(6)]],
                           device const int2 *tiles [[buffer(7)]],
                           uint3 tg  [[threadgroup_position_in_grid]],
                           uint  tid [[thread_index_in_threadgroup]],
                           uint  sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sw[64 * 64];
    threadgroup float sx[32 * 64];
    threadgroup int spair[32];
    if (tg.y >= (uint)tiles[0].x) return;
    const int2 tl = tiles[1 + tg.y];
    const uint e = (uint)tl.x, t0 = (uint)tl.y;
    const uint nt = min(32u, (uint)counts[e] - t0);
    const uint j0 = tg.x * 32u;
    if (tid < 32u) spair[tid] = tid < nt ? lists[e * a.list_cap + t0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    const uint rb = (sg & 1u) * 16u, tb = (sg >> 1) * 16u;
    simdgroup_float8x8 ag[2][2], au[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) { ag[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f); au[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f); }
    constexpr uint UPR = 64u / W::UE;   /* weight units per row per k step */
    for (uint k0 = 0; k0 < a.in_dim; k0 += 64u) {
        for (uint u = tid; u < 64u * UPR; u += 128u) {
            const uint row = u / UPR, uu = u % UPR;
            const uint grow = row < 32u ? j0 + row : IKM_FF + j0 + (row - 32u);
            float w[W::UE];
            W::load_at(ew + (ulong)grow * a.row_bytes, k0 / W::UE + uu, w);
            threadgroup float *dst = sw + row * 64u + uu * W::UE;
            for (uint i = 0; i < W::UE; i++) dst[i] = w[i];
        }
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + 128u * i, tok = idx >> 4, k4 = idx & 15u;
            float4 v = float4(0.0f);
            if (tok < nt) v = ((device const float4 *)(h + (ulong)(spair[tok] >> 3) * IKM_E + k0))[k4];
            ((threadgroup float4 *)(sx + tok * 64u))[k4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 8; kk++) {
            simdgroup_float8x8 am[2], bg[2], bu[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(am[i], sx + (tb + 8u * i) * 64u + kk * 8u, 64);
            for (uint j = 0; j < 2; j++) simdgroup_load(bg[j], sw + (rb + 8u * j) * 64u + kk * 8u, 64, ulong2(0, 0), true);
            for (uint j = 0; j < 2; j++) simdgroup_load(bu[j], sw + (32u + rb + 8u * j) * 64u + kk * 8u, 64, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) {
                simdgroup_multiply_accumulate(ag[i][j], am[i], bg[j], ag[i][j]);
                simdgroup_multiply_accumulate(au[i][j], am[i], bu[j], au[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) {
        simdgroup_store(ag[i][j], sw + (tb + 8u * i) * 32u + rb + 8u * j, 32);
        simdgroup_store(au[i][j], sw + 1024u + (tb + 8u * i) * 32u + rb + 8u * j, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sc = a.has_scale ? gs[e] : 1.0f;
    for (uint i = 0; i < 8; i++) {
        const uint idx = tid + 128u * i, tok = idx >> 5, j = idx & 31u;
        if (tok < nt) {
            const float g = sw[tok * 32u + j] * sc, u = sw[1024u + tok * 32u + j] * sc;
            ac[(ulong)spair[tok] * IKM_FF + j0 + j] = 4.0f * precise::tanh(g / 4.0f) * ikm_sigmoid(g) * (25.0f * precise::tanh(u / 25.0f));
        }
    }
}
#define IKM_GUM(NAME, W) typedef decltype(ikm_moe_gu_mma<W>) ikm_gum_##NAME##_t; \
    template [[host_name("kernel_ik_moe_gum_" #NAME)]] kernel ikm_gum_##NAME##_t ikm_moe_gu_mma<W>;
IKM_GUM(f16, ikm_w_f16)
IKM_GUM(bf16, ikm_w_bf16)
IKM_GUM(q8_0, ikm_w_q8_0)
IKM_GUM(mxfp4, ikm_w_mxfp4)
IKM_GUM(nvfp4, ikm_w_nvfp4)

template<typename W>
kernel void ikm_moe_down_mma(constant ikm_moe_args &a [[buffer(0)]],
                             device const uchar *wt   [[buffer(1)]],
                             device const float *ac   [[buffer(2)]],
                             device const int *lists  [[buffer(3)]],
                             device const int *counts [[buffer(4)]],
                             device float *yd         [[buffer(5)]],
                             device const int2 *tiles [[buffer(6)]],
                             uint3 tg  [[threadgroup_position_in_grid]],
                             uint  tid [[thread_index_in_threadgroup]],
                             uint  sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sw[32 * 64];
    threadgroup float sx[32 * 64];
    threadgroup int spair[32];
    if (tg.y >= (uint)tiles[0].x) return;
    const int2 tl = tiles[1 + tg.y];
    const uint e = (uint)tl.x, t0 = (uint)tl.y;
    const uint nt = min(32u, (uint)counts[e] - t0);
    const uint r0 = tg.x * 32u;
    if (tid < 32u) spair[tid] = tid < nt ? lists[e * a.list_cap + t0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *ew = wt + (ulong)e * a.expert_bytes;
    const uint rb = (sg & 1u) * 16u, tb = (sg >> 1) * 16u;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    constexpr uint UPR = 64u / W::UE;
    for (uint k0 = 0; k0 < a.in_dim; k0 += 64u) {
        for (uint u = tid; u < 32u * UPR; u += 128u) {
            const uint row = u / UPR, uu = u % UPR;
            float w[W::UE];
            W::load_at(ew + (ulong)(r0 + row) * a.row_bytes, k0 / W::UE + uu, w);
            threadgroup float *dst = sw + row * 64u + uu * W::UE;
            for (uint i = 0; i < W::UE; i++) dst[i] = w[i];
        }
        for (uint i = 0; i < 4; i++) {
            const uint idx = tid + 128u * i, tok = idx >> 4, k4 = idx & 15u;
            float4 v = float4(0.0f);
            if (tok < nt) v = ((device const float4 *)(ac + (ulong)spair[tok] * IKM_FF + k0))[k4];
            ((threadgroup float4 *)(sx + tok * 64u))[k4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 8; kk++) {
            simdgroup_float8x8 am[2], bm[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(am[i], sx + (tb + 8u * i) * 64u + kk * 8u, 64);
            for (uint j = 0; j < 2; j++) simdgroup_load(bm[j], sw + (rb + 8u * j) * 64u + kk * 8u, 64, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++)
        simdgroup_store(acc[i][j], sw + (tb + 8u * i) * 32u + rb + 8u * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; i++) {
        const uint idx = tid + 128u * i, tok = idx >> 5, r = idx & 31u;
        if (tok < nt) yd[(ulong)spair[tok] * IKM_E + r0 + r] = sw[tok * 32u + r];
    }
}
#define IKM_DNM(NAME, W) typedef decltype(ikm_moe_down_mma<W>) ikm_dnm_##NAME##_t; \
    template [[host_name("kernel_ik_moe_dnm_" #NAME)]] kernel ikm_dnm_##NAME##_t ikm_moe_down_mma<W>;
IKM_DNM(f16, ikm_w_f16)
IKM_DNM(bf16, ikm_w_bf16)
IKM_DNM(q8_0, ikm_w_q8_0)
IKM_DNM(mxfp4, ikm_w_mxfp4)
IKM_DNM(nvfp4, ikm_w_nvfp4)

/* out[t][i] = sum_k wt[t][k] * down_scale[pick] * yd[t*8+k][i], k in order. */
struct ikm_combine_args { uint n_tokens, has_scale; };
kernel void kernel_ik_moe_combine(constant ikm_combine_args &a [[buffer(0)]],
                                  device const float *yd    [[buffer(1)]],
                                  device const float *wts   [[buffer(2)]],
                                  device const int *picks   [[buffer(3)]],
                                  device const float *ds    [[buffer(4)]],
                                  device float *out         [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint t = gid / IKM_E, i = gid % IKM_E;
    if (t >= a.n_tokens) return;
    float s = 0.0f;
    for (uint k = 0; k < IKM_TOPK; k++) {
        const uint p = t * IKM_TOPK + k;
        const float w = wts[p] * (a.has_scale ? ds[picks[p]] : 1.0f);
        s = fma(w, yd[(ulong)p * IKM_E + i], s);
    }
    out[gid] = s;
}

/* Router: logits [n][128] -> p = sigmoid, score = p + bias, top 8 (ties to the lower id), weights = p / sum * 8.
 * Eight rounds of a threadgroup arg-max (one thread per expert).  force_ntok != 0: expert ids come from the
 * forced-pick buffer (int32 [48][force_ntok][8], this slot), weights still from this run's probabilities.
 * picks is [n][8] for this slot. */
struct ikm_router_args { uint n_tokens, p0, slot, force_ntok; };
kernel void kernel_ik_router(constant ikm_router_args &a [[buffer(0)]],
                             device const float *logits  [[buffer(1)]],
                             device const float *bias    [[buffer(2)]],
                             device int *picks           [[buffer(3)]],
                             device float *wts           [[buffer(4)]],
                             device const int *forced    [[buffer(5)]],
                             uint t    [[threadgroup_position_in_grid]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint sgi  [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float p_s[IKM_NEXP], red_v[4];
    threadgroup uint red_i[4];
    threadgroup int sel_s[IKM_TOPK];
    if (t >= a.n_tokens) return;
    /* A NaN or infinite router logit makes the expert unselectable (score -1e30, weight 0): a NaN score would match no
     * expert at all, the pick would become 255 and the expert kernels would read past the 128-expert tensors. */
    const float lg = logits[(ulong)t * IKM_NEXP + tid];
    const bool lg_ok = ikm_finite(lg);
    const float p = lg_ok ? ikm_sigmoid(lg) : 0.0f;
    p_s[tid] = p;
    const float sc = lg_ok && ikm_finite(bias[tid]) ? p + bias[tid] : -1e30f;
    bool taken = false;
    for (uint k = 0; k < IKM_TOPK; k++) {
        const float v = taken ? -1e30f : sc;
        const float m = simd_max(v);
        if (lane == 0) red_v[sgi] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float M = max(max(red_v[0], red_v[1]), max(red_v[2], red_v[3]));
        const uint mi = simd_min((!taken && v == M) ? tid : 255u);   /* ties go to the lower id */
        if (lane == 0) red_i[sgi] = mi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pick = min(min(min(red_i[0], red_i[1]), min(red_i[2], red_i[3])), IKM_NEXP - 1u);
        if (tid == pick) taken = true;
        if (tid == 0) sel_s[k] = (int)pick;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        if (a.force_ntok) {
            device const int *f = forced + ((ulong)a.slot * a.force_ntok + a.p0 + t) * IKM_TOPK;
            for (uint k = 0; k < IKM_TOPK; k++) sel_s[k] = (uint)f[k] < IKM_NEXP ? f[k] : (int)k;   /* diagnostic path: stay in range */
        }
        float sum = 0.0f;
        for (uint k = 0; k < IKM_TOPK; k++) sum += p_s[sel_s[k]];
        for (uint k = 0; k < IKM_TOPK; k++) {
            picks[t * IKM_TOPK + k] = sel_s[k];
            wts[t * IKM_TOPK + k] = p_s[sel_s[k]] / (sum + 1e-20f) * 8.0f;
        }
    }
}

/* lists[e][i] = pair index (token*8+k) of the i-th pick of expert e, in token order; counts[e]. */
struct ikm_lists_args { uint n_pairs, list_cap, make_tiles, pad; };
kernel void kernel_ik_moe_lists(constant ikm_lists_args &a [[buffer(0)]],
                                device const int *picks [[buffer(1)]],
                                device int *lists      [[buffer(2)]],
                                device int *counts     [[buffer(3)]],
                                device int2 *tiles     [[buffer(4)]],
                                uint e [[thread_position_in_grid]]) {
    threadgroup int tcnt[IKM_NEXP];
    if (e >= IKM_NEXP) return;
    uint c = 0;
    for (uint i = 0; i < a.n_pairs; i++)
        if ((uint)picks[i] == e) lists[e * a.list_cap + c++] = (int)i;
    counts[e] = (int)c;
    if (!a.make_tiles) return;
    tcnt[e] = (int)c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {   /* work list for the grouped matmuls: (expert, first pair of a 32-pair tile) */
        int T = 0;
        for (uint x = 0; x < IKM_NEXP; x++)
            for (int t0 = 0; t0 < tcnt[x]; t0 += 32) tiles[1 + T++] = int2((int)x, t0);
        tiles[0] = int2(T, 0);
    }
}

/* ------------------------------------------------------------------------- */
/* Depth mix, norms, control, conv                                           */
/* ------------------------------------------------------------------------- */

/* out[q][t] = sum_s softmax_s(rmsnorm(S_s[t]) . query_q / sqrt(E)) * rmsnorm(S_s[t]).  States 0..ns-2 are streams
 * (index idx[s], each [n][E]); state ns-1 is `cur` ([n][E]).  One threadgroup (256 threads) per token. */
struct ikm_mix_args { uint n_tokens, ns, nq, stream_stride; uint idx[8]; };
kernel void kernel_ik_mix(constant ikm_mix_args &a [[buffer(0)]],
                          device const float *streams [[buffer(1)]],
                          device const float *cur     [[buffer(2)]],
                          device const float *queries [[buffer(3)]],
                          device float *out           [[buffer(4)]],
                          uint t    [[threadgroup_position_in_grid]],
                          uint tid  [[thread_position_in_threadgroup]],
                          uint sgi  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8][40];
    threadgroup float sh[40];
    threadgroup float wq_s[4][8];
    if (t >= a.n_tokens) return;
    const uint nq = min(a.nq, 4u);
    for (uint s = 0; s < a.ns; s++) {
        device const float *S = (s + 1 < a.ns) ? streams + (ulong)a.idx[s] * a.stream_stride + (ulong)t * IKM_E
                                               : cur + (ulong)t * IKM_E;
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (uint i = tid; i < IKM_E; i += 256) {
            const float v = S[i];
            ss = fma(v, v, ss);
            d0 = fma(v, queries[i], d0);
            if (nq > 1) d1 = fma(v, queries[IKM_E + i], d1);
            if (nq > 2) d2 = fma(v, queries[2 * IKM_E + i], d2);
            if (nq > 3) d3 = fma(v, queries[3 * IKM_E + i], d3);
        }
        ss = simd_sum(ss); d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        if (lane == 0) {
            red[sgi][s * 5] = ss; red[sgi][s * 5 + 1] = d0; red[sgi][s * 5 + 2] = d1; red[sgi][s * 5 + 3] = d2; red[sgi][s * 5 + 4] = d3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 40) { float v = 0.0f; for (uint g = 0; g < 8; g++) v += red[g][tid]; sh[tid] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < nq) {   /* softmax over the states for query tid; weight includes the state's 1/rms */
        float sc[8], mx = -1e30f, den = 0.0f;
        for (uint s = 0; s < a.ns; s++) {
            const float inv = 1.0f / precise::sqrt(sh[s * 5] / (float)IKM_E + IKM_EPS);
            sc[s] = sh[s * 5 + 1 + tid] * inv / precise::sqrt((float)IKM_E);
            mx = max(mx, sc[s]);
        }
        for (uint s = 0; s < a.ns; s++) { sc[s] = precise::exp(sc[s] - mx); den += sc[s]; }
        for (uint s = 0; s < a.ns; s++) {
            const float inv = 1.0f / precise::sqrt(sh[s * 5] / (float)IKM_E + IKM_EPS);
            wq_s[tid][s] = sc[s] / den * inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < IKM_E; i += 256) {
        float o[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint s = 0; s < a.ns; s++) {
            device const float *S = (s + 1 < a.ns) ? streams + (ulong)a.idx[s] * a.stream_stride + (ulong)t * IKM_E
                                                   : cur + (ulong)t * IKM_E;
            const float v = S[i];
            for (uint q = 0; q < nq; q++) o[q] = fma(wq_s[q][s], v, o[q]);
        }
        for (uint q = 0; q < nq; q++) out[((ulong)q * a.n_tokens + t) * IKM_E + i] = o[q];
    }
}

/* y[row] = x[row] * rsqrt(mean(x^2) + eps) * w (w optional, in place allowed); rows may be gathered by `rows`. */
struct ikm_rms_args { uint dim, n_rows, has_w, gather; };
kernel void kernel_ik_rms_rows(constant ikm_rms_args &a [[buffer(0)]],
                               device const float *x    [[buffer(1)]],
                               device const float *w    [[buffer(2)]],
                               device float *y          [[buffer(3)]],
                               device const int *rows   [[buffer(4)]],
                               uint r    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint sgi  [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (r >= a.n_rows) return;
    device const float *xr = x + (ulong)(a.gather ? (uint)rows[r] : r) * a.dim;
    device float *yr = y + (ulong)r * a.dim;
    float ss = 0.0f;
    for (uint i = tid; i < a.dim; i += 256) ss = fma(xr[i], xr[i], ss);
    ss = simd_sum(ss);
    if (lane == 0) red[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8; g++) tot += red[g];
    const float inv = 1.0f / precise::sqrt(tot / (float)a.dim + IKM_EPS);
    for (uint i = tid; i < a.dim; i += 256) yr[i] = xr[i] * inv * (a.has_w ? w[i] : 1.0f);
}

/* Control prep: g[t] = [rmsnorm(cur[t]) | rmsnorm(msg[t])]. */
kernel void kernel_ik_ctl_prep(constant uint &n [[buffer(0)]],
                               device const float *cur [[buffer(1)]],
                               device const float *msg [[buffer(2)]],
                               device float *g         [[buffer(3)]],
                               uint t    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint sgi  [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2][8];
    if (t >= n) return;
    device const float *c = cur + (ulong)t * IKM_E, *m = msg + (ulong)t * IKM_E;
    float s0 = 0.0f, s1 = 0.0f;
    for (uint i = tid; i < IKM_E; i += 256) { s0 = fma(c[i], c[i], s0); s1 = fma(m[i], m[i], s1); }
    s0 = simd_sum(s0); s1 = simd_sum(s1);
    if (lane == 0) { red[0][sgi] = s0; red[1][sgi] = s1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t0 = 0.0f, t1 = 0.0f;
    for (uint k = 0; k < 8; k++) { t0 += red[0][k]; t1 += red[1][k]; }
    const float i0 = 1.0f / precise::sqrt(t0 / (float)IKM_E + IKM_EPS), i1 = 1.0f / precise::sqrt(t1 / (float)IKM_E + IKM_EPS);
    for (uint i = tid; i < IKM_E; i += 256) { g[(ulong)t * 2 * IKM_E + i] = c[i] * i0; g[(ulong)t * 2 * IKM_E + IKM_E + i] = m[i] * i1; }
}

/* out = cur - tanh(forget) * cur + tanh(gain) * msg  (out may alias cur) */
kernel void kernel_ik_ctl_apply(constant uint &count [[buffer(0)]],
                                device const float *cur  [[buffer(1)]],
                                device const float *msg  [[buffer(2)]],
                                device const float *gain [[buffer(3)]],
                                device const float *forget [[buffer(4)]],
                                device float *out        [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float c = cur[i];
    out[i] = c + (-precise::tanh(forget[i]) * c + precise::tanh(gain[i]) * msg[i]);
}

/* Causal depthwise 4-tap conv with residual: out = (x + conv(x) [+ dyn]) * scale, tail = three previous inputs. */
struct ikm_conv_args { uint n_tokens, C, hd, has_dyn; float scale; uint tail_off; };
kernel void kernel_ik_conv(constant ikm_conv_args &a [[buffer(0)]],
                           device const float *x    [[buffer(1)]],   /* [n][C] */
                           device const float *tail [[buffer(2)]],   /* [C][3] (already offset) */
                           device const float *w    [[buffer(3)]],   /* [C][4], tap 0 = oldest */
                           device const float *dyn  [[buffer(4)]],   /* [n][C/hd][4] */
                           device float *out        [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint ch = gid.x, t = gid.y;
    if (ch >= a.C || t >= a.n_tokens) return;
    float col[4];
    for (uint j = 0; j < 4; j++) {
        const int idx = (int)t + (int)j - 3;   /* window oldest..newest = x[t-3 .. t] */
        col[j] = idx >= 0 ? x[(ulong)idx * a.C + ch] : tail[ch * 3 + (uint)(3 + idx)];
    }
    device const float *wc = w + ch * 4;
    float y = wc[0] * col[0] + wc[1] * col[1] + wc[2] * col[2] + wc[3] * col[3];
    if (a.has_dyn) {
        device const float *d = dyn + (ulong)t * (a.C / a.hd) * 4 + (ch / a.hd) * 4;
        for (uint lag = 0; lag < 4; lag++) y += d[lag] * col[3 - lag];
    }
    out[(ulong)t * a.C + ch] = (col[3] + y) * a.scale;
}

/* tail = last three inputs of [old tail, x[0..n)] (in place; one thread per channel). */
kernel void kernel_ik_conv_tail(constant ikm_conv_args &a [[buffer(0)]],
                                device const float *x    [[buffer(1)]],
                                device float *tail       [[buffer(2)]],
                                uint ch [[thread_position_in_grid]]) {
    if (ch >= a.C) return;
    float nt[3];
    for (uint j = 0; j < 3; j++) {
        const uint idx = a.n_tokens + j;   /* index in [tail(3), x(n)] */
        nt[j] = idx < 3 ? tail[ch * 3 + idx] : x[(ulong)(idx - 3) * a.C + ch];
    }
    for (uint j = 0; j < 3; j++) tail[ch * 3 + j] = nt[j];
}


/* ------------------------------------------------------------------------- */
/* Decode (one token) fused kernels                                          */
/* ------------------------------------------------------------------------- */

/* Depth mix for one token with the states spread over the simdgroups (state s on simdgroup s, up to 8 states):
 * same math as kernel_ik_mix, shorter dependent chain.  One threadgroup of 256 threads. */
kernel void kernel_ik_mix_p(constant ikm_mix_args &a [[buffer(0)]],
                            device const float *streams [[buffer(1)]],
                            device const float *cur     [[buffer(2)]],
                            device const float *queries [[buffer(3)]],
                            device float *out           [[buffer(4)]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint sgi  [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8][5];
    threadgroup float wq_s[4][8];
    const uint nq = min(a.nq, 4u);
    if (sgi < a.ns) {
        const uint s = sgi;
        device const float4 *S = (device const float4 *)((s + 1 < a.ns) ? streams + (ulong)a.idx[s] * a.stream_stride : cur);
        device const float4 *q0 = (device const float4 *)queries;
        float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        #pragma clang loop unroll_count(6)
        for (uint i = lane; i < IKM_E / 4u; i += 32u) {
            const float4 v = S[i];
            ss += dot(v, v);
            d0 += dot(v, q0[i]);
            if (nq > 1) d1 += dot(v, q0[IKM_E / 4u + i]);
            if (nq > 2) d2 += dot(v, q0[2u * IKM_E / 4u + i]);
            if (nq > 3) d3 += dot(v, q0[3u * IKM_E / 4u + i]);
        }
        ss = simd_sum(ss); d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        if (lane == 0) { red[s][0] = ss; red[s][1] = d0; red[s][2] = d1; red[s][3] = d2; red[s][4] = d3; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < nq) {
        float sc[8], mx = -1e30f, den = 0.0f;
        for (uint s = 0; s < a.ns; s++) {
            const float inv = 1.0f / precise::sqrt(red[s][0] / (float)IKM_E + IKM_EPS);
            sc[s] = red[s][1 + tid] * inv / precise::sqrt((float)IKM_E);
            mx = max(mx, sc[s]);
        }
        for (uint s = 0; s < a.ns; s++) { sc[s] = precise::exp(sc[s] - mx); den += sc[s]; }
        for (uint s = 0; s < a.ns; s++) {
            const float inv = 1.0f / precise::sqrt(red[s][0] / (float)IKM_E + IKM_EPS);
            wq_s[tid][s] = sc[s] / den * inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* weights of unused state slots are zero so the loop has a constant trip count and its loads can be issued together */
    if (tid < 32u) for (uint q = 0; q < 4u; q++) for (uint s = a.ns; s < 8u; s++) if (tid == 0u) wq_s[q][s] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float *Sp[8];
    for (uint s = 0; s < 8u; s++) Sp[s] = (s + 1 < a.ns) ? streams + (ulong)a.idx[s] * a.stream_stride : cur;
    #pragma clang loop unroll_count(3)
    for (uint i = tid; i < IKM_E; i += 256) {
        float v[8];
        for (uint s = 0; s < 8u; s++) v[s] = Sp[s][i];
        for (uint q = 0; q < nq; q++) {
            float o = 0.0f;
            for (uint s = 0; s < 8u; s++) o = fma(wq_s[q][s], v[s], o);
            out[(ulong)q * IKM_E + i] = o;
        }
    }
}

/* One token: conv (with optional dynamic part) and the tail update in one dispatch, plus an optional residual:
 * out = (x + conv(x) [+ dyn]) * scale [+ addend].  One thread per channel; out may alias addend. */
struct ikm_conv1_args { uint C, hd, has_dyn, has_add; float scale; };
kernel void kernel_ik_conv1(constant ikm_conv1_args &a [[buffer(0)]],
                            device const float *x    [[buffer(1)]],
                            device float *tail       [[buffer(2)]],
                            device const float *w    [[buffer(3)]],
                            device const float *dyn  [[buffer(4)]],
                            device const float *addv [[buffer(5)]],
                            device float *out        [[buffer(6)]],
                            uint ch [[thread_position_in_grid]]) {
    if (ch >= a.C) return;
    float col[4];
    for (uint j = 0; j < 3; j++) col[j] = tail[ch * 3 + j];
    col[3] = x[ch];
    device const float *wc = w + ch * 4;
    float y = wc[0] * col[0] + wc[1] * col[1] + wc[2] * col[2] + wc[3] * col[3];
    if (a.has_dyn) {
        device const float *d = dyn + (ch / a.hd) * 4;
        for (uint lag = 0; lag < 4; lag++) y += d[lag] * col[3 - lag];
    }
    float o = (col[3] + y) * a.scale;
    if (a.has_add) o += addv[ch];
    out[ch] = o;
    tail[ch * 3] = col[1]; tail[ch * 3 + 1] = col[2]; tail[ch * 3 + 2] = col[3];
}


/* Loop controllers of one token, batched over up to three controllers.
 * d[j][r] = sum_i W_down[j][r][i] * [rmsnorm(cur_j) | rmsnorm(msg_j)][i]   (one simdgroup per (j, r)) */
struct ikm_ctld_args { uint nctl, cur_stride, msg_stride, pad; long woff[3]; };
kernel void kernel_ik_ctl_down(constant ikm_ctld_args &a [[buffer(0)]],
                               device const uchar *wb  [[buffer(1)]],
                               device const float *cur [[buffer(2)]],
                               device const float *msg [[buffer(3)]],
                               device float *d         [[buffer(4)]],
                               uint3 tg   [[threadgroup_position_in_grid]],
                               uint  sg   [[simdgroup_index_in_threadgroup]],
                               uint  lane [[thread_index_in_simdgroup]]) {
    const uint gr = tg.x * 4u + sg;
    if (gr >= a.nctl * 64u) return;
    const uint j = gr >> 6, r = gr & 63u;
    device const float4 *w4 = (device const float4 *)(wb + a.woff[j] + (ulong)r * (2u * IKM_E * 4u));
    device const float4 *c4 = (device const float4 *)(cur + (ulong)j * a.cur_stride);
    device const float4 *m4 = (device const float4 *)(msg + (ulong)j * a.msg_stride);
    float dc = 0.0f, dm = 0.0f, sc = 0.0f, sm = 0.0f;
    #pragma clang loop unroll_count(6)
    for (uint i = lane; i < IKM_E / 4u; i += 32u) {
        const float4 c = c4[i], m = m4[i];
        dc += dot(w4[i], c); dm += dot(w4[IKM_E / 4u + i], m);
        sc += dot(c, c); sm += dot(m, m);
    }
    dc = simd_sum(dc); dm = simd_sum(dm); sc = simd_sum(sc); sm = simd_sum(sm);
    if (lane == 0) {
        const float ic = 1.0f / precise::sqrt(sc / (float)IKM_E + IKM_EPS), im = 1.0f / precise::sqrt(sm / (float)IKM_E + IKM_EPS);
        d[gr] = dc * ic + dm * im;
    }
}

/* out_j[i] = cur_j[i] + (-tanh(forget_j[i]) * cur_j[i] + tanh(gain_j[i]) * msg_j[i]), gain/forget = W[j][i] . d[j] (64 wide). */
struct ikm_ctla_args { uint nctl, cur_stride, msg_stride, pad; long wg[3]; long wf[3]; };
kernel void kernel_ik_ctl_apply_m(constant ikm_ctla_args &a [[buffer(0)]],
                                  device const uchar *wb  [[buffer(1)]],
                                  device const float *cur [[buffer(2)]],
                                  device const float *msg [[buffer(3)]],
                                  device const float *d   [[buffer(4)]],
                                  device float *o0 [[buffer(5)]], device float *o1 [[buffer(6)]], device float *o2 [[buffer(7)]],
                                  uint3 tg   [[threadgroup_position_in_grid]],
                                  uint  sg   [[simdgroup_index_in_threadgroup]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    const uint gr = tg.x * 4u + sg;
    if (gr >= a.nctl * IKM_E) return;
    const uint j = gr / IKM_E, i = gr % IKM_E;
    const float2 dd = ((device const float2 *)(d + j * 64u))[lane];
    const float2 wg = ((device const float2 *)(wb + a.wg[j] + (ulong)i * 256u))[lane];
    const float2 wf = ((device const float2 *)(wb + a.wf[j] + (ulong)i * 256u))[lane];
    const float pg = simd_sum(wg.x * dd.x + wg.y * dd.y), pf = simd_sum(wf.x * dd.x + wf.y * dd.y);
    if (lane == 0) {
        const float c = cur[(ulong)j * a.cur_stride + i], m = msg[(ulong)j * a.msg_stride + i];
        const float o = c + (-precise::tanh(pf) * c + precise::tanh(pg) * m);
        device float *out = j == 0 ? o0 : (j == 1 ? o1 : o2);
        out[i] = o;
    }
}

/* Up to three rms norms of one token with their own weights (same math as kernel_ik_rms_rows); in place allowed. */
struct ikm_rms3_args { uint dim, nrow, pad0, pad1; long woff[3]; };
kernel void kernel_ik_rms3(constant ikm_rms3_args &a [[buffer(0)]],
                           device const uchar *wb [[buffer(1)]],
                           device const float *x0 [[buffer(2)]], device const float *x1 [[buffer(3)]], device const float *x2 [[buffer(4)]],
                           device float *y0 [[buffer(5)]], device float *y1 [[buffer(6)]], device float *y2 [[buffer(7)]],
                           uint r    [[threadgroup_position_in_grid]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint sgi  [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (r >= a.nrow) return;
    device const float *xr = r == 0 ? x0 : (r == 1 ? x1 : x2);
    device float *yr = r == 0 ? y0 : (r == 1 ? y1 : y2);
    device const float *w = (device const float *)(wb + a.woff[r]);
    /* dim is 2304 = 9 x 256: read each thread's nine values and weights once, up front (loads in flight together) */
    float xv[9], wv[9];
    for (uint k = 0; k < 9u; k++) { xv[k] = xr[tid + k * 256u]; wv[k] = w[tid + k * 256u]; }
    float ss = 0.0f;
    for (uint k = 0; k < 9u; k++) ss = fma(xv[k], xv[k], ss);
    ss = simd_sum(ss);
    if (lane == 0) red[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8; g++) tot += red[g];
    const float inv = 1.0f / precise::sqrt(tot / (float)a.dim + IKM_EPS);
    for (uint k = 0; k < 9u; k++) yr[tid + k * 256u] = xv[k] * inv * wv[k];
}

/* ------------------------------------------------------------------------- */
/* Attention                                                                 */
/* ------------------------------------------------------------------------- */

/* Q: per (token, head) multiply by q_clip, clip the l2 norm to 22.627417, times tau on global layers;
 * R likewise with rel_clip and limit 8. */
struct ikm_qr_args { uint n_tokens, global; };
kernel void kernel_ik_qr_prep(constant ikm_qr_args &a [[buffer(0)]],
                              device float *Q          [[buffer(1)]],
                              device float *R          [[buffer(2)]],
                              device const float *qclip [[buffer(3)]],
                              device const float *rclip [[buffer(4)]],
                              device const float *tau  [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg.x, t = tg.y;
    if (t >= a.n_tokens) return;
    device float4 *q4 = (device float4 *)(Q + ((ulong)t * IKM_H + h) * IKM_D);
    device const float4 *c4 = (device const float4 *)(qclip + h * IKM_D);
    float4 v = q4[lane] * c4[lane];
    float ss = simd_sum(dot(v, v));
    float nrm = precise::sqrt(ss);
    float s = 22.627417f / max(nrm, 22.627417f);
    v *= s;
    if (a.global) v *= tau[t];
    q4[lane] = v;
    device float *rr = R + ((ulong)t * IKM_H + h) * IKM_R;
    float rv = lane < IKM_R ? rr[lane] * rclip[h * IKM_R + lane] : 0.0f;
    const float rs = simd_sum(rv * rv);
    const float rn = precise::sqrt(rs);
    float rscale = 8.0f / max(rn, 8.0f);
    rv *= rscale;
    if (a.global) rv *= tau[t];
    if (lane < IKM_R) rr[lane] = rv;
}

/* K: l2 clip then half into the cache; V: rms norm then half.  Row = (p0 + t) % rows. */
struct ikm_kv_args { uint n_tokens, p0, rows, pad; };
kernel void kernel_ik_kv_store(constant ikm_kv_args &a [[buffer(0)]],
                               device const float *Kc   [[buffer(1)]],
                               device const float *Vc   [[buffer(2)]],
                               device half *kcache      [[buffer(3)]],
                               device half *vcache      [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint g = tg.x, t = tg.y;
    if (t >= a.n_tokens) return;
    const ulong src = ((ulong)t * IKM_HK + g) * IKM_D;
    const ulong dst = ((ulong)((a.p0 + t) % a.rows) * IKM_HK + g) * IKM_D;
    float4 k = ((device const float4 *)(Kc + src))[lane];
    float4 v = ((device const float4 *)(Vc + src))[lane];
    const float kn = precise::sqrt(simd_sum(dot(k, k)));
    k *= 22.627417f / max(kn, 22.627417f);
    const float vs = simd_sum(dot(v, v));
    v *= 1.0f / precise::sqrt(vs / (float)IKM_D + IKM_EPS);
    /* Never store a NaN or infinity: tiled attention reads whole 8-key tiles and multiplies masked rows by 0, so one bad row
     * would poison every later request that shares the cache (it is not cleared by a session reset). */
    ((device half4 *)(kcache + dst))[lane] = ikm_finite_h4(half4(k));
    ((device half4 *)(vcache + dst))[lane] = ikm_finite_h4(half4(v));
}

/* O[t][h] = softmax(q.k/sqrt(D) + rel_bias(distance) - sink, virtual key logit 0 value 0) . V, times sigmoid(gate).
 * One threadgroup (256 threads) per (head, token); keys in blocks of 256 with an online softmax. */
struct ikm_attn_args { uint n_tokens, p0, rows, local; };
kernel void kernel_ik_attn(constant ikm_attn_args &a [[buffer(0)]],
                           device const float *Q     [[buffer(1)]],
                           device const float *R     [[buffer(2)]],
                           device const float *sink  [[buffer(3)]],
                           device const float *gate  [[buffer(4)]],
                           device const float *relb  [[buffer(5)]],
                           device const half *K      [[buffer(6)]],
                           device const half *V      [[buffer(7)]],
                           device float *O           [[buffer(8)]],
                           uint3 tg3  [[threadgroup_position_in_grid]],
                           uint3 tid3 [[thread_position_in_threadgroup]],
                           uint  sgi  [[simdgroup_index_in_threadgroup]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float q_s[IKM_D], r_s[IKM_R], sc[256], red[8], part[IKM_D];
    const uint tid = tid3.x;
    const uint h = tg3.x, t = tg3.y, g = h / (IKM_H / IKM_HK);
    if (t >= a.n_tokens) return;
    const uint p = a.p0 + t;
    const uint first = (a.local && p >= IKM_EXTENT) ? p - (IKM_EXTENT - 1) : 0u;
    if (tid < IKM_D) q_s[tid] = Q[((ulong)t * IKM_H + h) * IKM_D + tid];
    if (tid < IKM_R) r_s[tid] = R[((ulong)t * IKM_H + h) * IKM_R + tid];
    const float sinkv = sink[(ulong)t * IKM_H + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = 0.0f, den = 1.0f, acc = 0.0f;
    const uint jj = tid & 127u, half_id = tid >> 7;
    for (uint kb = first; kb <= p; kb += 256) {
        const uint k = kb + tid;
        float s = -INFINITY;
        if (k <= p) {
            const uint d = p - k;
            float b = 0.0f;
            if (d < IKM_EXTENT) for (uint j = 0; j < IKM_R; j++) b = fma(r_s[j], relb[d * IKM_R + j], b);
            device const half4 *kr = (device const half4 *)(K + (ulong)(k % a.rows) * IKM_HK * IKM_D + g * IKM_D);
            float dt = 0.0f;
            for (uint i = 0; i < IKM_D / 4; i++) {
                const float4 kv = float4(kr[i]);
                dt = fma(q_s[4*i], kv.x, dt); dt = fma(q_s[4*i+1], kv.y, dt);
                dt = fma(q_s[4*i+2], kv.z, dt); dt = fma(q_s[4*i+3], kv.w, dt);
            }
            s = dt / 11.313708499f + b - sinkv;
        }
        const float bm = simd_max(s);
        if (lane == 0) red[sgi] = bm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bmax = red[0];
        for (uint i = 1; i < 8; i++) bmax = max(bmax, red[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float mnew = max(m, bmax);
        const float e = (k <= p) ? precise::exp(s - mnew) : 0.0f;
        sc[tid] = e;
        const float es = simd_sum(e);
        if (lane == 0) red[sgi] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8; i++) tot += red[i];
        const float rescale = precise::exp(m - mnew);
        den = den * rescale + tot;
        acc *= rescale;
        m = mnew;
        const uint nk = min(256u, p - kb + 1u);
        #pragma clang loop unroll_count(16)
        for (uint kk = half_id; kk < nk; kk += 2)
            acc = fma(sc[kk], (float)V[(ulong)((kb + kk) % a.rows) * IKM_HK * IKM_D + g * IKM_D + jj], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (half_id == 1) part[jj] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_id == 0) {
        const float o = (acc + part[jj]) / den;
        const ulong oi = ((ulong)t * IKM_H + h) * IKM_D + jj;
        O[oi] = o * ikm_sigmoid(gate[oi]);
    }
}


/* Decode attention over a long context (one token, global slot): the keys are split over nsplit threadgroups per head so the GPU is
 * filled (the plain kernel above uses 24 threadgroups, each walking every key).  Each split runs the same online softmax over its
 * key range and writes (m, den, acc[128]); split 0 also carries the virtual key (logit 0, value 0).  kernel_ik_attn_reduce merges
 * the splits and applies the output gate. */
struct ikm_attn_split_args { uint n_tokens, p0, rows, local, nsplit, chunk, pad0, pad1; };
kernel void kernel_ik_attn_split(constant ikm_attn_split_args &a [[buffer(0)]],
                                 device const float *Q     [[buffer(1)]],
                                 device const float *R     [[buffer(2)]],
                                 device const float *sink  [[buffer(3)]],
                                 device const float *relb  [[buffer(4)]],
                                 device const half *K      [[buffer(5)]],
                                 device const half *V      [[buffer(6)]],
                                 device float *part        [[buffer(7)]],
                                 uint3 tg3  [[threadgroup_position_in_grid]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint  sgi  [[simdgroup_index_in_threadgroup]],
                                 uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float q_s[IKM_D], r_s[IKM_R], sc[256], red[8], accp[IKM_D];
    const uint tid = tid3.x;
    const uint h = tg3.x, split = tg3.y, g = h / (IKM_H / IKM_HK);
    const uint p = a.p0;
    const uint first = (a.local && p >= IKM_EXTENT) ? p - (IKM_EXTENT - 1) : 0u;
    const uint kstart = first + split * a.chunk;
    const uint kend = min(p + 1u, kstart + a.chunk);   /* exclusive */
    if (tid < IKM_D) q_s[tid] = Q[(ulong)h * IKM_D + tid];
    if (tid < IKM_R) r_s[tid] = R[(ulong)h * IKM_R + tid];
    const float sinkv = sink[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = split == 0 ? 0.0f : -1e30f, den = split == 0 ? 1.0f : 0.0f, acc = 0.0f;
    const uint jj = tid & 127u, half_id = tid >> 7;
    for (uint kb = kstart; kb < kend; kb += 256) {
        const uint k = kb + tid;
        float s = -INFINITY;
        if (k < kend) {
            const uint d = p - k;
            float b = 0.0f;
            if (d < IKM_EXTENT) for (uint j = 0; j < IKM_R; j++) b = fma(r_s[j], relb[d * IKM_R + j], b);
            device const half4 *kr = (device const half4 *)(K + (ulong)(k % a.rows) * IKM_HK * IKM_D + g * IKM_D);
            float dt = 0.0f;
            for (uint i = 0; i < IKM_D / 4; i++) {
                const float4 kv = float4(kr[i]);
                dt = fma(q_s[4*i], kv.x, dt); dt = fma(q_s[4*i+1], kv.y, dt);
                dt = fma(q_s[4*i+2], kv.z, dt); dt = fma(q_s[4*i+3], kv.w, dt);
            }
            s = dt / 11.313708499f + b - sinkv;
        }
        const float bm = simd_max(s);
        if (lane == 0) red[sgi] = bm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bmax = red[0];
        for (uint i = 1; i < 8; i++) bmax = max(bmax, red[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float mnew = max(m, bmax);
        const float e = (k < kend) ? precise::exp(s - mnew) : 0.0f;
        sc[tid] = e;
        const float es = simd_sum(e);
        if (lane == 0) red[sgi] = es;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8; i++) tot += red[i];
        const float rescale = precise::exp(m - mnew);
        den = den * rescale + tot;
        acc *= rescale;
        m = mnew;
        const uint nk = min(256u, kend - kb);
        #pragma clang loop unroll_count(16)
        for (uint kk = half_id; kk < nk; kk += 2)
            acc = fma(sc[kk], (float)V[(ulong)((kb + kk) % a.rows) * IKM_HK * IKM_D + g * IKM_D + jj], acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (half_id == 1) accp[jj] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_id == 0) {
        device float *o = part + ((ulong)h * a.nsplit + split) * (IKM_D + 2);
        o[2 + jj] = acc + accp[jj];
        if (jj == 0) { o[0] = m; o[1] = den; }
    }
}

kernel void kernel_ik_attn_reduce(constant ikm_attn_split_args &a [[buffer(0)]],
                                  device const float *part [[buffer(1)]],
                                  device const float *gate [[buffer(2)]],
                                  device float *O          [[buffer(3)]],
                                  uint3 tg3  [[threadgroup_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint h = tg3.x, jj = tid3.x;
    device const float *base = part + (ulong)h * a.nsplit * (IKM_D + 2);
    float M = -1e30f;
    for (uint i = 0; i < a.nsplit; i++) M = max(M, base[(ulong)i * (IKM_D + 2)]);
    float den = 0.0f, acc = 0.0f;
    for (uint i = 0; i < a.nsplit; i++) {
        device const float *pi = base + (ulong)i * (IKM_D + 2);
        const float w = precise::exp(pi[0] - M);
        den = fma(pi[1], w, den);
        acc = fma(pi[2 + jj], w, acc);
    }
    const ulong oi = (ulong)h * IKM_D + jj;
    O[oi] = acc / den * ikm_sigmoid(gate[oi]);
}


/* Prefill attention on simdgroup matrices (flash style): one threadgroup = one KV head x 8 consecutive tokens, one simdgroup per
 * query head of the group (three), 8 query rows each.  Q and the softmax weights are rounded to half for the matrix products (K and V
 * are half already), everything else is f32.  The ring buffer must hold a multiple of 8 rows so an 8-key tile never wraps. */
kernel void kernel_ik_attn_mma(constant ikm_attn_args &a [[buffer(0)]],
                               device const float *Q     [[buffer(1)]],
                               device const float *R     [[buffer(2)]],
                               device const float *sink  [[buffer(3)]],
                               device const float *gate  [[buffer(4)]],
                               device const float *relb  [[buffer(5)]],
                               device const half *K      [[buffer(6)]],
                               device const half *V      [[buffer(7)]],
                               device float *O           [[buffer(8)]],
                               uint2 tg2 [[threadgroup_position_in_grid]],
                               uint  sg  [[simdgroup_index_in_threadgroup]],
                               uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup half qh[3][8 * 128];
    threadgroup float sS[3][8 * 32];
    threadgroup half sP[3][8 * 32];
    threadgroup float sD[3][64];
    threadgroup float sO[3][8 * 128];
    const uint g = tg2.x, t0 = tg2.y * 8u;
    if (t0 >= a.n_tokens || sg >= 3u) return;
    const uint h = g * 3u + sg;
    const uint r = lane >> 2, q4 = lane & 3u;
    const uint t = t0 + r;
    const bool rv = t < a.n_tokens;
    const uint p_r = a.p0 + (rv ? t : a.n_tokens - 1u);
    const uint first_r = (a.local && p_r >= IKM_EXTENT) ? p_r - (IKM_EXTENT - 1u) : 0u;
    for (uint i = 0; i < 8; i++) {
        float4 v = rv ? ((device const float4 *)(Q + ((ulong)t * IKM_H + h) * IKM_D))[q4 * 8u + i] : float4(0.0f);
        ((threadgroup half4 *)(qh[sg] + r * 128u + q4 * 32u))[i] = half4(v);
    }
    float Rr[16];
    for (uint j = 0; j < 16; j++) Rr[j] = rv ? R[((ulong)t * IKM_H + h) * IKM_R + j] : 0.0f;
    const float sinkv = rv ? sink[(ulong)t * IKM_H + h] : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Oa[16];
    for (uint j = 0; j < 16; j++) Oa[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    float m = 0.0f, den = 1.0f;
    const uint p_first = a.p0 + t0, p_last = a.p0 + min(t0 + 7u, a.n_tokens - 1u);
    const uint first0 = (a.local && p_first >= IKM_EXTENT) ? p_first - (IKM_EXTENT - 1u) : 0u;
    for (uint kb = first0 & ~7u; kb <= p_last; kb += 32u) {
        for (uint kt = 0; kt < 4; kt++) {
            const uint key0 = kb + kt * 8u;
            simdgroup_float8x8 sm = make_filled_simdgroup_matrix<float, 8>(0.0f);
            if (key0 <= p_last) {
                device const half *kp = K + (ulong)(key0 % a.rows) * (IKM_HK * IKM_D) + g * IKM_D;
                for (uint c = 0; c < 16; c++) {
                    simdgroup_half8x8 qa, kk;
                    simdgroup_load(qa, qh[sg] + c * 8u, 128);
                    simdgroup_load(kk, kp + c * 8u, IKM_HK * IKM_D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sm, qa, kk, sm);
                }
            }
            simdgroup_store(sm, sS[sg] + kt * 8u, 32);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sv[8];
        float mb = -INFINITY;
        for (uint i = 0; i < 8; i++) {
            const uint k = kb + q4 * 8u + i;
            float val = -INFINITY;
            if (rv && k <= p_r && k >= first_r) {
                const uint d = p_r - k;
                float b = 0.0f;
                if (d < IKM_EXTENT) for (uint j = 0; j < IKM_R; j++) b = fma(Rr[j], relb[d * IKM_R + j], b);
                val = sS[sg][r * 32u + q4 * 8u + i] / 11.313708499f + b - sinkv;
            }
            sv[i] = val;
            mb = max(mb, val);
        }
        mb = max(mb, simd_shuffle_xor(mb, 1u)); mb = max(mb, simd_shuffle_xor(mb, 2u));
        const float mnew = max(m, mb);
        float ps = 0.0f;
        for (uint i = 0; i < 8; i++) {
            const half pv = half(sv[i] == -INFINITY ? 0.0f : precise::exp(sv[i] - mnew));
            sP[sg][r * 32u + q4 * 8u + i] = pv;
            ps += (float)pv;
        }
        ps += simd_shuffle_xor(ps, 1u); ps += simd_shuffle_xor(ps, 2u);
        const float rescale = precise::exp(m - mnew);
        den = den * rescale + ps;
        m = mnew;
        for (uint e = lane; e < 64u; e += 32u) sD[sg][e] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (q4 == 0u) sD[sg][r * 8u + r] = rescale;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 Dm;
        simdgroup_load(Dm, sD[sg], 8);
        for (uint j = 0; j < 16; j++) { simdgroup_float8x8 tmp; simdgroup_multiply(tmp, Dm, Oa[j]); Oa[j] = tmp; }
        for (uint kt = 0; kt < 4; kt++) {
            const uint key0 = kb + kt * 8u;
            if (key0 > p_last) break;
            simdgroup_half8x8 pt;
            simdgroup_load(pt, sP[sg] + kt * 8u, 32);
            device const half *vp = V + (ulong)(key0 % a.rows) * (IKM_HK * IKM_D) + g * IKM_D;
            for (uint j = 0; j < 16; j++) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, vp + j * 8u, IKM_HK * IKM_D);
                simdgroup_multiply_accumulate(Oa[j], pt, vb, Oa[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 16; j++) simdgroup_store(Oa[j], sO[sg] + j * 8u, 128);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (rv) {
        const float inv = 1.0f / den;
        for (uint i = 0; i < 32; i++) {
            const uint dim = q4 * 32u + i;
            const ulong oi = ((ulong)t * IKM_H + h) * IKM_D + dim;
            O[oi] = sO[sg][r * 128u + dim] * inv * ikm_sigmoid(gate[oi]);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Embedding, adds, output                                                   */
/* ------------------------------------------------------------------------- */

static inline float ikm_elem(uint type, device const uchar *row, uint i) {
    switch (type) {
    case 0u:  return ((device const float *)row)[i];
    case 1u:  return (float)((device const half *)row)[i];
    case 30u: return as_type<float>((uint)((device const ushort *)row)[i] << 16);
    case 8u: {
        device const uchar *b = row + (i >> 5) * 34u;
        return (float)(*(device const half *)b) * (float)((device const char *)b)[2 + (i & 31u)];
    }
    default: return 0.0f;
    }
}

/* x[t] = rmsnorm(embedding row of token t) * embd_norm.  type: 0 f32, 1 f16, 8 q8_0, 30 bf16. */
struct ikm_embed_args { uint n_tokens, type, row_bytes, pad; };
kernel void kernel_ik_embed(constant ikm_embed_args &a [[buffer(0)]],
                            device const uchar *emb  [[buffer(1)]],
                            device const int *tokens [[buffer(2)]],
                            device const float *nw   [[buffer(3)]],
                            device float *x          [[buffer(4)]],
                            uint t    [[threadgroup_position_in_grid]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint sgi  [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (t >= a.n_tokens) return;
    device const uchar *row = emb + (ulong)tokens[t] * a.row_bytes;
    float ss = 0.0f;
    for (uint i = tid; i < IKM_E; i += 256) { const float v = ikm_elem(a.type, row, i); ss = fma(v, v, ss); }
    ss = simd_sum(ss);
    if (lane == 0) red[sgi] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8; g++) tot += red[g];
    const float inv = 1.0f / precise::sqrt(tot / (float)IKM_E + IKM_EPS);
    for (uint i = tid; i < IKM_E; i += 256) x[(ulong)t * IKM_E + i] = ikm_elem(a.type, row, i) * inv * nw[i];
}

kernel void kernel_ik_add(constant uint &count [[buffer(0)]],
                          device const float *a  [[buffer(1)]],
                          device const float *b  [[buffer(2)]],
                          device float *out      [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < count) out[i] = a[i] + b[i];
}

kernel void kernel_ik_softcap(constant uint &count [[buffer(0)]],
                              device float *x      [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < count) x[i] = 30.0f * precise::tanh(x[i] / 30.0f);
}
