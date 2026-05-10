// DS4 Metal routed-MoE matvec kernels.

#define QK_K 256
#define N_R0_Q2_K 4
#define N_R0_Q4_K 2
#define N_R0_IQ2_XXS 4

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
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

constant bool FC_mul_mm_id_mpp [[function_constant(FC_MUL_MM + 2)]];

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum6_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

kernel void kernel_dsv4_moe_sum6_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        d[col] = v;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

template<int nr0>
void kernel_mul_mv_iq2_xxs_pair_f32_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * xg = (device const block_iq2_xxs *) (src0_gate + offset0);
    device const block_iq2_xxs * xu = (device const block_iq2_xxs *) (src0_up   + offset0);
    device const float         * y  = (device const float         *) (src1      + offset1);

    float yl[32];
    float sumg[nr0]={0.f};
    float sumu[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xgr = xg + ibl;
        device const block_iq2_xxs * xur = xu + ibl;
        device const uint16_t * qg = xgr->qs + 4 * ib;
        device const uint16_t * qu = xur->qs + 4 * ib;
        device const half * dhg = &xgr->d;
        device const half * dhu = &xur->d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t * aux8g = (device const uint8_t *)qg;
            device const uint8_t * aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t * gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7*l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8*l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01/2;
            dhu += args.nb01/2;
            qg  += args.nb01/2;
            qu  += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_gate_f32 = (device float *) dst_gate + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_up_f32   = (device float *) dst_up   + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            dst_gate_f32[first_row + row] = sum_gate * 0.25f;
            dst_up_f32[first_row + row]   = sum_up   * 0.25f;
        }
    }
}

typedef void (kernel_mul_mv2_disp_t)(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>) mul_mv_id_disp_fn_t;

// Decode-time expert matvec. The ids tensor selects the routed expert for each
// slot, then this wrapper invokes the quantized row kernel for Q8_0, Q2_K, or
// IQ2_XXS weights without materializing per-expert dispatches on the CPU.
template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>) kernel_mul_mv_id_q_t;
typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>) kernel_mul_mv_id_q8_0_t;

// Host-visible decode MoE matvec variants for the DS4 quant formats.
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_q8_0_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;

// DS4 attention output low projection, specialized for the fixed block
// diagonal mapping used by the model:
//
//     low[token, group, rank] = heads[token, group, :] * Woa[group, rank, :]
//
// The generic GGML-style id matvec supports arbitrary routed expert ids.  Here
// the id is always equal to the group number, so this wrapper keeps the exact
// Q8_0 dot kernel but removes the id-buffer load and the CPU-side id table.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_id_iq2_xxs_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_gate_cur = src0_gate + i02*args.nb02;
    device const char * src0_up_cur   = src0_up   + i02*args.nb02;
    device const char * src1_cur      = src1      + i11*args.nb11 + i12*args.nb12;

    device char * dst_gate_cur = dst_gate + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    device char * dst_up_cur   = dst_up   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Decode-only routed expert gate/up projection fused with the DS4 activation:
//
//     mid = silu(clamp(gate)) * clamp(up) * route_weight
//
// The quantized dot products are intentionally the same IQ2_XXS paired path as
// `kernel_mul_mv_id_iq2_xxs_pair_f32`.  The only extra work is done by lane 0
// after each exact reduced row has been produced.  This removes the separate
// routed activation dispatch and avoids rereading the gate/up rows before the
// down projection.  The host uses this only for the normal release path where
// diagnostics do not request clamped gate/up intermediates.
kernel void kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + (uint64_t)idx * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q4_K_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Same release-path fusion as the IQ2_XXS kernel above for the Q4_K expert
// variant.  The Q4 pair path reuses the existing exact matvec implementation
// for gate and up, then the same lane that wrote each row derives the routed
// SwiGLU input.  This keeps Q4 behavior aligned with the Q2 optimization while
// preserving the old pair projection arithmetic.
kernel void kernel_mul_mv_id_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + expert*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_id_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + expert * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

#define QK_NL 16

// Builds the compact per-expert work map used by batched MoE matmul. DS4 routes
// each token to a small fixed top-k list, so this turns token-major ids into
// expert-major slices that the tiled matmul can consume.
template<short ne20>
kernel void kernel_mul_mm_id_map0(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

// Host-visible map builders for the routed-expert counts used by DS4 graph
// shapes. Some arities are generic leftovers retained for nearby batch sizes.
template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Batched routed-expert matmul. It reads the expert-major map produced above,
// loads selected expert weights, and writes results back to token-major slots
// so the DS4 FFN can apply SwiGLU, weighting, and the down projection.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
#ifdef DS4_METAL_HAS_TENSOR
    threadgroup float *sc = (threadgroup float *)shmem;
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
#ifdef DS4_METAL_HAS_TENSOR
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NR1, NK));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

#ifdef DS4_METAL_HAS_TENSOR
                if (FC_mul_mm_id_mpp) {
                    const short lx = i%8;
                    const short ly = (tiitg/NL0)%8;

                    *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
                } else
#endif
                {
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
                }
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

#ifdef DS4_METAL_HAS_TENSOR
                if (FC_mul_mm_id_mpp) {
                    const short lx = i%8;
                    const short ly = (tiitg/NL0)%8;

                    *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
                } else
#endif
                {
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
                }
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

#ifdef DS4_METAL_HAS_TENSOR
                if (FC_mul_mm_id_mpp) {
                    *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                } else
#endif
                {
                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

#ifdef DS4_METAL_HAS_TENSOR
            if (FC_mul_mm_id_mpp) {
                *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
            } else
#endif
            {
            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
            }
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef DS4_METAL_HAS_TENSOR
        if (FC_mul_mm_id_mpp) {
            auto sA = tA.slice(0, 0);
            auto sB = tB.slice(0, 0);
            mm.run(sB, sA, cT);
        } else
#endif
        {
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef DS4_METAL_HAS_TENSOR
    if (FC_mul_mm_id_mpp) {
        auto tC = tensor(sc, dextents<int32_t, 2>(NR0, NR1));
        cT.store(tC);
    } else
#endif
    {
    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;
typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_f16_rhs;

// Host-visible batched MoE matmul variants for the DS4 quant formats.
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_attn_out_low_q8_0_mpp(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne21;
    const int G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12/sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (r0 + row < M) {
                const int block_idx = k_pos/32;
                const short il = (k_pos/16)%2;
                device const block_q8_0 *row_ptr =
                    (device const block_q8_0 *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_q8_0(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_group = (device float *)dst + group*M;
    auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}
#endif

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef QK_K
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_IQ2_XXS
