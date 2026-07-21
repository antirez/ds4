struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;
template [[host_name("kernel_cpy_f16_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<half, half>;

// Contiguous 1D conversions avoid the generic tensor-index reconstruction
// above. Packed vector types retain scalar alignment, so tensor views whose
// offsets are float/half aligned do not need additional 16/8-byte alignment.
// The final vector is converted element-by-element when n is not divisible by
// four, preserving the generic kernel's bounds and conversion semantics.
kernel void kernel_cpy_contig_f32_f16_4(
        constant uint & n,
        device const packed_float4 * src,
        device       packed_half4  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const float4 value = float4(src[gid]);
        dst[gid] = packed_half4(half4(value));
        return;
    }

    device const float * src_scalar = (device const float *)src;
    device       half  * dst_scalar = (device       half  *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = half(src_scalar[i + lane]);
    }
}

kernel void kernel_cpy_contig_f16_f32_4(
        constant uint & n,
        device const packed_half4  * src,
        device       packed_float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        const half4 value = half4(src[gid]);
        dst[gid] = packed_float4(float4(value));
        return;
    }

    device const half  * src_scalar = (device const half  *)src;
    device       float * dst_scalar = (device       float *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = float(src_scalar[i + lane]);
    }
}

// Bitwise F16 transport for cache staging. Use ushort rather than half so NaN
// payloads and every other binary16 encoding pass through unchanged.
kernel void kernel_cpy_contig_f16_f16_bits_4(
        constant uint & n,
        device const packed_ushort4 * src,
        device       packed_ushort4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) {
        return;
    }

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = src[gid];
        return;
    }

    device const ushort * src_scalar = (device const ushort *)src;
    device       ushort * dst_scalar = (device       ushort *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = src_scalar[i + lane];
    }
}

struct ds4_metal_args_flash_kv_stage_f16 {
    uint raw_cap;
    uint raw_start;
    uint n_raw;
    uint n_comp;
    uint pad_rows;
    uint shared_pad;
};

// Decode-time gathered attention consumes a logical raw-cache ring followed
// by an already-F16 compressed cache. Pack both regions into the contiguous
// F16 FlashAttention scratch in one dispatch. The raw conversion expression
// and compressed ushort4 transport exactly match the standalone copy kernels.
kernel void kernel_dsv4_flash_kv_stage_f16(
        constant ds4_metal_args_flash_kv_stage_f16 & args,
        device const char * raw_src,
        device const char * comp_src,
        device       char * dst,
        device const char * mask_src,
        device       char * pad_dst,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint row_vecs = 128;
    const uint raw_vecs = args.n_raw * row_vecs;
    const uint n_keys = args.n_raw + args.n_comp;
    const uint total_vecs = n_keys * row_vecs;

    if (gid < raw_vecs) {
        const uint logical_row = gid >> 7;
        const uint col = gid & 127u;
        uint physical_row = args.raw_start + logical_row;
        if (physical_row >= args.raw_cap) {
            physical_row -= args.raw_cap;
        }
        device const packed_float4 *raw =
            (device const packed_float4 *)raw_src;
        device packed_half4 *dst_half = (device packed_half4 *)dst;
        const float4 value =
            float4(raw[physical_row * row_vecs + col]);
        dst_half[gid] = packed_half4(half4(value));
        return;
    }

    if (gid < total_vecs) {
        device const packed_ushort4 *comp =
            (device const packed_ushort4 *)comp_src;
        device packed_ushort4 *dst_bits = (device packed_ushort4 *)dst;
        dst_bits[gid] = comp[gid - raw_vecs];
        return;
    }

    // The vector FlashAttention kernel redirects its final partial block to
    // a compact K/V/mask buffer. When requested, append those writes to this
    // dispatch so gathered decode does not need a standalone pad dispatch.
    const uint pad_rows = args.pad_rows;
    const uint pad_vecs = pad_rows * row_vecs;
    uint pad_gid = gid - total_vecs;
    if (pad_gid < pad_vecs) {
        const uint row = pad_gid >> 7;
        const uint col = pad_gid & 127u;
        const uint valid_rows = n_keys % pad_rows;
        device packed_half4 *pad_half = (device packed_half4 *)pad_dst;
        device packed_ushort4 *pad_bits = (device packed_ushort4 *)pad_dst;
        if (row >= valid_rows) {
            const packed_half4 zero = packed_half4(half4(0.0h));
            pad_half[pad_gid] = zero;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = zero;
            }
            return;
        }

        const uint logical_row = n_keys - valid_rows + row;
        if (logical_row < args.n_raw) {
            uint physical_row = args.raw_start + logical_row;
            if (physical_row >= args.raw_cap) {
                physical_row -= args.raw_cap;
            }
            device const packed_float4 *raw =
                (device const packed_float4 *)raw_src;
            const float4 value =
                float4(raw[physical_row * row_vecs + col]);
            const packed_half4 value_half = packed_half4(half4(value));
            pad_half[pad_gid] = value_half;
            if (!args.shared_pad) {
                pad_half[pad_vecs + pad_gid] = value_half;
            }
        } else {
            device const packed_ushort4 *comp =
                (device const packed_ushort4 *)comp_src;
            const packed_ushort4 value_bits =
                comp[(logical_row - args.n_raw) * row_vecs + col];
            pad_bits[pad_gid] = value_bits;
            if (!args.shared_pad) {
                pad_bits[pad_vecs + pad_gid] = value_bits;
            }
        }
        return;
    }

    pad_gid -= pad_vecs;
    if (pad_gid < pad_rows) {
        const uint valid_rows = n_keys % pad_rows;
        device const ushort *mask_bits = (device const ushort *)mask_src;
        device ushort *pad_mask_bits =
            (device ushort *)pad_dst + 2u * pad_vecs * 4u;
        pad_mask_bits[pad_gid] = pad_gid < valid_rows
            ? mask_bits[n_keys - valid_rows + pad_gid]
            : 0xfbffu;
    }
}
