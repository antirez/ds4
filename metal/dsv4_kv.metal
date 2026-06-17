constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

constant float dsv4_e2m1fn_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    int n_rot;
};

struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

struct ds4_metal_args_dsv4_indexer_qat {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
};

struct ds4_metal_args_dsv4_ratio4_shift {
    uint32_t width;
};

struct ds4_metal_args_dsv4_compressor_store_one {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

// Round a non-negative magnitude to the nearest e4m3fn code (0..126), ties to
// even code. Shared by the value-returning dequant and the byte-packing path so
// the packed e4m3 byte unpacks to exactly the value the float path would store.
static inline int dsv4_e4m3fn_code(float ax) {
    ax = min(ax, 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }
    return best;
}

// Software e4m3fn round-trip: round |x| to the nearest representable e4m3fn
// magnitude (ties to even) and reapply the sign. Monotonic and correct across
// the whole range; used directly when the native packer is unavailable and as
// the subnormal fallback for the hybrid path below.
static inline float dsv4_e4m3fn_dequant_sw(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

#if defined(DS4_METAL_FP8_NATIVE) && __METAL_VERSION__ >= 410
// Hybrid e4m3fn round-trip using the MSL 4.1 hardware packer. On macOS 27 beta
// (build 26A5353q) the metal_fp8_e4m3_format packer is bit-exact with the
// software ladder on the normal range [2^-6, 448], but it rounds subnormals
// non-monotonically and returns NaN for magnitudes above the max normal instead
// of saturating. So below the min normal we keep the software ladder, and above
// it we use the native pack/unpack with a defensive clamp that also guards the
// >448 NaN. Verified bit-exact against dsv4_e4m3fn_dequant_sw over 4M samples
// spanning [-448, 448] (see ds4_gpu_validate_fp8_native). Once Apple fixes the
// subnormal/saturation behavior this can collapse to a bare pack/unpack.
static inline float dsv4_e4m3fn_dequant(float x) {
    if (abs(x) < 0.015625f) {
        return dsv4_e4m3fn_dequant_sw(x);
    }
    const vec<float, 4> v(clamp(x, -448.0f, 448.0f), 0.0f, 0.0f, 0.0f);
    return unpack<float, metal_fp8_e4m3_format, 4>(pack<metal_fp8_e4m3_format>(v))[0];
}
#else
static inline float dsv4_e4m3fn_dequant(float x) {
    return dsv4_e4m3fn_dequant_sw(x);
}
#endif

static inline float dsv4_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 6.0f);
    int best = 0;
    float best_diff = abs(ax - dsv4_e2m1fn_values[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = abs(ax - dsv4_e2m1fn_values[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_values[best];
}

// Quantizes the non-RoPE part of a KV row through E4M3FN and writes the
// dequantized value back as float. DS4 uses this to match the FP8 KV-cache
// semantics while keeping the Metal graph's cache buffers float-addressable.
kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_base = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((device const float *) (src_base + (off + tid)*args.nb00));
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
            *((device float *) (dst_base + (off + tid)*args.nb0)) = q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *) (dst_base + i*args.nb0)) = *((device const float *) (src_base + i*args.nb00));
    }
}

// The official DS4 indexer applies a 128-wide Hadamard rotation and then an
// inplace FP4 activation-simulation pass to both indexer Q and indexer KV.
kernel void kernel_dsv4_indexer_hadamard_fp4_f32(
        constant ds4_metal_args_dsv4_indexer_qat & args,
        device   char  * x,
        threadgroup float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows || args.head_dim != 128u || tid >= 128u) {
        return;
    }

    threadgroup float *vals = scratch;
    threadgroup float *absbuf = scratch + 128;
    device float *xr = (device float *)(x + (uint64_t)row * args.row_stride);

    vals[tid] = xr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float v = vals[tid] * 0.08838834764831845f;
    const uint block = tid >> 5u;
    const uint lane = tid & 31u;
    const uint block_base = block * 32u;
    absbuf[tid] = abs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = max(absbuf[block_base + lane],
                                            absbuf[block_base + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float amax = max(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = exp2(ceil(log2(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
}

// Decode-side KV finalizer after RoPE. The normal RoPE kernel intentionally
// remains separate because tiny trigonometric codegen changes can flip later
// sampled tokens. This kernel only fuses the FP8 round-trip for the non-RoPE
// prefix with the F16-rounded raw-cache row used by FlashAttention.
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            // Diagnostic only: skip the FP16 round-trip that normally matches the
            // half-typed FlashAttention KV buffer's precision. With this enabled the
            // indexer will see higher-precision raw values than FlashAttention does,
            // which is informative but not a production-ready setting.
#ifdef DS4_METAL_KV_RAW_F32
            raw[off + tid] = q;
#else
            raw[off + tid] = (float)((half)q);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
#ifdef DS4_METAL_KV_RAW_F32
        raw[i] = kv[i];
#else
        raw[i] = (float)((half)kv[i]);
#endif
    }
}

// Packed FP8 compressed-KV writer. Produces the same e4m3fn-quantized values as
// kernel_dsv4_fp8_kv_quantize_f32 but stores them compactly for FlashAttention:
//   - nope plane : one e4m3 byte per element (sign<<7 | code).
//   - scale plane: one ue8m0 byte per 64-element block (exponent + 127).
//   - rot plane  : the RoPE prefix copied verbatim as float (precision-preserving).
// Reconstruction is value = e4m3_value(code) * 2^(ue8m0-127), which is bit-exact
// with the float path's stored value (verified offline over 4M values). One row
// per threadgroup, 64 threads, mirroring the quantize kernel's block reduction.
kernel void kernel_dsv4_kv_pack_fp8_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const float * src0,
        device        uchar * nope_bytes,
        device        uchar * scale_bytes,
        device        float * rot_out,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows || tid >= 64) {
        return;
    }

    const int head_dim = (int)args.ne00;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    const int n_blk = n_nope / 64;

    device const float * src = src0 + (int64_t)row * head_dim;
    device       uchar * nb  = nope_bytes  + (int64_t)row * n_nope;
    device       uchar * sb  = scale_bytes + (int64_t)row * n_blk;
    device       float * rt  = rot_out     + (int64_t)row * n_rot;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = src[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        int exp = (int)ceil(log2(amax / 448.0f));
        if (exp < -127) exp = -127;
        if (exp > 127) exp = 127;
        const float scale = exp2((float)exp);
        if (tid == 0) {
            sb[off / 64] = (uchar)(exp + 127);
        }
        if (off + (int)tid < n_nope) {
            const float vs = clamp(v / scale, -448.0f, 448.0f);
            const uint sign = vs < 0.0f ? 0x80u : 0x00u;
            const int code = dsv4_e4m3fn_code(abs(vs));
            nb[off + tid] = (uchar)(sign | (uint)code);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tid; i < n_rot; i += 64) {
        rt[i] = src[n_nope + i];
    }
}

// Row-interleaved packed FP8 writer consumed directly by decode FlashAttention's
// dequantize_fp8_kv_t4 (see flash_attn.metal struct ds4_fp8_kv_row). Produces the
// same e4m3-quantized values as the float quantize path but in one buffer with a
// single per-row stride, so the F16 staging copy is eliminated:
//   [scale: n_blk ue8m0 bytes, padded to 8][nope: n_nope e4m3 bytes][rot: n_rot half]
// The rot prefix is rounded to half here to match the prior F16 staging exactly,
// making the reconstructed (half) KV bit-identical to the current attention input.
// One row per threadgroup, 64 threads, mirroring the quantize kernel's reduction.
kernel void kernel_dsv4_kv_pack_fp8_row_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const float * src0,
        device        uchar * rows,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows || tid >= 64) {
        return;
    }

    const int head_dim = (int)args.ne00;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    const int n_blk = n_nope / 64;
    const int scale_pad = (n_blk + 7) & ~7;          // 8 for n_blk=7; keeps rot 8-aligned
    const int rot_off = scale_pad + n_nope;          // byte offset of the rot (half) plane
    const int stride = rot_off + n_rot * (int)sizeof(half);

    device const float * src = src0 + (int64_t)row * head_dim;
    device       uchar * base = rows + (int64_t)row * stride;
    device       uchar * sb   = base;                                  // scale plane
    device       uchar * nb   = base + scale_pad;                      // nope plane
    device       half  * rt   = (device half *)(base + rot_off);       // rot plane

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = src[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride2 = 32; stride2 > 0; stride2 >>= 1) {
            if (tid < stride2) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride2]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        int exp = (int)ceil(log2(amax / 448.0f));
        if (exp < -127) exp = -127;
        if (exp > 127) exp = 127;
        const float scale = exp2((float)exp);
        if (tid == 0) {
            sb[off / 64] = (uchar)(exp + 127);
        }
        if (off + (int)tid < n_nope) {
            const float vs = clamp(v / scale, -448.0f, 448.0f);
            const uint sign = vs < 0.0f ? 0x80u : 0x00u;
            const int code = dsv4_e4m3fn_code(abs(vs));
            nb[off + tid] = (uchar)(sign | (uint)code);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = tid; i < n_rot; i += 64) {
        rt[i] = (half)src[n_nope + i];
    }
}

// Inverse of kernel_dsv4_kv_pack_fp8_row_f32: unpack packed FP8 comp rows back to
// a contiguous f16 buffer for the (cold) flash-attention staging path, which keeps
// using the existing kvpad-correct F16 machinery. The hot indexed-attention path
// reads the packed cache directly and does not use this. One row per threadgroup,
// 128 threads (each emits one half4 via the shared dequantize_fp8_kv_t4 decode).
kernel void kernel_dsv4_kv_unpack_fp8_row_f16(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const uchar * rows,
        device        half  * dst,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows || tid >= 128u) {
        return;
    }
    const int head_dim = (int)args.ne00;
    if (head_dim != 512 || args.n_rot != 64) {
        return; // packed layout (ds4_fp8_kv_row) is fixed to DS4's 512/64 dims
    }
    device const ds4_fp8_kv_row * src = ((device const ds4_fp8_kv_row *) rows) + row;
    device half4 * out = (device half4 *)(dst + (int64_t)row * head_dim);
    half4 v;
    dequantize_fp8_kv_t4(src, (short)tid, v);
    out[tid] = v;
}

// Ratio-4 compression keeps two 4-row halves of recurrent state. After an
// emitted compressed row, the second half becomes the next window's previous
// half. The old encoder expressed this as four generic copies; this DS4-specific
// kernel performs the KV and score copies together.
kernel void kernel_dsv4_ratio4_shift_f32(
        constant ds4_metal_args_dsv4_ratio4_shift & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const uint n = 4u * args.width;
    if (gid >= n) return;

    state_kv[gid] = state_kv[n + gid];
    state_score[gid] = state_score[n + gid];
}

// One-token compressor frontier update. Decode appends exactly one projected KV
// row and one score row into a small recurrent state. The generic batch helper
// expresses this as APE copy, score add, and two set_rows operations; this
// kernel writes both state tensors directly while preserving the same
// score + APE arithmetic.
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char  * ape,
        device       float * state_kv,
        device       float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {
        return;
    }

    const uint pos_mod = args.pos % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * args.width + gid;
    const uint ape_i = pos_mod * args.width + gid;

    float ape_v;
    if (args.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}
