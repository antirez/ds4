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

struct ds4_metal_args_dsv4_compressor_pack_ratio4 {
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t n_threads;
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

static inline float dsv4_e4m3fn_dequant(float x) {
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

// Builds the two ratio-4 softmax-pool packs together. Each output plane holds
// four previous-half rows followed by four current-half rows. Normal prefill
// seeds plane zero with 0/-inf; replay seeds it from the previous compressor
// state. Later planes take their previous half from the prior input group.
kernel void kernel_dsv4_compressor_pack_ratio4(
        constant ds4_metal_args_dsv4_compressor_pack_ratio4 & args,
        device const uint * kv,
        device const uint * score,
        device const uint * state_kv,
        device const uint * state_score,
        device       uint * packed_kv,
        device       uint * packed_score,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (group.x >= args.n_comp || group.y >= 8u ||
        args.head_dim == 0u || args.n_threads == 0u) {
        return;
    }

    const uint plane = group.x;
    const uint row = group.y;
    const uint64_t input_row_stride = 2ull * args.head_dim;
    const uint64_t dst_row = ((uint64_t)plane * 8u + row) * args.head_dim;

    for (uint col = tid; col < args.head_dim; col += args.n_threads) {
        const uint64_t dst = dst_row + col;
        if (row >= 4u) {
            const uint token = plane * 4u + (row - 4u);
            const uint64_t src = (uint64_t)token * input_row_stride +
                                 args.head_dim + col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (plane != 0u) {
            const uint token = (plane - 1u) * 4u + row;
            const uint64_t src = (uint64_t)token * input_row_stride + col;
            packed_kv[dst] = kv[src];
            packed_score[dst] = score[src];
        } else if (args.replay != 0u) {
            const uint64_t src = (uint64_t)row * input_row_stride + col;
            packed_kv[dst] = state_kv[src];
            packed_score[dst] = state_score[src];
        } else {
            packed_kv[dst] = 0u;
            packed_score[dst] = 0xff800000u;
        }
    }
}

// Decode already holds the complete ratio-4 recurrent window in state layout:
// eight rows of two head_dim planes. Pack the previous plane from rows 0..3
// and the current plane from rows 4..7 directly into the transposed [head_dim,
// 8] layout consumed by the exact GGML softmax/multiply/sum sequence. KV and
// score move together; no arithmetic or reduction order changes.
kernel void kernel_dsv4_compressor_pack_ratio4_decode_ggml(
        constant ds4_metal_args_dsv4_compressor_pack_ratio4 & args,
        device const uint * state_kv,
        device const uint * state_score,
        device       uint * packed_kv,
        device       uint * packed_score,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (row >= 8u || args.head_dim == 0u || args.n_threads == 0u) {
        return;
    }

    const uint64_t state_row_stride = 2ull * args.head_dim;
    const uint64_t src_plane = row >= 4u ? args.head_dim : 0u;
    for (uint col = tid; col < args.head_dim; col += args.n_threads) {
        const uint64_t src = (uint64_t)row * state_row_stride +
                             src_plane + col;
        const uint64_t dst = (uint64_t)col * 8u + row;
        packed_kv[dst] = state_kv[src];
        packed_score[dst] = state_score[src];
    }
}

// Exact decode specialization for the first two operations in GGML's
// softmax -> multiply -> sum_rows compressor reduction. The normalized
// softmax values are deliberately materialized in device memory and reloaded
// after a device barrier before the in-place product, preserving the dispatch
// boundary's float store/load semantics. The final sum remains the standalone
// eight-thread sum_rows kernel: changing a 32-thread group into the original
// eight-thread reduction inside this kernel would make its threadgroup
// barriers non-uniform or alter simd_sum's active-lane topology.
kernel void kernel_dsv4_compressor_exact_softmax_product_ratio4(
        constant ds4_metal_args_dsv4_compressor_pack_ratio4 & args,
        device const float * packed_kv,
        device const float * packed_score,
        device       float * softmax,
        device       float * product,
        threadgroup  float * softmax_scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.head_dim || args.n_comp != 1u ||
        args.n_threads != 32u) {
        return;
    }

    device const float4 * score4 =
        (device const float4 *)(packed_score + (uint64_t)row * 8u);
    device float4 * softmax4 =
        (device float4 *)(softmax + (uint64_t)row * 8u);
    const float scale = (float)args.replay;
    const float zero = (float)(args.n_comp - 1u);

    // Match kernel_soft_max_f32_4(width=8, nth=32) literally. Only lanes zero
    // and one own float4s, while all 32 lanes participate in both reductions.
    float4 lmax4 = -INFINITY;
    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        lmax4 = fmax(lmax4, score4[i00] * scale + (float4)zero);
    }

    const float lmax =
        MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));
    const float max_val = simd_max(lmax);

    float4 lsum4 = 0.0f;
    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        const float4 exp_score4 =
            exp((score4[i00] * scale + (float4)zero) - max_val);
        lsum4 += exp_score4;
        softmax4[i00] = exp_score4;
    }

    const float lsum =
        lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
    threadgroup_barrier(mem_flags::mem_none);
    const float sum = simd_sum(lsum);
    const float inv_sum = 1.0f / sum;

    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        softmax4[i00] *= inv_sum;
    }

    // Force the same normalized-softmax device store/reload boundary that the
    // separate multiply dispatch observes.
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float * reloaded_softmax =
        (device volatile const float *)(softmax + (uint64_t)row * 8u);
    device const float * kv_row = packed_kv + (uint64_t)row * 8u;
    device float * product_row = product + (uint64_t)row * 8u;

    // Match kernel_bin_fuse_f32_f32_f32(width=8, nth=4): four lanes each
    // process their low element followed by the element four positions later.
    if (tid < 4u) {
        for (uint i0 = tid; i0 < 8u; i0 += 4u) {
            float value = kv_row[i0];
            value *= reloaded_softmax[i0];
            product_row[i0] = value;
        }
    }

    // All 32 lanes reach the final device barrier. The following standalone
    // sum_rows dispatch performs the required global reload and exact TG8
    // two-stage simd_sum topology.
    threadgroup_barrier(mem_flags::mem_device);
    (void)softmax_scratch;
}

// Exact one-dispatch ratio-4 decode pool. This specializes the three-dispatch
// pack -> exact softmax/product -> sum_rows chain above without changing any
// floating-point operation or reduction topology. The normalized softmax and
// product are still materialized and volatile-reloaded through device memory.
// The two simd_sum calls in the final reduction execute under an eight-lane
// active mask, exactly matching kernel_sum_rows_f32_f32's original TG8.
kernel void kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml(
        constant ds4_metal_args_dsv4_compressor_pack_ratio4 & args,
        device const float * state_kv,
        device const float * state_score,
        device       float * softmax,
        device       float * product,
        device       float * dst,
        threadgroup  float * sum_scratch [[threadgroup(0)]],
        uint col [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (col >= args.head_dim || args.n_comp != 1u ||
        args.n_threads != 32u) {
        return;
    }

    const uint64_t state_row_stride = 2ull * args.head_dim;
    const float scale = (float)args.replay;
    const float zero = (float)(args.n_comp - 1u);

    // Match the packed float4 ownership: lane 0 owns rows 0..3 and lane 1
    // rows 4..7. The gather itself is an integer-addressed bit-preserving load.
    float4 score_values = -INFINITY;
    if (tid < 2u) {
        const uint row0 = 4u * tid;
        for (uint j = 0u; j < 4u; ++j) {
            const uint row = row0 + j;
            const uint64_t src = (uint64_t)row * state_row_stride +
                                 (row >= 4u ? args.head_dim : 0u) + col;
            score_values[j] = state_score[src];
        }
    }

    const uint64_t scratch_base = (uint64_t)col * 8u;
    device float4 * softmax4 =
        (device float4 *)(softmax + scratch_base);

    // Verbatim kernel_soft_max_f32_4(width=8, nth=32) arithmetic.
    float4 lmax4 = -INFINITY;
    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        lmax4 = fmax(lmax4, score_values * scale + (float4)zero);
    }
    const float lmax =
        MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));
    const float max_val = simd_max(lmax);

    float4 lsum4 = 0.0f;
    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        const float4 exp_score4 =
            exp((score_values * scale + (float4)zero) - max_val);
        lsum4 += exp_score4;
        softmax4[i00] = exp_score4;
    }
    const float lsum =
        lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
    threadgroup_barrier(mem_flags::mem_none);
    const float sum = simd_sum(lsum);
    const float inv_sum = 1.0f / sum;
    for (int i00 = (int)tid; i00 < 2; i00 += 32) {
        softmax4[i00] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float * reloaded_softmax =
        (device volatile const float *)(softmax + scratch_base);
    device float * product_row = product + scratch_base;

    // Verbatim width=8, TG4 multiply ownership: low element, then +4.
    if (tid < 4u) {
        for (uint i0 = tid; i0 < 8u; i0 += 4u) {
            const uint64_t src = (uint64_t)i0 * state_row_stride +
                                 (i0 >= 4u ? args.head_dim : 0u) + col;
            float value = state_kv[src];
            value *= reloaded_softmax[i0];
            product_row[i0] = value;
        }
    }

    // Preserve the product dispatch's device store/reload boundary.
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float * reloaded_product =
        (device volatile const float *)product_row;

    // Reproduce kernel_sum_rows_f32_f32(width=8, TG8) literally. MSL defines
    // simdgroup collectives over active lanes, so the branch recreates the
    // original eight-lane partial SIMD group inside this 32-thread group.
    sum_scratch[tid] = 0.0f;
    float row_sum = 0.0f;
    if (tid < 8u) {
        row_sum += reloaded_product[tid];
        row_sum = simd_sum(row_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        sum_scratch[0] = row_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        row_sum = sum_scratch[tid];
        row_sum = simd_sum(row_sum);
        if (tid == 0u) {
            dst[col] = row_sum;
        }
    }
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
