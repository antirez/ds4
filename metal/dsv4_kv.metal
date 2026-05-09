#include <metal_stdlib>
using namespace metal;

/*
    Lookup table for E4M3FN exponent scaling.
    Used during quantization/dequantization to reconstruct FP8 values.
*/
constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

/*
    KV-cache FP8 quantization kernel arguments.
*/
struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00, ne01, ne02, ne03;
    ulong nb00, nb01, nb02, nb03;
    ulong nb0, nb1, nb2, nb3;

    int n_rot; // RoPE region size (excluded from quantization)
};

/*
    KV final store arguments after RoPE + FP8 round-trip.
*/
struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

/*
    Utility: decode FP8 index into float approximation.
    NOTE: This is NOT true hardware FP8, but emulation for consistency.
*/
static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;

    return (exp == 0)
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

/*
    Dequantize by searching nearest representable FP8 value.
    Uses binary search + tie-breaking rule for deterministic rounding.
*/
static inline float dsv4_e4m3fn_dequant(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;

    // Binary search for closest FP8 bucket
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }

    int best = lo;

    // Refine edge case (choose closer or deterministic tie-break)
    if (best < 126) {
        const float d0 = abs(ax - dsv4_e4m3fn_value(best));
        const float d1 = abs(ax - dsv4_e4m3fn_value(best + 1));

        if (d1 < d0 || (d1 == d0 && ((best & 1) != 0))) {
            best++;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

/*
    FP8 KV-cache quantization kernel.

    Flow:
    1. Split KV row into non-RoPE + RoPE parts
    2. Compute max abs per block (threadgroup reduction)
    3. Compute scaling factor
    4. Quantize + dequantize back to float (simulation of FP8 round-trip)
*/
kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if (row >= n_rows) return;

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base =
        src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03;

    device char * dst_base =
        dst + i1 * args.nb1 + i2 * args.nb2 + i3 * args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {

        float v = 0.0f;

        // Load + compute local abs
        if (tid < 64) {
            v = *((device const float *)(src_base + (off + tid) * args.nb00));
            scratch[tid] = abs(v);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel max reduction
        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);

        // Power-of-two scaling for FP8 stability
        const float scale = exp2(ceil(log2(amax / 448.0f)));

        if (tid < 64) {
            const float q =
                dsv4_e4m3fn_dequant(
                    clamp(v / scale, -448.0f, 448.0f)
                ) * scale;

            *((device float *)(dst_base + (off + tid) * args.nb0)) = q;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Copy RoPE tail (no quantization)
    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *)(dst_base + i * args.nb0)) =
            *((device const float *)(src_base + i * args.nb00));
    }
}

/*
    KV store finalizer after RoPE.

    Responsibilities:
    - Apply FP8 round-trip on non-RoPE region
    - Store half-precision cached version for FlashAttention path
*/
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device float * kv,
        device float * raw_cache,
        threadgroup float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {

    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;

    if (head_dim <= 0 || n_nope <= 0 || tid >= 64) return;

    device float * raw =
        raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {

        float v = 0.0f;

        if (off + tid < n_nope) {
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
        const float scale = exp2(ceil(log2(amax / 448.0f)));

        if (off + tid < n_nope) {
            const float q =
                dsv4_e4m3fn_dequant(
                    clamp(v / scale, -448.0f, 448.0f)
                ) * scale;

            kv[off + tid] = q;

            // Store FP16 compressed view for attention backend
            raw[off + tid] = (float)((half)q);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Copy tail region directly
    for (int i = n_nope + tid; i < head_dim; i += 64) {
        raw[i] = (float)((half)kv[i]);
    }
}

/*
    Ratio-4 temporal shift kernel.

    Moves second half of buffer into first half for recurrent KV compression.
*/
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

/*
    One-step KV compressor.

    Writes:
    - KV state
    - attention score with APE bias
*/
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char * ape,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {

    if (gid >= args.width || args.width == 0 || args.ratio == 0) return;

    const uint pos_mod = args.pos % args.ratio;

    const uint dst_row =
        (args.ratio == 4u) ? (args.ratio + pos_mod) : pos_mod;

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
