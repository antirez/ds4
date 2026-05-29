// Metal port of TurboQuant+ 3-bit packed KV cache from ds4_cuda.cu.
//
// Mirrors the device-side primitive + pack/dequant kernels from the CUDA
// implementation.  Constants are byte-equivalent to the CUDA __constant__
// arrays DS4_TURBO3_CODEBOOK_D + DS4_TURBO_SIGNS{1,2}_64_D (Lloyd-Max 3-bit
// codebook for N(0,1) + two-sided Rademacher signs for the 64-point WHT).
//
// This file ships the primitive + pack/dequant kernels.  Inline-dequant
// attention kernels live in the surrounding ds4_metal.m as stubs until the
// Metal turbo3 cache path goes live.

#include <metal_stdlib>
using namespace metal;

#ifndef DS4_TURBO3_GROUP_SIZE
#define DS4_TURBO3_GROUP_SIZE 64u
#endif

#define DS4_TURBO3_DATA_BYTES_PER_GROUP 24u
#define DS4_FP8_E4M3_MAX_D              448.0f
#define DS4_TURBO3_MAX_D                2.1520f

constant float DS4_TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};

constant float DS4_TURBO3_BOUNDS[7] = {
    -1.748f, -1.050f, -0.501f, 0.0f, 0.501f, 1.050f, 1.748f
};

constant float DS4_TURBO_SIGNS1_64[64] = {
    +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f,
};

constant float DS4_TURBO_SIGNS2_64[64] = {
    +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, +1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

// 64-element in-place WHT butterfly (self-inverse, used for both forward
// rotation on pack and inverse rotation on dequant).  Operates on a
// thread-local 64-float buffer.
static inline void turbo3_wht64_inplace(thread float buf[64]) {
    for (int stride = 1; stride < 64; stride <<= 1) {
        for (int base = 0; base < 64; base += (stride << 1)) {
            for (int i = 0; i < stride; i++) {
                const float a = buf[base + i];
                const float b = buf[base + i + stride];
                buf[base + i] = a + b;
                buf[base + i + stride] = a - b;
            }
        }
    }
}

// FP8 E4M3 -> float32.  Apple Silicon Metal doesn't expose a hardware FP8
// dequant op, so we use the same lookup-table approach as the existing
// dsv4_kv.metal::dsv4_e4m3fn_value (used by the FP8 KV path).  Inlined here
// to keep this file self-contained.
static inline float turbo3_fp8_e4m3_value(int i) {
    constexpr float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int sign = (i >> 7) & 1;
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    const float m = exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * exp_scale[exp];
    return sign ? -m : m;
}

// float32 -> FP8 E4M3 byte.  Mirror of dsv4_kv.metal::dsv4_e4m3fn_dequant
// (which returns the dequanted float); this returns the encoded byte
// directly.  Pure-scalar binary search across the 127 positive
// representable E4M3 values; ties broken to even-mantissa (round-half-to-
// even), matching CUDA's __nv_cvt_f32_to_fp8(value, __NV_E4M3).
static inline uchar turbo3_fp8_e4m3_encode(float x) {
    const int sign_bit = (x < 0.0f) ? 0x80 : 0;
    const float ax = min(fabs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (turbo3_fp8_e4m3_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabs(ax - turbo3_fp8_e4m3_value(best));
        const float next_diff = fabs(ax - turbo3_fp8_e4m3_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return (uchar)(best | sign_bit);
}

// One-shot per-group dequant.  Mirror of CUDA's
// turbo3_dequant_group64_device.  Reads 24 packed bytes + 1 FP8 scale byte
// from a packed row, writes 64 original-basis floats into `out64`.
//
// Cost per call (per thread): 24 byte loads + 1 FP8 byte + 64 LUT lookups +
// 64 muls + 6-stage 64-element butterfly + signs1 mul ~ 200 fp ops.  Same
// envelope as CUDA.
static inline void turbo3_dequant_group64(
        thread float        out64[64],
        device const uchar *row_base,
        uint                group_idx,
        uint                n_nope,
        int                 signs_on) {
    device const uchar *data_slot = row_base + group_idx * DS4_TURBO3_DATA_BYTES_PER_GROUP;
    const uint  data_bytes  = n_nope * 3u / 8u;
    const uchar scale_byte  = row_base[data_bytes + group_idx];

    // FP8 E4M3 scale byte -> float (LUT, no hardware cvt on Metal).
    const float scale = turbo3_fp8_e4m3_value((int)scale_byte);

    // Pre-scaled centroid cache (per-block scaled-centroid hoist pattern
    // from llama-cpp-turboquant - saves 64 muls per group).
    float sc[8];
    for (int c = 0; c < 8; c++) sc[c] = DS4_TURBO3_CODEBOOK[c] * scale;

    // Unpack 24 bytes -> 64 rotated-basis floats via the pre-scaled LUT.
    for (int chunk = 0; chunk < 8; chunk++) {
        device const uchar *b = data_slot + chunk * 3;
        thread float *o = out64 + chunk * 8;
        const uint b0 = b[0], b1 = b[1], b2 = b[2];
        o[0] = sc[(b0)                  & 0x7];
        o[1] = sc[(b0 >> 3)             & 0x7];
        o[2] = sc[((b0 >> 6) | (b1<<2)) & 0x7];
        o[3] = sc[(b1 >> 1)             & 0x7];
        o[4] = sc[(b1 >> 4)             & 0x7];
        o[5] = sc[((b1 >> 7) | (b2<<1)) & 0x7];
        o[6] = sc[(b2 >> 2)             & 0x7];
        o[7] = sc[(b2 >> 5)             & 0x7];
    }

    // Inverse rotation: signs2 -> WHT -> 1/sqrt(64) -> signs1.
    if (signs_on) {
        for (int i = 0; i < 64; i++) out64[i] *= DS4_TURBO_SIGNS2_64[i];
    }
    turbo3_wht64_inplace(out64);
    const float inv_sqrt_n = rsqrt(64.0f);
    for (int i = 0; i < 64; i++) out64[i] *= inv_sqrt_n;
    if (signs_on) {
        for (int i = 0; i < 64; i++) out64[i] *= DS4_TURBO_SIGNS1_64[i];
    }
}

// Unaligned f32 load helper.  Turbo3 row is 431 B (not 4-aligned), so the
// RoPE tail at byte offset 175 can't be dereferenced as `device float *` on
// CUDA (we use memcpy-byte-loads there).  Metal's device address space
// historically tolerates unaligned loads, but byte-wise reconstruction is
// the portable fallback if hardware traps on a specific combination.
static inline float turbo3_load_unaligned_f32(device const uchar *p) {
    // Byte-wise reconstruction (matches CUDA's memcpy-based fallback).
    // The Metal compiler typically lowers this to a single uint32 load when
    // alignment is statically known; otherwise it falls back to byte loads.
    uint v = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
    return as_type<float>(v);
}

// Pack kernel - sibling of CUDA's turbo3_kv_pack_kernel.  Reads a
// [n_tok, head_dim] float tensor (post-RoPE KV projection output) and writes
// [n_tok * dst_row_bytes] packed bytes.  Grid: (n_tok, 1, 1) x tg(64,1,1).
//
// One thread per group of 64 elements.  Thread 0 also copies the RoPE tail.
kernel void kernel_dsv4_turbo3_kv_pack_f32(
        device const float *src              [[ buffer(0) ]],
        device       uchar *dst              [[ buffer(1) ]],
        constant     uint  &n_tok            [[ buffer(2) ]],
        constant     uint  &head_dim         [[ buffer(3) ]],
        constant     uint  &n_rot            [[ buffer(4) ]],
        constant     ulong &dst_row_bytes    [[ buffer(5) ]],
        constant     int   &signs_on         [[ buffer(6) ]],
        uint                row              [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (row >= n_tok) return;
    const uint n_nope    = head_dim - n_rot;
    const uint n_groups  = n_nope / DS4_TURBO3_GROUP_SIZE;
    device const float *src_row = src + row * head_dim;
    device       uchar *dst_row = dst + row * dst_row_bytes;
    const ulong data_bytes  = (ulong)n_nope * 3u / 8u;
    const float inv_sqrt_n  = rsqrt(64.0f);

    if (tid < n_groups) {
        // Per-group forward rotation in registers.  64 floats per thread.
        float buf[64];
        device const float *gs = src_row + tid * DS4_TURBO3_GROUP_SIZE;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] = gs[i] * DS4_TURBO_SIGNS1_64[i];
        } else {
            for (int i = 0; i < 64; i++) buf[i] = gs[i];
        }
        // WHT butterfly is self-inverse - same body for forward.
        turbo3_wht64_inplace(buf);
        for (int i = 0; i < 64; i++) buf[i] *= inv_sqrt_n;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] *= DS4_TURBO_SIGNS2_64[i];
        }

        // Matched-norm L2 scale (byte-equivalent to CUDA's
        // turbo3_pack_group64_device): scale = sqrt(norm_sq) / sqrt(recon_sq)
        // where recon = nearest centroid of (v * k_inv).  Falls back to
        // amax / codebook_max when reconstruction is near-zero.  Clamped
        // to E4M3 representable range.
        float amax    = 0.0f;
        float norm_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v  = buf[i];
            const float av = fabs(v);
            if (av > amax) amax = av;
            norm_sq += v * v;
        }
        const float k_inv = (amax > 1e-12f) ? (DS4_TURBO3_MAX_D / amax) : 1.0f;

        uchar idx[64];
        float recon_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v = buf[i] * k_inv;
            int code = 0;
            for (int j = 0; j < 7; j++) {
                if (v >= DS4_TURBO3_BOUNDS[j]) code = j + 1;
            }
            idx[i] = (uchar)code;
            const float c = DS4_TURBO3_CODEBOOK[code];
            recon_sq += c * c;
        }
        const float recon_norm = sqrt(recon_sq);
        float scale = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm)
                                            : (amax / DS4_TURBO3_MAX_D);
        if (scale > DS4_FP8_E4M3_MAX_D) scale = DS4_FP8_E4M3_MAX_D;
        if (scale < 0.0f) scale = 0.0f;

        // Pack 8 values per 3 bytes (matches CUDA layout: bit-stream of
        // 3-bit codes, little-endian).
        device uchar *data_slot = dst_row + tid * DS4_TURBO3_DATA_BYTES_PER_GROUP;
        for (int chunk = 0; chunk < 8; chunk++) {
            const uint v0 = idx[chunk * 8 + 0];
            const uint v1 = idx[chunk * 8 + 1];
            const uint v2 = idx[chunk * 8 + 2];
            const uint v3 = idx[chunk * 8 + 3];
            const uint v4 = idx[chunk * 8 + 4];
            const uint v5 = idx[chunk * 8 + 5];
            const uint v6 = idx[chunk * 8 + 6];
            const uint v7 = idx[chunk * 8 + 7];
            data_slot[chunk * 3 + 0] = (uchar)((v0) | (v1 << 3) | (v2 << 6));
            data_slot[chunk * 3 + 1] = (uchar)((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7));
            data_slot[chunk * 3 + 2] = (uchar)((v5 >> 1) | (v6 << 2) | (v7 << 5));
        }
        // FP8 E4M3 encode of the matched-norm scale into one byte.
        dst_row[data_bytes + tid] = turbo3_fp8_e4m3_encode(scale);
    }

    // RoPE tail copy - one thread handles 64 floats.
    if (tid == 0 && n_rot > 0) {
        const ulong scale_bytes = (ulong)n_groups;
        device uchar *rope_slot = dst_row + data_bytes + scale_bytes;
        device const uchar *src_tail = (device const uchar *)(src_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            rope_slot[i] = src_tail[i];
        }
    }
}

// Ring-aware batched pack - sibling of CUDA's turbo3_kv_pack_batch_kernel.
// Each token writes into ring slot (pos0 + t) % raw_cap.  Grid:
// (n_tokens, 1, 1) x tg(64, 1, 1).
kernel void kernel_dsv4_turbo3_kv_pack_batch_f32(
        device const float *src              [[ buffer(0) ]],
        device       uchar *raw              [[ buffer(1) ]],
        constant     uint  &raw_cap          [[ buffer(2) ]],
        constant     uint  &pos0             [[ buffer(3) ]],
        constant     uint  &n_tokens         [[ buffer(4) ]],
        constant     uint  &head_dim         [[ buffer(5) ]],
        constant     uint  &n_rot            [[ buffer(6) ]],
        constant     ulong &row_bytes        [[ buffer(7) ]],
        constant     int   &signs_on         [[ buffer(8) ]],
        uint                t                [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (t >= n_tokens) return;
    const uint  n_nope    = head_dim - n_rot;
    const uint  n_groups  = n_nope / DS4_TURBO3_GROUP_SIZE;
    const uint  ring_row  = (pos0 + t) % raw_cap;
    device const float *src_row = src + t * head_dim;
    device       uchar *dst_row = raw + ring_row * row_bytes;
    const ulong data_bytes = (ulong)n_nope * 3u / 8u;
    const float inv_sqrt_n = rsqrt(64.0f);

    if (tid < n_groups) {
        float buf[64];
        device const float *gs = src_row + tid * DS4_TURBO3_GROUP_SIZE;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] = gs[i] * DS4_TURBO_SIGNS1_64[i];
        } else {
            for (int i = 0; i < 64; i++) buf[i] = gs[i];
        }
        turbo3_wht64_inplace(buf);
        for (int i = 0; i < 64; i++) buf[i] *= inv_sqrt_n;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] *= DS4_TURBO_SIGNS2_64[i];
        }

        // Matched-norm L2 scale (same as the non-batched pack above).
        float amax    = 0.0f;
        float norm_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v  = buf[i];
            const float av = fabs(v);
            if (av > amax) amax = av;
            norm_sq += v * v;
        }
        const float k_inv = (amax > 1e-12f) ? (DS4_TURBO3_MAX_D / amax) : 1.0f;
        uchar idx[64];
        float recon_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v = buf[i] * k_inv;
            int code = 0;
            for (int j = 0; j < 7; j++) {
                if (v >= DS4_TURBO3_BOUNDS[j]) code = j + 1;
            }
            idx[i] = (uchar)code;
            const float c = DS4_TURBO3_CODEBOOK[code];
            recon_sq += c * c;
        }
        const float recon_norm = sqrt(recon_sq);
        float scale = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm)
                                            : (amax / DS4_TURBO3_MAX_D);
        if (scale > DS4_FP8_E4M3_MAX_D) scale = DS4_FP8_E4M3_MAX_D;
        if (scale < 0.0f) scale = 0.0f;

        device uchar *data_slot = dst_row + tid * DS4_TURBO3_DATA_BYTES_PER_GROUP;
        for (int chunk = 0; chunk < 8; chunk++) {
            const uint v0 = idx[chunk * 8 + 0];
            const uint v1 = idx[chunk * 8 + 1];
            const uint v2 = idx[chunk * 8 + 2];
            const uint v3 = idx[chunk * 8 + 3];
            const uint v4 = idx[chunk * 8 + 4];
            const uint v5 = idx[chunk * 8 + 5];
            const uint v6 = idx[chunk * 8 + 6];
            const uint v7 = idx[chunk * 8 + 7];
            data_slot[chunk * 3 + 0] = (uchar)((v0) | (v1 << 3) | (v2 << 6));
            data_slot[chunk * 3 + 1] = (uchar)((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7));
            data_slot[chunk * 3 + 2] = (uchar)((v5 >> 1) | (v6 << 2) | (v7 << 5));
        }
        dst_row[data_bytes + tid] = turbo3_fp8_e4m3_encode(scale);
    }

    if (tid == 0 && n_rot > 0) {
        const ulong scale_bytes = (ulong)n_groups;
        device uchar *rope_slot = dst_row + data_bytes + scale_bytes;
        device const uchar *src_tail = (device const uchar *)(src_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            rope_slot[i] = src_tail[i];
        }
    }
}

// Float-sim quantize kernel - sibling of CUDA's
// turbo3_kv_quantize_kernel.  Applies turbo3 quantization noise to a
// float [n_tok, head_dim] tensor in place (used for comp_kv round
// trips where attention kernels read comp_kv as floats but the values
// must look like what turbo3 storage would dequant to).  No packing,
// no storage change - input + output are both float.
//
// 64 threads per row, one element per thread.  Uses 64-float
// threadgroup scratch for the WHT butterfly + a second 64-float
// scratch for per-thread amax/norm reductions.
kernel void kernel_dsv4_turbo3_kv_quantize_f32(
        device       float *x                [[ buffer(0) ]],
        constant     uint  &n_tok            [[ buffer(1) ]],
        constant     uint  &head_dim         [[ buffer(2) ]],
        constant     uint  &n_rot            [[ buffer(3) ]],
        constant     int   &signs_on         [[ buffer(4) ]],
        uint                row              [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (row >= n_tok) return;
    const uint n_nope = head_dim - n_rot;
    device float *xr = x + row * head_dim;
    threadgroup float buf[64];
    threadgroup float redux[64];
    const float inv_sqrt_n = rsqrt(64.0f);

    for (uint off = 0; off < n_nope; off += 64) {
        float v = (off + tid < n_nope) ? xr[off + tid] : 0.0f;
        if (signs_on) v *= DS4_TURBO_SIGNS1_64[tid];
        buf[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Forward WHT in threadgroup memory.  Match CUDA wht64_block exactly:
        // low thread (lower index of pair) writes (self + other);
        // high thread writes (other - self) - NOT (self - other).
        for (int stride = 1; stride < 64; stride <<= 1) {
            const uint pair = tid ^ stride;
            const float self_v = buf[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float other_v = buf[pair];
            const float out = (tid < pair) ? (self_v + other_v) : (other_v - self_v);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf[tid] = out;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float rotated = buf[tid] * inv_sqrt_n;
        if (signs_on) rotated *= DS4_TURBO_SIGNS2_64[tid];

        // Block max of |rotated| via threadgroup memory reduce.
        redux[tid] = fabs(rotated);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) redux[tid] = fmax(redux[tid], redux[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float amax = redux[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block sum of rotated*rotated.
        redux[tid] = rotated * rotated;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) redux[tid] = redux[tid] + redux[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float norm_sq = redux[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float k_inv = (amax > 1e-12f) ? (DS4_TURBO3_MAX_D / amax) : 1.0f;

        // Quantize this thread's element to nearest centroid.
        const float rv = rotated * k_inv;
        int code = 0;
        for (int j = 0; j < 7; j++) {
            if (rv >= DS4_TURBO3_BOUNDS[j]) code = j + 1;
        }
        const float centroid = DS4_TURBO3_CODEBOOK[code];

        // Block sum of centroid*centroid -> recon L2.
        redux[tid] = centroid * centroid;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) redux[tid] = redux[tid] + redux[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float recon_norm = sqrt(redux[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm)
                                            : (amax / DS4_TURBO3_MAX_D);
        if (scale > DS4_FP8_E4M3_MAX_D) scale = DS4_FP8_E4M3_MAX_D;

        float dequant = centroid * scale;
        if (signs_on) dequant *= DS4_TURBO_SIGNS2_64[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[tid] = dequant;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inverse WHT (same self-inverse butterfly).
        for (int stride = 1; stride < 64; stride <<= 1) {
            const uint pair = tid ^ stride;
            const float a = buf[tid];
            const float b = buf[pair];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf[tid] = (tid < pair) ? (a + b) : (a - b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float final_v = buf[tid] * inv_sqrt_n;
        if (signs_on) final_v *= DS4_TURBO_SIGNS1_64[tid];

        if (off + tid < n_nope) xr[off + tid] = final_v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Inline-dequant attention decode kernel for the n_tokens=1 decode-token
// simple path.  Sibling of CUDA's attention_decode_mixed_turbo3_kernel.
// Reads packed turbo3 bytes directly via the inline dequant primitive:
// no dequant-to-scratch hop, no extra kernel launch.
//
// Threadgroup memory budget (Apple Silicon 32KB cap):
//   scores[2048]    8KB
//   raw_rows[256]   1KB
//   partial[256]    1KB
//   kv_tile[8*512] 16KB    (ROWS_PER_TILE=8 vs CUDA's 16; Apple
//                           cap is half of Blackwell's 48KB default)
//   plus singletons ~24B
//   total ~26KB, fits.
//
// Grid: (n_tokens, n_head, 1) x threadgroup (256, 1, 1).
// head_dim=512 + blockDim=256 -> each thread owns 2 d's (d0, d1).
struct ds4_metal_args_attn_decode_mixed_turbo3 {
    ulong  row_bytes;
    uint   use_comp_mask;
    uint   n_tokens;
    uint   pos0;
    uint   n_raw;
    uint   raw_cap;
    uint   raw_start;
    uint   n_comp;
    uint   window;
    uint   ratio;
    uint   n_head;
    uint   head_dim;
    uint   n_rot;
    int    signs_on;
};

// Mac comp_kv buffer is always f16 (DS4_GPU_ATTN_COMP_CACHE_F16=1 on Apple),
// so this kernel only supports the half path.
kernel void kernel_dsv4_attention_decode_mixed_turbo3_f32(
        device       float *heads        [[ buffer(0) ]],
        device const float *sinks        [[ buffer(1) ]],
        device const float *q            [[ buffer(2) ]],
        device const uchar *raw_kv_bytes [[ buffer(3) ]],
        device const half  *comp_kv      [[ buffer(4) ]],
        device const float *comp_mask    [[ buffer(5) ]],
        constant struct ds4_metal_args_attn_decode_mixed_turbo3 &args [[ buffer(6) ]],
        uint3   tg_pos_v             [[ threadgroup_position_in_grid ]],
        uint3   tid_v                [[ thread_position_in_threadgroup ]],
        uint3   tpt_v                [[ threads_per_threadgroup ]]) {
    const uint t              = tg_pos_v.x;
    const uint h              = tg_pos_v.y;
    const uint tid            = tid_v.x;
    const uint threads_per_tg = tpt_v.x;
    if (t >= args.n_tokens || h >= args.n_head) return;

    const uint  n_nope     = args.head_dim - args.n_rot;
    const uint  n_groups   = n_nope / DS4_TURBO3_GROUP_SIZE;
    const ulong data_bytes = (ulong)n_nope * 3u / 8u;
    const ulong scale_bytes= (ulong)n_groups;
    const bool  single_all = (args.n_tokens == 1u && args.ratio == 0u);
    const uint  qpos       = args.pos0 + t;
    const uint  first_raw_pos = args.pos0 + args.n_tokens - args.n_raw;
    uint visible_comp = single_all ? args.n_comp
                                    : (args.n_comp ? (qpos + 1u) / args.ratio : 0u);
    if (visible_comp > args.n_comp) visible_comp = args.n_comp;
    device const float *qh = q + ((ulong)t * args.n_head + h) * args.head_dim;

    constexpr uint TURBO3_DECODE_SCORE_CAP = 2048u;
    threadgroup float    scores[TURBO3_DECODE_SCORE_CAP];
    threadgroup uint     raw_rows[256];
    threadgroup float    partial[256];
    threadgroup float    max_s;
    threadgroup float    denom;
    threadgroup uint     raw_count;
    threadgroup uint     raw_first_idx;
    const float scale = rsqrt((float)args.head_dim);

    if (tid == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (args.n_raw != 0) {
            const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
            if (single_all) {
                raw_count = args.n_raw > 256u ? 256u : args.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint lo = first_raw_pos;
                if (args.window != 0 && qpos + 1u > args.window) {
                    const uint wlo = qpos + 1u - args.window;
                    if (wlo > lo) lo = wlo;
                }
                const uint hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        raw_rows[r] = (args.raw_start + raw_first_idx + r) % args.raw_cap;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_score = raw_count + visible_comp;
    float local_max = sinks[h];

    // K-dot: per-thread per-row, inline turbo3 dequant.
    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r] * args.row_bytes;
        float dot = 0.0f;
        float group[64];
        for (uint g = 0; g < n_groups; g++) {
            turbo3_dequant_group64(group, kv_bytes, g, n_nope, args.signs_on);
            for (uint i = 0; i < 64; i++) {
                dot += qh[g * 64 + i] * group[i];
            }
        }
        device const uchar *rope_tail = kv_bytes + data_bytes + scale_bytes;
        for (uint d = 0; d < args.n_rot; d++) {
            dot += qh[n_nope + d] * turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        scores[r] = dot * scale;
        local_max = max(local_max, scores[r]);
    }
    // comp_kv path: f16 input on Mac, promote to float per multiply.
    for (uint c = tid; c < visible_comp; c += threads_per_tg) {
        float add = args.use_comp_mask ? comp_mask[(ulong)t * args.n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            device const half *kvrow = comp_kv + (ulong)c * args.head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) dot += qh[d] * (float)kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = max(local_max, s);
    }

    // Softmax reduction.
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) max_s = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float den_local = 0.0f;
    for (uint i = tid; i < n_score; i += threads_per_tg) {
        scores[i] = exp(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[tid] = den_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = partial[tid] + partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) denom = partial[0] + exp(sinks[h] - max_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // V-acc: tile-batched cooperative shmem dequant.  ROWS_PER_TILE=8
    // on Apple (32KB threadgroup cap) vs 16 on Blackwell (48KB).  More
    // syncs per call but same correctness.
    device float *oh = heads + ((ulong)t * args.n_head + h) * args.head_dim;
    // Half-precision kv_tile halves byte footprint -> fits ROWS_PER_TILE=16
    // within Apple9's 32 KB threadgroup cap (16 KB tile + 8 KB scores +
    // small singletons ~ 26 KB).  Doubles the tile-loop unroll vs CUDA's
    // float tile.  Half mantissa loses ~1e-3 relative precision per V
    // element - softmax weights sum to 1.0 so accumulated drift stays in
    // the noise floor (verified via output coherence parity vs CUDA).
    constexpr uint ROWS_PER_TILE = 20u;
    threadgroup half kv_tile_h[ROWS_PER_TILE * 512u];
    const uint d0 = tid;
    const uint d1 = d0 + 256u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint r_base = 0; r_base < raw_count; r_base += ROWS_PER_TILE) {
        uint tile_rows = raw_count - r_base;
        if (tile_rows > ROWS_PER_TILE) tile_rows = ROWS_PER_TILE;
        const uint total_groups = tile_rows * n_groups;   // <= 16*7 = 112
        if (tid < total_groups) {
            const uint tr = tid / n_groups;
            const uint g  = tid % n_groups;
            device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r_base + tr] * args.row_bytes;
            float buf[64];
            turbo3_dequant_group64(buf, kv_bytes, g, n_nope, args.signs_on);
            threadgroup half *gd = kv_tile_h + (ulong)tr * 512u + (ulong)g * DS4_TURBO3_GROUP_SIZE;
            for (uint i = 0; i < 64; i++) gd[i] = (half)buf[i];
        }
        // RoPE: tile_rows * n_rot floats = up to 16*64 = 1024.  256 threads x 4 each.
        const uint total_rope = tile_rows * args.n_rot;
        for (uint idx = tid; idx < total_rope; idx += threads_per_tg) {
            const uint tr = idx / args.n_rot;
            const uint d  = idx % args.n_rot;
            device const uchar *rope_tail = raw_kv_bytes
                    + (ulong)raw_rows[r_base + tr] * args.row_bytes
                    + data_bytes + scale_bytes;
            kv_tile_h[(ulong)tr * 512u + n_nope + d] =
                    (half)turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < ROWS_PER_TILE; i++) {
            if (i < tile_rows) {
                float s = scores[r_base + i];
                acc0 += (float)kv_tile_h[(ulong)i * 512u + d0] * s;
                acc1 += (float)kv_tile_h[(ulong)i * 512u + d1] * s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint c = 0; c < visible_comp; c++) {
        float s = scores[raw_count + c];
        device const half *kv = comp_kv + (ulong)c * args.head_dim;
        acc0 += (float)kv[d0] * s;
        acc1 += (float)kv[d1] * s;
    }
    oh[d0] = acc0 / denom;
    oh[d1] = acc1 / denom;
}

// Dequant-to-scratch kernel - sibling of CUDA's
// turbo3_kv_dequant_to_scratch_kernel.  Reads `n_rows` packed turbo3 rows
// from `src` (each `src_row_bytes` long) and writes original-basis floats
// into `dst` at the natural [n_rows, head_dim] float layout.  Grid:
// (n_rows, 1, 1) x tg(64, 1, 1).  Thread `tid` in {0..n_groups-1} handles
// its group; thread 0 also copies the RoPE tail.
kernel void kernel_dsv4_turbo3_kv_dequant_to_scratch_f32(
        device const uchar *src              [[ buffer(0) ]],
        device       float *dst              [[ buffer(1) ]],
        constant     uint  &n_rows           [[ buffer(2) ]],
        constant     uint  &head_dim         [[ buffer(3) ]],
        constant     uint  &n_rot            [[ buffer(4) ]],
        constant     ulong &src_row_bytes    [[ buffer(5) ]],
        constant     int   &signs_on         [[ buffer(6) ]],
        uint                row              [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (row >= n_rows) return;
    const uint  n_nope   = head_dim - n_rot;
    const uint  n_groups = n_nope / DS4_TURBO3_GROUP_SIZE;
    device const uchar *src_row = src + row * src_row_bytes;
    device       float *dst_row = dst + row * head_dim;

    if (tid < n_groups) {
        float buf[64];
        turbo3_dequant_group64(buf, src_row, tid, n_nope, signs_on);
        device float *gd = dst_row + tid * DS4_TURBO3_GROUP_SIZE;
        for (int i = 0; i < 64; i++) gd[i] = buf[i];
    }

    if (tid == 0 && n_rot > 0) {
        const ulong data_bytes  = (ulong)n_nope * 3u / 8u;
        const ulong scale_bytes = (ulong)n_groups;
        device const uchar *rope_slot = src_row + data_bytes + scale_bytes;
        device uchar *dst_tail = (device uchar *)(dst_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            dst_tail[i] = rope_slot[i];
        }
    }
}

// =========================================================================
// turbo4 - 4-bit Lloyd-Max codebook, 32 B / 64-element group, 2 nibbles/byte
// =========================================================================
//
// Storage per row (head_dim=512, n_rot=64, n_nope=448, n_groups=7):
//   data:   n_nope * 4 / 8 = 224 B (7 groups x 32 B)
//   scales: 7 B (one FP8 E4M3 byte per group)
//   rope:   n_rot * 4 = 256 B
//   total:  487 B / row  (vs 2048 B fp8, 431 B turbo3)
//
// Compression: 2048/487 = 4.21x per row (vs turbo3 4.75x).  Quality gain:
// 4-bit Lloyd-Max MSE on N(0,1) is 0.0095 vs 3-bit MSE 0.0345 - ~3.6x lower
// quantization noise.  Target use case: workloads where turbo3's +2% ppl
// regression is too much but you still want >4x KV memory savings.
//
// Shares with turbo3: DS4_TURBO_SIGNS{1,2}_64, turbo3_wht64_inplace,
// turbo3_fp8_e4m3_value / encode, turbo3_load_unaligned_f32.

#define DS4_TURBO4_DATA_BYTES_PER_GROUP 32u
#define DS4_TURBO4_MAX_D                2.7326f

constant float DS4_TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

constant float DS4_TURBO4_BOUNDS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};

// 4-bit inline dequant - sibling of turbo3_dequant_group64.  Reads 32
// packed bytes + 1 FP8 scale byte and writes 64 original-basis floats.
static inline void turbo4_dequant_group64(
        thread float        out64[64],
        device const uchar *row_base,
        uint                group_idx,
        uint                n_nope,
        int                 signs_on) {
    device const uchar *data_slot = row_base + group_idx * DS4_TURBO4_DATA_BYTES_PER_GROUP;
    const uint  data_bytes  = n_nope / 2u;   // 4 bits per element
    const uchar scale_byte  = row_base[data_bytes + group_idx];
    const float scale = turbo3_fp8_e4m3_value((int)scale_byte);

    // Pre-scaled centroid cache - 16 entries vs turbo3's 8.
    float sc[16];
    for (int c = 0; c < 16; c++) sc[c] = DS4_TURBO4_CODEBOOK[c] * scale;

    // Unpack 32 bytes -> 64 nibbles -> 64 rotated floats.  Low nibble first.
    for (int i = 0; i < 32; i++) {
        const uint b = data_slot[i];
        out64[2*i + 0] = sc[(b)      & 0xFu];
        out64[2*i + 1] = sc[(b >> 4) & 0xFu];
    }

    // Inverse rotation: signs2 -> WHT -> 1/sqrt(64) -> signs1.
    if (signs_on) {
        for (int i = 0; i < 64; i++) out64[i] *= DS4_TURBO_SIGNS2_64[i];
    }
    turbo3_wht64_inplace(out64);
    const float inv_sqrt_n = rsqrt(64.0f);
    for (int i = 0; i < 64; i++) out64[i] *= inv_sqrt_n;
    if (signs_on) {
        for (int i = 0; i < 64; i++) out64[i] *= DS4_TURBO_SIGNS1_64[i];
    }
}

// Per-thread 4-bit quantize: linear scan over 15 bounds -> idx [0..15].
static inline uchar turbo4_quant_idx(float v) {
    int code = 0;
    for (int j = 0; j < 15; j++) {
        if (v >= DS4_TURBO4_BOUNDS[j]) code = j + 1;
    }
    return (uchar)code;
}

kernel void kernel_dsv4_turbo4_kv_pack_f32(
        device const float *src              [[ buffer(0) ]],
        device       uchar *dst              [[ buffer(1) ]],
        constant     uint  &n_tok            [[ buffer(2) ]],
        constant     uint  &head_dim         [[ buffer(3) ]],
        constant     uint  &n_rot            [[ buffer(4) ]],
        constant     ulong &dst_row_bytes    [[ buffer(5) ]],
        constant     int   &signs_on         [[ buffer(6) ]],
        uint                row              [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (row >= n_tok) return;
    const uint n_nope    = head_dim - n_rot;
    const uint n_groups  = n_nope / DS4_TURBO3_GROUP_SIZE;
    device const float *src_row = src + row * head_dim;
    device       uchar *dst_row = dst + row * dst_row_bytes;
    const ulong data_bytes = (ulong)n_nope / 2u;
    const float inv_sqrt_n = rsqrt(64.0f);

    if (tid < n_groups) {
        float buf[64];
        device const float *gs = src_row + tid * DS4_TURBO3_GROUP_SIZE;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] = gs[i] * DS4_TURBO_SIGNS1_64[i];
        } else {
            for (int i = 0; i < 64; i++) buf[i] = gs[i];
        }
        turbo3_wht64_inplace(buf);
        for (int i = 0; i < 64; i++) buf[i] *= inv_sqrt_n;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] *= DS4_TURBO_SIGNS2_64[i];
        }

        float amax    = 0.0f;
        float norm_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v  = buf[i];
            const float av = fabs(v);
            if (av > amax) amax = av;
            norm_sq += v * v;
        }
        const float k_inv = (amax > 1e-12f) ? (DS4_TURBO4_MAX_D / amax) : 1.0f;

        uchar idx[64];
        float recon_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const uchar code = turbo4_quant_idx(buf[i] * k_inv);
            idx[i] = code;
            const float c = DS4_TURBO4_CODEBOOK[code];
            recon_sq += c * c;
        }
        const float recon_norm = sqrt(recon_sq);
        float scale = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm)
                                            : (amax / DS4_TURBO4_MAX_D);
        if (scale > DS4_FP8_E4M3_MAX_D) scale = DS4_FP8_E4M3_MAX_D;
        if (scale < 0.0f) scale = 0.0f;

        device uchar *data_slot = dst_row + tid * DS4_TURBO4_DATA_BYTES_PER_GROUP;
        for (int i = 0; i < 32; i++) {
            data_slot[i] = (uchar)((idx[2*i] & 0xFu) | ((idx[2*i + 1] & 0xFu) << 4));
        }
        dst_row[data_bytes + tid] = turbo3_fp8_e4m3_encode(scale);
    }

    if (tid == 0 && n_rot > 0) {
        const ulong scale_bytes = (ulong)n_groups;
        device uchar *rope_slot = dst_row + data_bytes + scale_bytes;
        device const uchar *src_tail = (device const uchar *)(src_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            rope_slot[i] = src_tail[i];
        }
    }
}

kernel void kernel_dsv4_turbo4_kv_pack_batch_f32(
        device const float *src              [[ buffer(0) ]],
        device       uchar *raw              [[ buffer(1) ]],
        constant     uint  &raw_cap          [[ buffer(2) ]],
        constant     uint  &pos0             [[ buffer(3) ]],
        constant     uint  &n_tokens         [[ buffer(4) ]],
        constant     uint  &head_dim         [[ buffer(5) ]],
        constant     uint  &n_rot            [[ buffer(6) ]],
        constant     ulong &row_bytes        [[ buffer(7) ]],
        constant     int   &signs_on         [[ buffer(8) ]],
        uint                t                [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (t >= n_tokens) return;
    const uint  n_nope    = head_dim - n_rot;
    const uint  n_groups  = n_nope / DS4_TURBO3_GROUP_SIZE;
    const uint  ring_row  = (pos0 + t) % raw_cap;
    device const float *src_row = src + t * head_dim;
    device       uchar *dst_row = raw + ring_row * row_bytes;
    const ulong data_bytes = (ulong)n_nope / 2u;
    const float inv_sqrt_n = rsqrt(64.0f);

    if (tid < n_groups) {
        float buf[64];
        device const float *gs = src_row + tid * DS4_TURBO3_GROUP_SIZE;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] = gs[i] * DS4_TURBO_SIGNS1_64[i];
        } else {
            for (int i = 0; i < 64; i++) buf[i] = gs[i];
        }
        turbo3_wht64_inplace(buf);
        for (int i = 0; i < 64; i++) buf[i] *= inv_sqrt_n;
        if (signs_on) {
            for (int i = 0; i < 64; i++) buf[i] *= DS4_TURBO_SIGNS2_64[i];
        }

        float amax    = 0.0f;
        float norm_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const float v  = buf[i];
            const float av = fabs(v);
            if (av > amax) amax = av;
            norm_sq += v * v;
        }
        const float k_inv = (amax > 1e-12f) ? (DS4_TURBO4_MAX_D / amax) : 1.0f;

        uchar idx[64];
        float recon_sq = 0.0f;
        for (int i = 0; i < 64; i++) {
            const uchar code = turbo4_quant_idx(buf[i] * k_inv);
            idx[i] = code;
            const float c = DS4_TURBO4_CODEBOOK[code];
            recon_sq += c * c;
        }
        const float recon_norm = sqrt(recon_sq);
        float scale = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm)
                                            : (amax / DS4_TURBO4_MAX_D);
        if (scale > DS4_FP8_E4M3_MAX_D) scale = DS4_FP8_E4M3_MAX_D;
        if (scale < 0.0f) scale = 0.0f;

        device uchar *data_slot = dst_row + tid * DS4_TURBO4_DATA_BYTES_PER_GROUP;
        for (int i = 0; i < 32; i++) {
            data_slot[i] = (uchar)((idx[2*i] & 0xFu) | ((idx[2*i + 1] & 0xFu) << 4));
        }
        dst_row[data_bytes + tid] = turbo3_fp8_e4m3_encode(scale);
    }

    if (tid == 0 && n_rot > 0) {
        const ulong scale_bytes = (ulong)n_groups;
        device uchar *rope_slot = dst_row + data_bytes + scale_bytes;
        device const uchar *src_tail = (device const uchar *)(src_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            rope_slot[i] = src_tail[i];
        }
    }
}

kernel void kernel_dsv4_turbo4_kv_dequant_to_scratch_f32(
        device const uchar *src              [[ buffer(0) ]],
        device       float *dst              [[ buffer(1) ]],
        constant     uint  &n_rows           [[ buffer(2) ]],
        constant     uint  &head_dim         [[ buffer(3) ]],
        constant     uint  &n_rot            [[ buffer(4) ]],
        constant     ulong &src_row_bytes    [[ buffer(5) ]],
        constant     int   &signs_on         [[ buffer(6) ]],
        uint                row              [[ threadgroup_position_in_grid ]],
        uint                tid              [[ thread_position_in_threadgroup ]]) {
    if (row >= n_rows) return;
    const uint  n_nope   = head_dim - n_rot;
    const uint  n_groups = n_nope / DS4_TURBO3_GROUP_SIZE;
    device const uchar *src_row = src + row * src_row_bytes;
    device       float *dst_row = dst + row * head_dim;

    if (tid < n_groups) {
        float buf[64];
        turbo4_dequant_group64(buf, src_row, tid, n_nope, signs_on);
        device float *gd = dst_row + tid * DS4_TURBO3_GROUP_SIZE;
        for (int i = 0; i < 64; i++) gd[i] = buf[i];
    }

    if (tid == 0 && n_rot > 0) {
        const ulong data_bytes  = (ulong)n_nope / 2u;
        const ulong scale_bytes = (ulong)n_groups;
        device const uchar *rope_slot = src_row + data_bytes + scale_bytes;
        device uchar *dst_tail = (device uchar *)(dst_row + n_nope);
        for (uint i = 0; i < (uint)n_rot * sizeof(float); i++) {
            dst_tail[i] = rope_slot[i];
        }
    }
}

// turbo4 inline-dequant attention decode - sibling of the turbo3 kernel
// above.  Same threadgroup memory budget (kv_tile 16 KB at ROWS_PER_TILE=8
// + scores 8 KB + small).
struct ds4_metal_args_attn_decode_mixed_turbo4 {
    ulong  row_bytes;
    uint   use_comp_mask;
    uint   n_tokens;
    uint   pos0;
    uint   n_raw;
    uint   raw_cap;
    uint   raw_start;
    uint   n_comp;
    uint   window;
    uint   ratio;
    uint   n_head;
    uint   head_dim;
    uint   n_rot;
    int    signs_on;
};

kernel void kernel_dsv4_attention_decode_mixed_turbo4_f32(
        device       float *heads        [[ buffer(0) ]],
        device const float *sinks        [[ buffer(1) ]],
        device const float *q            [[ buffer(2) ]],
        device const uchar *raw_kv_bytes [[ buffer(3) ]],
        device const half  *comp_kv      [[ buffer(4) ]],
        device const float *comp_mask    [[ buffer(5) ]],
        constant struct ds4_metal_args_attn_decode_mixed_turbo4 &args [[ buffer(6) ]],
        uint3   tg_pos_v             [[ threadgroup_position_in_grid ]],
        uint3   tid_v                [[ thread_position_in_threadgroup ]],
        uint3   tpt_v                [[ threads_per_threadgroup ]]) {
    const uint t              = tg_pos_v.x;
    const uint h              = tg_pos_v.y;
    const uint tid            = tid_v.x;
    const uint threads_per_tg = tpt_v.x;
    if (t >= args.n_tokens || h >= args.n_head) return;

    const uint  n_nope     = args.head_dim - args.n_rot;
    const uint  n_groups   = n_nope / DS4_TURBO3_GROUP_SIZE;
    const ulong data_bytes = (ulong)n_nope / 2u;
    const ulong scale_bytes= (ulong)n_groups;
    const bool  single_all = (args.n_tokens == 1u && args.ratio == 0u);
    const uint  qpos       = args.pos0 + t;
    const uint  first_raw_pos = args.pos0 + args.n_tokens - args.n_raw;
    uint visible_comp = single_all ? args.n_comp
                                    : (args.n_comp ? (qpos + 1u) / args.ratio : 0u);
    if (visible_comp > args.n_comp) visible_comp = args.n_comp;
    device const float *qh = q + ((ulong)t * args.n_head + h) * args.head_dim;

    constexpr uint TURBO4_DECODE_SCORE_CAP = 2048u;
    threadgroup float    scores[TURBO4_DECODE_SCORE_CAP];
    threadgroup uint     raw_rows[256];
    threadgroup float    partial[256];
    threadgroup float    max_s;
    threadgroup float    denom;
    threadgroup uint     raw_count;
    threadgroup uint     raw_first_idx;
    const float scale = rsqrt((float)args.head_dim);

    if (tid == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (args.n_raw != 0) {
            const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
            if (single_all) {
                raw_count = args.n_raw > 256u ? 256u : args.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint lo = first_raw_pos;
                if (args.window != 0 && qpos + 1u > args.window) {
                    const uint wlo = qpos + 1u - args.window;
                    if (wlo > lo) lo = wlo;
                }
                const uint hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        raw_rows[r] = (args.raw_start + raw_first_idx + r) % args.raw_cap;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_score = raw_count + visible_comp;
    float local_max = sinks[h];

    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r] * args.row_bytes;
        float dot = 0.0f;
        float group[64];
        for (uint g = 0; g < n_groups; g++) {
            turbo4_dequant_group64(group, kv_bytes, g, n_nope, args.signs_on);
            for (uint i = 0; i < 64; i++) {
                dot += qh[g * 64 + i] * group[i];
            }
        }
        device const uchar *rope_tail = kv_bytes + data_bytes + scale_bytes;
        for (uint d = 0; d < args.n_rot; d++) {
            dot += qh[n_nope + d] * turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        scores[r] = dot * scale;
        local_max = max(local_max, scores[r]);
    }
    for (uint c = tid; c < visible_comp; c += threads_per_tg) {
        float add = args.use_comp_mask ? comp_mask[(ulong)t * args.n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            device const half *kvrow = comp_kv + (ulong)c * args.head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < args.head_dim; d++) dot += qh[d] * (float)kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = max(local_max, s);
    }

    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) max_s = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float den_local = 0.0f;
    for (uint i = tid; i < n_score; i += threads_per_tg) {
        scores[i] = exp(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[tid] = den_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = partial[tid] + partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) denom = partial[0] + exp(sinks[h] - max_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *oh = heads + ((ulong)t * args.n_head + h) * args.head_dim;
    // Half tile -> ROWS_PER_TILE=16 (same trick as turbo3 above).
    constexpr uint ROWS_PER_TILE = 20u;
    threadgroup half kv_tile_h[ROWS_PER_TILE * 512u];
    const uint d0 = tid;
    const uint d1 = d0 + 256u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint r_base = 0; r_base < raw_count; r_base += ROWS_PER_TILE) {
        uint tile_rows = raw_count - r_base;
        if (tile_rows > ROWS_PER_TILE) tile_rows = ROWS_PER_TILE;
        const uint total_groups = tile_rows * n_groups;
        if (tid < total_groups) {
            const uint tr = tid / n_groups;
            const uint g  = tid % n_groups;
            device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r_base + tr] * args.row_bytes;
            float buf[64];
            turbo4_dequant_group64(buf, kv_bytes, g, n_nope, args.signs_on);
            threadgroup half *gd = kv_tile_h + (ulong)tr * 512u + (ulong)g * DS4_TURBO3_GROUP_SIZE;
            for (uint i = 0; i < 64; i++) gd[i] = (half)buf[i];
        }
        const uint total_rope = tile_rows * args.n_rot;
        for (uint idx = tid; idx < total_rope; idx += threads_per_tg) {
            const uint tr = idx / args.n_rot;
            const uint d  = idx % args.n_rot;
            device const uchar *rope_tail = raw_kv_bytes
                    + (ulong)raw_rows[r_base + tr] * args.row_bytes
                    + data_bytes + scale_bytes;
            kv_tile_h[(ulong)tr * 512u + n_nope + d] =
                    (half)turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < ROWS_PER_TILE; i++) {
            if (i < tile_rows) {
                float s = scores[r_base + i];
                acc0 += (float)kv_tile_h[(ulong)i * 512u + d0] * s;
                acc1 += (float)kv_tile_h[(ulong)i * 512u + d1] * s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint c = 0; c < visible_comp; c++) {
        float s = scores[raw_count + c];
        device const half *kv = comp_kv + (ulong)c * args.head_dim;
        acc0 += (float)kv[d0] * s;
        acc1 += (float)kv[d1] * s;
    }
    oh[d0] = acc0 / denom;
    oh[d1] = acc1 / denom;
}

// =========================================================================
// h8 head-batched Flash attention with online softmax
//
// Batches 8 heads per threadgroup so the cooperatively-dequanted K=V tile
// is reused across all 8 heads (single dequant pass per row).  Uses
// Flash-attention online softmax so no per-head scores buffer is needed
// in threadgroup memory.  1 simdgroup per head, 32 threads per simdgroup
// covering 32 dim-lanes x 16 output dims per thread = 512 dims per head.
//
// Threadgroup memory:
//   kv_tile_h[TILE_C * 512]  half =  8KB at TILE_C=8
//   raw_rows[256]            uint =  1KB
//   raw_count, raw_first_idx       =  8 B
//   total                          ~9KB  (lots of headroom)
//
// vs the per-head single-head kernel: dequants raw rows ONCE instead of
// twice (K-dot pass + V-acc pass), and shares dequanted tile across 8
// heads instead of 1 head per TG.  8x dequant reuse.
//
// This is v1 - no simdgroup_matrix yet, just simd_sum for K-dot reduction
// inside each per-head simdgroup.  v2 can swap simdgroup_matrix in for
// both K-dot and V-acc as a follow-up.
// =========================================================================

struct ds4_metal_args_attn_h8_turbo3 {
    uint64_t row_bytes;
    uint32_t use_comp_mask;
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    int      signs_on;
};

kernel void kernel_dsv4_attention_decode_h8_turbo3_f32(
        device       float *heads        [[ buffer(0) ]],
        device const float *sinks        [[ buffer(1) ]],
        device const float *q            [[ buffer(2) ]],
        device const uchar *raw_kv_bytes [[ buffer(3) ]],
        device const half  *comp_kv      [[ buffer(4) ]],
        device const float *comp_mask    [[ buffer(5) ]],
        constant struct ds4_metal_args_attn_h8_turbo3 &args [[ buffer(6) ]],
        uint3   tg_pos_v             [[ threadgroup_position_in_grid ]],
        uint3   tid_v                [[ thread_position_in_threadgroup ]],
        uint3   tpt_v                [[ threads_per_threadgroup ]]) {
    constexpr uint HEADS_PER_TG     = 8u;
    constexpr uint TILE_C           = 28u;
    constexpr uint THREADS_PER_HEAD = 32u;    // = SIMD width, 1 simdgroup per head
    constexpr uint DIMS_PER_THREAD  = 16u;    // 512 / 32 = 16 dims per thread per head

    const uint t  = tg_pos_v.x;
    const uint h0 = tg_pos_v.y * HEADS_PER_TG;   // first head in this tile
    const uint tid            = tid_v.x;
    const uint threads_per_tg = tpt_v.x;
    if (t >= args.n_tokens || h0 >= args.n_head) return;

    const uint  n_nope     = args.head_dim - args.n_rot;
    const uint  n_groups   = n_nope / DS4_TURBO3_GROUP_SIZE;
    const ulong data_bytes = (ulong)n_nope * 3u / 8u;
    const ulong scale_bytes= (ulong)n_groups;
    const bool  single_all = (args.n_tokens == 1u && args.ratio == 0u);
    const uint  qpos       = args.pos0 + t;
    const uint  first_raw_pos = args.pos0 + args.n_tokens - args.n_raw;

    uint visible_comp = single_all ? args.n_comp
                                   : (args.n_comp ? (qpos + 1u) / args.ratio : 0u);
    if (visible_comp > args.n_comp) visible_comp = args.n_comp;

    threadgroup half  kv_tile_h[TILE_C * 512u];
    threadgroup uint  raw_rows[256];
    threadgroup uint  raw_count;
    threadgroup uint  raw_first_idx;

    // Per-thread role: tid -> (head_local in [0..7], dim_lane in [0..31])
    const uint  my_head_local = tid / THREADS_PER_HEAD;
    const uint  my_dim_lane   = tid % THREADS_PER_HEAD;
    const uint  my_d0         = my_dim_lane * DIMS_PER_THREAD;
    const uint  my_head_abs   = h0 + my_head_local;
    const bool  head_valid    = (my_head_abs < args.n_head);

    device const float *qh = q + ((ulong)t * args.n_head + my_head_abs) * args.head_dim;
    device       float *oh = heads + ((ulong)t * args.n_head + my_head_abs) * args.head_dim;

    // Per-thread online-softmax state (registers).  q_reg caches my dim-
    // range slice of Q so the K-dot inner loop touches no device memory.
    float acc[DIMS_PER_THREAD];
    float q_reg[DIMS_PER_THREAD];
    for (uint i = 0; i < DIMS_PER_THREAD; i++) {
        acc[i]   = 0.0f;
        q_reg[i] = head_valid ? qh[my_d0 + i] : 0.0f;
    }
    float M = -INFINITY;
    float L = 0.0f;
    const float scale_qk = rsqrt((float)args.head_dim);

    // Compute raw_count / raw_first_idx (one thread, then broadcast).
    if (tid == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (args.n_raw != 0) {
            const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
            if (single_all) {
                raw_count = args.n_raw > 256u ? 256u : args.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint lo = first_raw_pos;
                if (args.window != 0 && qpos + 1u > args.window) {
                    const uint wlo = qpos + 1u - args.window;
                    if (wlo > lo) lo = wlo;
                }
                const uint hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        raw_rows[r] = (args.raw_start + raw_first_idx + r) % args.raw_cap;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: process K=V chunks of TILE_C rows.
    for (uint r_base = 0; r_base < raw_count; r_base += TILE_C) {
        const uint tile_rows = (raw_count - r_base) < TILE_C ? (raw_count - r_base) : TILE_C;

        // 1. Cooperatively dequant tile into kv_tile_h.
        const uint total_groups = tile_rows * n_groups;
        if (tid < total_groups) {
            const uint tr = tid / n_groups;
            const uint g  = tid % n_groups;
            device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r_base + tr] * args.row_bytes;
            float buf[64];
            turbo3_dequant_group64(buf, kv_bytes, g, n_nope, args.signs_on);
            threadgroup half *gd = kv_tile_h + (ulong)tr * 512u + (ulong)g * DS4_TURBO3_GROUP_SIZE;
            for (uint i = 0; i < 64; i++) gd[i] = (half)buf[i];
        }
        // RoPE tail
        const uint total_rope = tile_rows * args.n_rot;
        for (uint idx = tid; idx < total_rope; idx += threads_per_tg) {
            const uint tr = idx / args.n_rot;
            const uint d  = idx % args.n_rot;
            device const uchar *rope_tail = raw_kv_bytes
                    + (ulong)raw_rows[r_base + tr] * args.row_bytes
                    + data_bytes + scale_bytes;
            kv_tile_h[(ulong)tr * 512u + n_nope + d] =
                    (half)turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. K-dot for my head, per row in tile.  Each simdgroup is 1 head;
        // 32 lanes cover 32 x 16 = 512 dims via simd_sum reduction.
        float scores_chunk[TILE_C];
        for (uint j = 0; j < TILE_C; j++) {
            float s_partial = 0.0f;
            if (j < tile_rows && head_valid) {
                threadgroup const half *kvrow = kv_tile_h + (ulong)j * 512u;
                for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                    s_partial += q_reg[i] * (float)kvrow[my_d0 + i];
                }
            }
            scores_chunk[j] = simd_sum(s_partial) * scale_qk;
        }

        // 3. Online softmax update for this chunk.
        if (head_valid) {
            float m_chunk = -INFINITY;
            for (uint j = 0; j < tile_rows; j++) {
                m_chunk = max(m_chunk, scores_chunk[j]);
            }
            const float M_new = max(M, m_chunk);
            const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
            float l_chunk = 0.0f;
            for (uint j = 0; j < tile_rows; j++) {
                scores_chunk[j] = precise::exp(scores_chunk[j] - M_new);
                l_chunk += scores_chunk[j];
            }
            L = L * ms + l_chunk;
            for (uint i = 0; i < DIMS_PER_THREAD; i++) acc[i] *= ms;

            // 4. Accumulate weighted V into per-thread acc.
            for (uint j = 0; j < tile_rows; j++) {
                const float w = scores_chunk[j];
                /* Sparse V (TQ+ paper, "Why V is Free, K is Everything"):
                 * skip rows whose softmax weight is negligible.  No PPL
                 * impact since the contribution `w * V` is below fp16
                 * mantissa precision; ~22.8% decode win on Tom's TQ+
                 * Metal MoE at 32K context. */
                if (w < 1.0e-6f) continue;
                threadgroup const half *kvrow = kv_tile_h + (ulong)j * 512u;
                for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                    acc[i] += w * (float)kvrow[my_d0 + i];
                }
            }
            M = M_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. comp_kv rows - process one at a time with online softmax.
    for (uint c = 0; c < visible_comp; c++) {
        float add = args.use_comp_mask ? comp_mask[(ulong)t * args.n_comp + c] : 0.0f;
        if (add <= -1.0e20f) continue;
        if (!head_valid) continue;

        device const half *kvrow = comp_kv + (ulong)c * args.head_dim;
        float s_partial = 0.0f;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) {
            s_partial += q_reg[i] * (float)kvrow[my_d0 + i];
        }
        const float s_full = simd_sum(s_partial) * scale_qk + add;

        const float M_new = max(M, s_full);
        const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
        const float vs = precise::exp(s_full - M_new);
        L = L * ms + vs;
        /* Sparse V: when vs is below noise threshold, skip the device-
         * memory V reads (kvrow accesses) for this row.  acc still gets
         * scaled by ms - only the `+ vs * V` contribution drops out. */
        if (vs < 1.0e-6f) {
            for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                acc[i] *= ms;
            }
        } else {
            for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                acc[i] = acc[i] * ms + vs * (float)kvrow[my_d0 + i];
            }
        }
        M = M_new;
    }

    // 6. Sink contribution: adds to L but not to acc (acc is V-weighted only).
    if (head_valid) {
        const float sink_s = sinks[my_head_abs];
        const float M_new = max(M, sink_s);
        const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
        const float vs = precise::exp(sink_s - M_new);
        L = L * ms + vs;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) acc[i] *= ms;
        M = M_new;

        // 7. Normalize and write out.
        const float inv_L = 1.0f / L;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) {
            oh[my_d0 + i] = acc[i] * inv_L;
        }
    }
}

// turbo4 sibling of the h8 Flash kernel above.  Only the dequant primitive
// and data_bytes formula change (4-bit vs 3-bit).
struct ds4_metal_args_attn_h8_turbo4 {
    uint64_t row_bytes;
    uint32_t use_comp_mask;
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    int      signs_on;
};

kernel void kernel_dsv4_attention_decode_h8_turbo4_f32(
        device       float *heads        [[ buffer(0) ]],
        device const float *sinks        [[ buffer(1) ]],
        device const float *q            [[ buffer(2) ]],
        device const uchar *raw_kv_bytes [[ buffer(3) ]],
        device const half  *comp_kv      [[ buffer(4) ]],
        device const float *comp_mask    [[ buffer(5) ]],
        constant struct ds4_metal_args_attn_h8_turbo4 &args [[ buffer(6) ]],
        uint3   tg_pos_v             [[ threadgroup_position_in_grid ]],
        uint3   tid_v                [[ thread_position_in_threadgroup ]],
        uint3   tpt_v                [[ threads_per_threadgroup ]]) {
    constexpr uint HEADS_PER_TG     = 8u;
    constexpr uint TILE_C           = 28u;
    constexpr uint THREADS_PER_HEAD = 32u;
    constexpr uint DIMS_PER_THREAD  = 16u;

    const uint t  = tg_pos_v.x;
    const uint h0 = tg_pos_v.y * HEADS_PER_TG;
    const uint tid            = tid_v.x;
    const uint threads_per_tg = tpt_v.x;
    if (t >= args.n_tokens || h0 >= args.n_head) return;

    const uint  n_nope     = args.head_dim - args.n_rot;
    const uint  n_groups   = n_nope / DS4_TURBO3_GROUP_SIZE;
    const ulong data_bytes = (ulong)n_nope / 2u;        // 4 bits / element
    const ulong scale_bytes= (ulong)n_groups;
    const bool  single_all = (args.n_tokens == 1u && args.ratio == 0u);
    const uint  qpos       = args.pos0 + t;
    const uint  first_raw_pos = args.pos0 + args.n_tokens - args.n_raw;

    uint visible_comp = single_all ? args.n_comp
                                   : (args.n_comp ? (qpos + 1u) / args.ratio : 0u);
    if (visible_comp > args.n_comp) visible_comp = args.n_comp;

    threadgroup half  kv_tile_h[TILE_C * 512u];
    threadgroup uint  raw_rows[256];
    threadgroup uint  raw_count;
    threadgroup uint  raw_first_idx;

    const uint  my_head_local = tid / THREADS_PER_HEAD;
    const uint  my_dim_lane   = tid % THREADS_PER_HEAD;
    const uint  my_d0         = my_dim_lane * DIMS_PER_THREAD;
    const uint  my_head_abs   = h0 + my_head_local;
    const bool  head_valid    = (my_head_abs < args.n_head);

    device const float *qh = q + ((ulong)t * args.n_head + my_head_abs) * args.head_dim;
    device       float *oh = heads + ((ulong)t * args.n_head + my_head_abs) * args.head_dim;

    float acc[DIMS_PER_THREAD];
    float q_reg[DIMS_PER_THREAD];
    for (uint i = 0; i < DIMS_PER_THREAD; i++) {
        acc[i]   = 0.0f;
        q_reg[i] = head_valid ? qh[my_d0 + i] : 0.0f;
    }
    float M = -INFINITY;
    float L = 0.0f;
    const float scale_qk = rsqrt((float)args.head_dim);

    if (tid == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (args.n_raw != 0) {
            const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
            if (single_all) {
                raw_count = args.n_raw > 256u ? 256u : args.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint lo = first_raw_pos;
                if (args.window != 0 && qpos + 1u > args.window) {
                    const uint wlo = qpos + 1u - args.window;
                    if (wlo > lo) lo = wlo;
                }
                const uint hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < raw_count; r += threads_per_tg) {
        raw_rows[r] = (args.raw_start + raw_first_idx + r) % args.raw_cap;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r_base = 0; r_base < raw_count; r_base += TILE_C) {
        const uint tile_rows = (raw_count - r_base) < TILE_C ? (raw_count - r_base) : TILE_C;

        const uint total_groups = tile_rows * n_groups;
        if (tid < total_groups) {
            const uint tr = tid / n_groups;
            const uint g  = tid % n_groups;
            device const uchar *kv_bytes = raw_kv_bytes + (ulong)raw_rows[r_base + tr] * args.row_bytes;
            float buf[64];
            turbo4_dequant_group64(buf, kv_bytes, g, n_nope, args.signs_on);
            threadgroup half *gd = kv_tile_h + (ulong)tr * 512u + (ulong)g * DS4_TURBO3_GROUP_SIZE;
            for (uint i = 0; i < 64; i++) gd[i] = (half)buf[i];
        }
        const uint total_rope = tile_rows * args.n_rot;
        for (uint idx = tid; idx < total_rope; idx += threads_per_tg) {
            const uint tr = idx / args.n_rot;
            const uint d  = idx % args.n_rot;
            device const uchar *rope_tail = raw_kv_bytes
                    + (ulong)raw_rows[r_base + tr] * args.row_bytes
                    + data_bytes + scale_bytes;
            kv_tile_h[(ulong)tr * 512u + n_nope + d] =
                    (half)turbo3_load_unaligned_f32(rope_tail + d * sizeof(float));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scores_chunk[TILE_C];
        for (uint j = 0; j < TILE_C; j++) {
            float s_partial = 0.0f;
            if (j < tile_rows && head_valid) {
                threadgroup const half *kvrow = kv_tile_h + (ulong)j * 512u;
                for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                    s_partial += q_reg[i] * (float)kvrow[my_d0 + i];
                }
            }
            scores_chunk[j] = simd_sum(s_partial) * scale_qk;
        }

        if (head_valid) {
            float m_chunk = -INFINITY;
            for (uint j = 0; j < tile_rows; j++) m_chunk = max(m_chunk, scores_chunk[j]);
            const float M_new = max(M, m_chunk);
            const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
            float l_chunk = 0.0f;
            for (uint j = 0; j < tile_rows; j++) {
                scores_chunk[j] = precise::exp(scores_chunk[j] - M_new);
                l_chunk += scores_chunk[j];
            }
            L = L * ms + l_chunk;
            for (uint i = 0; i < DIMS_PER_THREAD; i++) acc[i] *= ms;
            for (uint j = 0; j < tile_rows; j++) {
                const float w = scores_chunk[j];
                /* Sparse V (TQ+ paper, "Why V is Free, K is Everything"):
                 * skip rows whose softmax weight is negligible.  No PPL
                 * impact since the contribution `w * V` is below fp16
                 * mantissa precision; ~22.8% decode win on Tom's TQ+
                 * Metal MoE at 32K context. */
                if (w < 1.0e-6f) continue;
                threadgroup const half *kvrow = kv_tile_h + (ulong)j * 512u;
                for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                    acc[i] += w * (float)kvrow[my_d0 + i];
                }
            }
            M = M_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint c = 0; c < visible_comp; c++) {
        float add = args.use_comp_mask ? comp_mask[(ulong)t * args.n_comp + c] : 0.0f;
        if (add <= -1.0e20f) continue;
        if (!head_valid) continue;

        device const half *kvrow = comp_kv + (ulong)c * args.head_dim;
        float s_partial = 0.0f;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) {
            s_partial += q_reg[i] * (float)kvrow[my_d0 + i];
        }
        const float s_full = simd_sum(s_partial) * scale_qk + add;

        const float M_new = max(M, s_full);
        const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
        const float vs = precise::exp(s_full - M_new);
        L = L * ms + vs;
        /* Sparse V: when vs is below noise threshold, skip the device-
         * memory V reads (kvrow accesses) for this row.  acc still gets
         * scaled by ms - only the `+ vs * V` contribution drops out. */
        if (vs < 1.0e-6f) {
            for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                acc[i] *= ms;
            }
        } else {
            for (uint i = 0; i < DIMS_PER_THREAD; i++) {
                acc[i] = acc[i] * ms + vs * (float)kvrow[my_d0 + i];
            }
        }
        M = M_new;
    }

    if (head_valid) {
        const float sink_s = sinks[my_head_abs];
        const float M_new = max(M, sink_s);
        const float ms = (isinf(M) && M < 0.0f) ? 0.0f : precise::exp(M - M_new);
        const float vs = precise::exp(sink_s - M_new);
        L = L * ms + vs;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) acc[i] *= ms;
        M = M_new;

        const float inv_L = 1.0f / L;
        for (uint i = 0; i < DIMS_PER_THREAD; i++) {
            oh[my_d0 + i] = acc[i] * inv_L;
        }
    }
}

