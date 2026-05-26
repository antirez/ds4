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
