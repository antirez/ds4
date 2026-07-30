#include "pulsar_cuda_internal.h"

/* raw_f16: per-call flag describing the storage format of the PASSED raw KV
 * operand (__half when set, f32 otherwise).  The store kernels already round
 * every raw row through __float2half, so a __half load + __half2float returns
 * the exact float the f32 path reads — bit-identical scores/outputs; f16 mode
 * only changes storage and read traffic. */
__device__ static inline float raw_kv_ld(const float *raw_kv, int f16, uint64_t idx) {
    return f16 ? __half2float(((const __half *)raw_kv)[idx]) : raw_kv[idx];
}

/* float4-shaped raw load: c4-th float4 of the row starting at flat element
 * row_base.  Keeps the vectorized load in f32 mode; f16 mode loads the same
 * four values scalar (identical floats, so downstream math is unchanged). */
__device__ static inline float4 raw_kv_ld4(const float *raw_kv, int f16, uint64_t row_base, uint32_t c4) {
    if (f16) {
        const uint64_t b = row_base + ((uint64_t)c4 << 2);
        float4 v;
        v.x = raw_kv_ld(raw_kv, 1, b + 0u);
        v.y = raw_kv_ld(raw_kv, 1, b + 1u);
        v.z = raw_kv_ld(raw_kv, 1, b + 2u);
        v.w = raw_kv_ld(raw_kv, 1, b + 3u);
        return v;
    }
    return ((const float4 *)(raw_kv + row_base))[c4];
}

__device__ static float pulsar_attn_e4m3_value(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float pulsar_attn_fp8_kv_dequant(uint8_t byte, float scale) {
    int idx = byte & 0x7f;
    float val = pulsar_attn_e4m3_value(idx);
    return (byte & 0x80) ? (-val * scale) : (val * scale);
}

/* comp_kv_pack: per-call flag — the comp cache operand is PULSAR_ATTN_PACK rows
 * (see PULSAR_ATTN_PACK_* in pulsar_cuda_internal.h): [n_nope e4m3][n_nope/64 E8M0]
 * [pad][n_rot f32 rope].  Nope dims decode e4m3_value * 2^(e8-127) — exactly
 * the fp8_kv_quantize roundtrip value the f32 cache holds — and rope dims read
 * f32 directly, so scores/outputs are bit-identical to the f32 comp cache.
 * The 2^(e8-127) scale is built as a float exponent; the pack amax floor
 * (1e-4 -> e8 >= 105) rules out byte 0. */
/* e4m3 byte * scale by pure bit math — bit-identical to
 * pulsar_attn_e4m3_value(b & 0x7f) * scale with the sign applied (normals become the
 * exact float (1 + mant/8)*2^(exp-7) built directly from its bit pattern;
 * subnormals use the same mant*2^-9 product; scale is an exact power of two,
 * and (-v)*s == -(v*s) in IEEE), but with no exp2f in the inner loops. */
__device__ static inline float attn_pack_e4m3(uint32_t b, float scale) {
    const uint32_t e = (b >> 3) & 15u;
    const uint32_t m = b & 7u;
    const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                      : (float)m * 0.001953125f;
    const float sv = v * scale;
    return (b & 0x80u) ? -sv : sv;
}

__device__ static inline float attn_comp_pack_ld(const float *comp_kv, uint64_t row, uint32_t d, uint32_t head_dim) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *r = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    if (d < n_nope) {
        const float scale = __uint_as_float((uint32_t)r[n_nope + (d / PULSAR_FP8_KV_BLOCK)] << 23);
        return attn_pack_e4m3(r[d], scale);
    }
    return ((const float *)(r + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim)))[d - n_nope];
}

/* Packed-row dot walked d = 0..head_dim-1 by one thread: per-64-block scale
 * hoisted, e4m3 bytes fetched four at a time as one uint32 (rows are 4-byte
 * aligned: 712-byte stride).  Accumulation order is identical to the scalar
 * d-ascending loop, so the result is bit-identical. */
__device__ static inline float attn_pack_dot_full(const float *qh, const float *comp_kv, uint64_t row, uint32_t head_dim, float dot) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *pr = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *psc = pr + n_nope;
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        const float scale = __uint_as_float((uint32_t)psc[off / PULSAR_FP8_KV_BLOCK] << 23);
        const uint32_t *pw = (const uint32_t *)(pr + off);
        for (uint32_t i = 0; i < PULSAR_FP8_KV_BLOCK / 4u; i++) {
            const uint32_t w = pw[i];
            const uint32_t d = off + i * 4u;
            dot += qh[d + 0u] * attn_pack_e4m3(w & 0xffu, scale);
            dot += qh[d + 1u] * attn_pack_e4m3((w >> 8) & 0xffu, scale);
            dot += qh[d + 2u] * attn_pack_e4m3((w >> 16) & 0xffu, scale);
            dot += qh[d + 3u] * attn_pack_e4m3(w >> 24, scale);
        }
    }
    const float *rope = (const float *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
    for (uint32_t d = 0; d < PULSAR_ATTN_PACK_NROT; d++) dot += qh[n_nope + d] * rope[d];
    return dot;
}

/* Packed-row dot walked d = lane, lane+8, ... by one thread (8-lane strided
 * kernels): per-64-block scale hoisted (8 dims per block per thread share it).
 * Same d-ascending visit order as the plain strided loop — bit-identical. */
__device__ static inline float attn_pack_dot_lane8(const float *qh, const float *comp_kv, uint64_t row, uint32_t lane, uint32_t head_dim, float dot) {
    const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
    const uint8_t *pr = (const uint8_t *)comp_kv + row * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
    const uint8_t *psc = pr + n_nope;
    for (uint32_t off = 0; off < n_nope; off += PULSAR_FP8_KV_BLOCK) {
        const float scale = __uint_as_float((uint32_t)psc[off / PULSAR_FP8_KV_BLOCK] << 23);
        for (uint32_t d = off + lane; d < off + PULSAR_FP8_KV_BLOCK; d += 8u) {
            dot += qh[d] * attn_pack_e4m3(pr[d], scale);
        }
    }
    const float *rope = (const float *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
    for (uint32_t d = n_nope + lane; d < head_dim; d += 8u) dot += qh[d] * rope[d - n_nope];
    return dot;
}



__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, (uint64_t)(raw_start + r) * head_dim + d);
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv_ld(raw_kv, raw_f16, (uint64_t)(raw_start + r) * head_dim + d) * scores[r];
        }
        oh[d] = acc / denom;
    }
}



__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, (uint64_t)(raw_start + r) * head_dim + d);
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, raw_f16, (uint64_t)(raw_start + r) * head_dim + d) * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}



__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}



__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}



__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim,
        int raw_f16) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv_ld(raw_kv, raw_f16, (uint64_t)r * head_dim + d)
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}



__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}



/* positions/seq_id/comp_cap (all descriptor-aware decode kernels): per-row
 * multi-session banking (design adapted from the MIT-licensed Entrpi/ds4
 * fork, v0.2 c71a49a — see pulsar_bank_slabs in the engine; reimplemented, no
 * code copied).  positions[t] is row t's absolute query position, seq_id[t]
 * its TRUE bank id (never a packed row ordinal); raw window, ring start and
 * the visible compressed count are derived per row from the position because
 * the raw ring is position-indexed (slot = pos % raw_cap per bank) and
 * compression closes one ratio group every `ratio` positions.  Banked rows
 * read the raw ring at seq_id*raw_cap and compressed rows at seq_id*comp_cap
 * offsets; the scalar n_comp becomes a cross-bank superset used ONLY as a
 * clamp/scratch bound, never to address into a specific bank.  The per-row
 * visible count is (qpos+1)/ratio — the SAME rule the engine's classic
 * single-session decode follows, because the engine emits a step's
 * compressed row BEFORE attention (gpu_decode.c: layer_n_comp is
 * incremented before the attention launch reads it), so at an emit step
 * (qpos ≡ ratio-1 mod ratio) the row attends to the compressed row emitted
 * that same step.  DRIVER CONTRACT (banked mode): every bank's compressed
 * rows for the current step — including same-step emits — must be written
 * before the attention launch; the scalar n_comp superset clamp is a safety
 * bound only.  If the clamp ever bites, the row reads fewer rows than
 * single-session would (fail-safe, not garbage) and its output DIVERGES
 * from classic — that is the mid-prefill-bank case the driver must never
 * co-schedule.  positions == NULL && seq_id == NULL degenerates to the
 * classic single-cache scalar path bit-exactly. */
__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t non_causal,
        uint32_t comp_kv_fp8,
        uint32_t comp_kv_pack,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row (stale or sentinel bank id — the host cannot audit
         * a device array): fail-visible.  Zero this row's head output and
         * read nothing; an out-of-pool id would otherwise be a silent wild
         * read across the whole slab. */
        float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) oh[d] = 0.0f;
        return;
    }
    const bool single_all = (n_tokens == 1u && ratio == 0u && positions == NULL);
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base: with split allocations the batched path passes a base-
     * pointer table (comp_bank_ptrs), so bank sid's comp cache is a separate
     * allocation read at LOCAL row (comp_base == 0). NULL table → the classic
     * single-slab base + seq_id*comp_cap addressing, byte-identical. */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const uint32_t comp_row_bytes = comp_kv_fp8 ? PULSAR_FP8_KV_ROWBYTES(head_dim) : head_dim * sizeof(float);
    if (comp_kv_fp8 && n_comp) { (void)comp_row_bytes; }
    __shared__ float scores[PULSAR_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (single_all) {
                raw_count = eff_n_raw > 256u ? 256u : eff_n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = non_causal ? raw_last_pos : (qpos < raw_last_pos ? qpos : raw_last_pos);
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const uint64_t rbase = (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, rbase + d);
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                float dot = 0.0f;
                if (comp_kv_fp8) {
                    const uint8_t *kv_u8 = (const uint8_t *)comp_src + (comp_base + c) * comp_row_bytes;
                    const float *sc = (const float *)(kv_u8 + head_dim);
                    for (uint32_t d = 0; d < head_dim; d++)
                        dot += qh[d] * pulsar_attn_fp8_kv_dequant(kv_u8[d], sc[d >> 6]);
                } else if (comp_kv_pack) {
                    dot = attn_pack_dot_full(qh, comp_src, comp_base + c, head_dim, dot);
                } else {
                    const float *kvrow = comp_src + (comp_base + c) * head_dim;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                }
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                bool have_row = false;
                uint32_t c_idx = 0;
                if (row < raw_count) {
                    have_row = true;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) { have_row = true; c_idx = c; }
                }
                float s = -INFINITY;
                if (have_row) {
                    float dot = 0.0f;
                    if (row < raw_count) {
                        const uint64_t rbase = (uint64_t)raw_rows[row] * head_dim;
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, rbase + d);
                    } else if (comp_kv_fp8) {
                        const uint8_t *kv_u8 = (const uint8_t *)comp_src + (comp_base + c_idx) * comp_row_bytes;
                        const float *sc = (const float *)(kv_u8 + head_dim);
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * pulsar_attn_fp8_kv_dequant(kv_u8[d], sc[d >> 6]);
                    } else if (comp_kv_pack) {
                        dot = attn_pack_dot_lane8(qh, comp_src, comp_base + c_idx, qlane, head_dim, dot);
                    } else {
                        const float *kvrow = comp_src + (comp_base + c_idx) * head_dim;
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    }
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const uint64_t rbase = (uint64_t)raw_rows[r] * head_dim;
            acc0 += raw_kv_ld(raw_kv, raw_f16, rbase + d0) * s;
            acc1 += raw_kv_ld(raw_kv, raw_f16, rbase + d1) * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            if (comp_kv_fp8) {
                const uint8_t *kv_u8 = (const uint8_t *)comp_src + (comp_base + c) * comp_row_bytes;
                const float *sc = (const float *)(kv_u8 + head_dim);
                acc0 += pulsar_attn_fp8_kv_dequant(kv_u8[d0], sc[d0 >> 6]) * s;
                acc1 += pulsar_attn_fp8_kv_dequant(kv_u8[d1], sc[d1 >> 6]) * s;
            } else if (comp_kv_pack) {
                acc0 += attn_comp_pack_ld(comp_src, comp_base + c, d0, head_dim) * s;
                acc1 += attn_comp_pack_ld(comp_src, comp_base + c, d1, head_dim) * s;
            } else {
                const float *kv = comp_src + (comp_base + c) * head_dim;
                acc0 += kv[d0] * s;
                acc1 += kv[d1] * s;
            }
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, raw_f16, (uint64_t)raw_rows[r] * head_dim + d) * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) {
                if (comp_kv_fp8) {
                    const uint8_t *kv_u8 = (const uint8_t *)comp_src + (comp_base + c) * comp_row_bytes;
                    const float *sc = (const float *)(kv_u8 + head_dim);
                    acc += pulsar_attn_fp8_kv_dequant(kv_u8[d], sc[d >> 6]) * scores[raw_count + c];
                } else if (comp_kv_pack) {
                    acc += attn_comp_pack_ld(comp_src, comp_base + c, d, head_dim) * scores[raw_count + c];
                } else {
                    acc += comp_src[(comp_base + c) * head_dim + d] * scores[raw_count + c];
                }
            }
            oh[d] = acc / denom;
        }
    }
}



__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16,
        uint32_t comp_kv_pack,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: see attention_decode_mixed_kernel. */
        float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) oh[d] = 0.0f;
        return;
    }
    /* Descriptor preamble: see attention_decode_mixed_kernel.  comp_rows[]
     * keeps bank-LOCAL compressed ids (top-k ids are per-bank); the bank
     * offset is applied at read time via comp_base. */
    uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base (see attention_decode_mixed_kernel): base-pointer table
     * → separate per-bank allocation at LOCAL row; NULL → classic seq_id*comp_cap. */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    /* Deterministic ordered compaction (was an atomicAdd slot race): the
     * ORDER of comp_rows fixes the float accumulation order of the indexed
     * attention that consumes it, so it must not depend on warp scheduling.
     * Order-preserving parallel compaction: parallel load + Hillis-Steele
     * scan over the validity flags in smem. top_k <= 512, and the add[2]
     * double-buffer needs blockDim.x >= lim/2 (launch is <<<grid, 256>>>). */
    __shared__ int32_t s_topk[512];
    __shared__ uint16_t s_scan[512];
    const uint32_t lim = top_k < 512u ? top_k : 512u;
    for (uint32_t i = threadIdx.x; i < lim; i += blockDim.x) {
        const int32_t c = topk[(uint64_t)t * top_k + i];
        s_topk[i] = c;
        s_scan[i] = (uint16_t)(c >= 0 && (uint32_t)c < visible_comp);
    }
    __syncthreads();
    for (uint32_t stride = 1; stride < lim; stride <<= 1) {
        uint16_t add[2] = {0, 0};
        for (uint32_t i = threadIdx.x, k = 0; i < lim; i += blockDim.x, k++)
            add[k & 1] = i >= stride ? s_scan[i - stride] : (uint16_t)0;
        __syncthreads();
        for (uint32_t i = threadIdx.x, k = 0; i < lim; i += blockDim.x, k++)
            s_scan[i] = (uint16_t)(s_scan[i] + add[k & 1]);
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < lim; i += blockDim.x) {
        const int32_t c = s_topk[i];
        if (c >= 0 && (uint32_t)c < visible_comp)
            comp_rows[s_scan[i] - 1u] = (uint32_t)c;   /* inclusive scan -> slot */
    }
    if (threadIdx.x == 0) comp_count = lim ? s_scan[lim - 1u] : 0u;
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const uint64_t rbase = (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, rbase + d);
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float dot = 0.0f;
                if (row < raw_count) {
                    const uint64_t rbase = (uint64_t)raw_rows[row] * head_dim;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * raw_kv_ld(raw_kv, raw_f16, rbase + d);
                } else if (comp_kv_pack) {
                    dot = attn_pack_dot_lane8(qh, comp_src, comp_base + comp_rows[row - raw_count], qlane, head_dim, dot);
                } else {
                    const float *kvrow = comp_src + (comp_base + comp_rows[row - raw_count]) * head_dim;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const uint64_t rbase = (uint64_t)raw_rows[r] * head_dim;
            acc0 += raw_kv_ld(raw_kv, raw_f16, rbase + d0) * s;
            acc1 += raw_kv_ld(raw_kv, raw_f16, rbase + d1) * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            if (comp_kv_pack) {
                acc0 += attn_comp_pack_ld(comp_src, comp_base + comp_rows[c], d0, head_dim) * s;
                acc1 += attn_comp_pack_ld(comp_src, comp_base + comp_rows[c], d1, head_dim) * s;
            } else {
                const float *kv = comp_src + (comp_base + comp_rows[c]) * head_dim;
                acc0 += kv[d0] * s;
                acc1 += kv[d1] * s;
            }
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv_ld(raw_kv, raw_f16, (uint64_t)raw_rows[r] * head_dim + d) * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) {
                acc += (comp_kv_pack ? attn_comp_pack_ld(comp_src, comp_base + comp_rows[s], d, head_dim)
                                     : comp_src[(comp_base + comp_rows[s]) * head_dim + d]) * scores[raw_count + s];
            }
            oh[d] = acc / denom;
        }
    }
}



__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            kv_shared[off] = sr < raw_count
                ? raw_kv_ld4(raw_kv, raw_f16, (uint64_t)raw_rows[sr] * head_dim, c4)
                : ((const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim))[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            kv_shared[off] = sr < raw_count
                ? raw_kv_ld4(raw_kv, raw_f16, (uint64_t)raw_rows[sr] * head_dim, c4)
                : ((const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim))[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16,
        uint32_t comp_kv_pack,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: zero the head, then all threads return together
         * (no __syncthreads has run yet).  See
         * attention_decode_mixed_heads8_online_kernel. */
        if (valid_head) {
            float *oh = heads + ((uint64_t)t * n_head + head) * head_dim;
            for (uint32_t d = lane; d < head_dim; d += 32u) oh[d] = 0.0f;
        }
        return;
    }

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    /* Descriptor (banked) preamble: per-row qpos, raw ring span, per-bank base
     * derived from positions[t]/seq_id[t], byte-for-byte as
     * attention_decode_mixed_heads8_online_kernel derives them.  NULL
     * descriptors reproduce the classic scalar pos0+t / raw_start / contiguous
     * comp path exactly (raw_base = comp_base = 0, comp_src = comp_kv). */
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (eff_n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            /* Clamp the top-k index to visible_comp: the engine's invariant keeps
             * padding sentinels (UINT32_MAX) out of this path, but a stray value
             * would otherwise be a ~4 GB wild read. Substitute row 0 on violation. */
            uint32_t comp_idx = 0u;
            if (sr >= raw_count) {
                int32_t c = topk[(uint64_t)t * top_k + (sr - raw_count)];
                comp_idx = (c >= 0 && (uint32_t)c < visible_comp) ? (uint32_t)c : 0u;
            }
            if (sr < raw_count) {
                kv_shared[off] = raw_kv_ld4(raw_kv, raw_f16, (uint64_t)raw_rows[sr] * head_dim, c4);
            } else if (comp_kv_pack) {
                /* ATTN_PACK comp row -> f32 float4 in smem, byte-identical to the
                 * pack dequant in attention_decode_mixed_heads8_online_kernel; the
                 * indexed row id is the top-k comp_idx, not sr-raw_count. */
                const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
                const uint8_t *pr = (const uint8_t *)comp_src +
                    (comp_base + (uint64_t)comp_idx) * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
                const uint32_t base = c4 << 2;
                float4 v;
                if (base < n_nope) {
                    const float scale = __uint_as_float((uint32_t)pr[n_nope + (base / PULSAR_FP8_KV_BLOCK)] << 23);
                    const uint32_t w = *(const uint32_t *)(pr + base);
                    v.x = attn_pack_e4m3(w & 0xffu, scale);
                    v.y = attn_pack_e4m3((w >> 8) & 0xffu, scale);
                    v.z = attn_pack_e4m3((w >> 16) & 0xffu, scale);
                    v.w = attn_pack_e4m3(w >> 24, scale);
                } else {
                    const float *rope = (const float *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
                    v.x = rope[base - n_nope + 0u];
                    v.y = rope[base - n_nope + 1u];
                    v.z = rope[base - n_nope + 2u];
                    v.w = rope[base - n_nope + 3u];
                }
                kv_shared[off] = v;
            } else {
                kv_shared[off] = ((const float4 *)(comp_src + (comp_base + (uint64_t)comp_idx) * head_dim))[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            kv_shared[off] = sr < raw_count
                ? raw_kv_ld4(raw_kv, raw_f16, (uint64_t)(raw_start + sr) * head_dim, c4)
                : ((const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim))[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t non_causal,
        uint32_t comp_kv_fp8,
        uint32_t comp_kv_pack,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        int raw_f16,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ comp_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: see attention_decode_mixed_kernel.  All threads
         * return together (no __syncthreads has run yet). */
        if (valid_head) {
            float *oh = heads + ((uint64_t)t * n_head + head) * head_dim;
            for (uint32_t d = lane; d < head_dim; d += 32u) oh[d] = 0.0f;
        }
        return;
    }

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    /* Descriptor preamble: see attention_decode_mixed_kernel. */
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    uint32_t eff_n_raw = n_raw;
    uint32_t eff_raw_start = raw_start;
    uint32_t first_raw_pos;
    if (positions) {
        eff_n_raw = (window != 0u && qpos + 1u > window) ? window : qpos + 1u;
        if (eff_n_raw > raw_cap) eff_n_raw = raw_cap;
        eff_raw_start = (qpos + 1u - eff_n_raw) % raw_cap;
        first_raw_pos = qpos + 1u - eff_n_raw;
    } else {
        first_raw_pos = pos0 + n_tokens - n_raw;
    }
    const uint32_t raw_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
    /* Per-bank comp base (see attention_decode_mixed_kernel). */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *comp_src = comp_bank_ptrs ? (const float *)comp_bank_ptrs[sid_b] : comp_kv;
    const uint64_t comp_base = comp_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    /* Scalar decode-entry convention: n_tokens==1, pos0==0, ratio==0,
     * positions==NULL — the caller means "attend to ALL cached rows" and does
     * NOT supply a real absolute position, so qpos is 0 and the windowing
     * arithmetic below does not apply.  Mirrors attention_decode_mixed_kernel. */
    const bool single_all = (n_tokens == 1u && ratio == 0u && positions == NULL);
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (single_all) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (eff_n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + eff_n_raw - 1u;
            /* single_all MUST be handled before the qpos comparison below.  In
             * that convention first_raw_pos = pos0 + n_tokens - n_raw = 1 - 128,
             * which UNDERFLOWS to 4294967169, so `qpos(0) >= first_raw_pos` is
             * false and raw_count would stay 0 — silently dropping the ENTIRE
             * SWA raw window while comp_count above stayed correct.  That is not
             * a visible failure: it just answers from compressed rows only, and
             * measures as a large FAKE speedup (skipping 128 rows/layer/token)
             * with heavy logit damage underneath. */
            if (single_all) {
                raw_count = eff_n_raw > 256u ? 256u : eff_n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = non_causal ? raw_last_pos : (qpos < raw_last_pos ? qpos : raw_last_pos);
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = raw_base + (eff_raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t comp_row_bytes = comp_kv_fp8 ? PULSAR_FP8_KV_ROWBYTES(head_dim) : head_dim * sizeof(float);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                kv_shared[off] = raw_kv_ld4(raw_kv, raw_f16, (uint64_t)raw_rows[sr] * head_dim, c4);
            } else if (comp_kv_fp8) {
                const uint8_t *kv_u8 = (const uint8_t *)comp_src + (comp_base + (sr - raw_count)) * comp_row_bytes;
                const float *sc = (const float *)(kv_u8 + head_dim);
                uint32_t base = c4 << 2;
                float4 v;
                v.x = pulsar_attn_fp8_kv_dequant(kv_u8[base + 0], sc[(base + 0) >> 6]);
                v.y = pulsar_attn_fp8_kv_dequant(kv_u8[base + 1], sc[(base + 1) >> 6]);
                v.z = pulsar_attn_fp8_kv_dequant(kv_u8[base + 2], sc[(base + 2) >> 6]);
                v.w = pulsar_attn_fp8_kv_dequant(kv_u8[base + 3], sc[(base + 3) >> 6]);
                kv_shared[off] = v;
            } else if (comp_kv_pack) {
                const uint32_t n_nope = head_dim - PULSAR_ATTN_PACK_NROT;
                const uint8_t *pr = (const uint8_t *)comp_src +
                    (comp_base + (sr - raw_count)) * PULSAR_ATTN_PACK_ROWBYTES(head_dim);
                const uint32_t base = c4 << 2;
                float4 v;
                if (base < n_nope) {
                    /* base%4 == 0 and blocks are 64-aligned, so the four dims
                     * share one scale and one aligned uint32 of e4m3 bytes. */
                    const float scale = __uint_as_float((uint32_t)pr[n_nope + (base / PULSAR_FP8_KV_BLOCK)] << 23);
                    const uint32_t w = *(const uint32_t *)(pr + base);
                    v.x = attn_pack_e4m3(w & 0xffu, scale);
                    v.y = attn_pack_e4m3((w >> 8) & 0xffu, scale);
                    v.z = attn_pack_e4m3((w >> 16) & 0xffu, scale);
                    v.w = attn_pack_e4m3(w >> 24, scale);
                } else {
                    const float *rope = (const float *)(pr + n_nope + PULSAR_ATTN_PACK_SCALES_PAD(head_dim));
                    v.x = rope[base - n_nope + 0u];
                    v.y = rope[base - n_nope + 1u];
                    v.z = rope[base - n_nope + 2u];
                    v.w = rope[base - n_nope + 3u];
                }
                kv_shared[off] = v;
            } else {
                const float4 *src = (const float4 *)(comp_src + (comp_base + (sr - raw_count)) * head_dim);
                kv_shared[off] = src[c4];
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}


int pulsar_gpu_attention_decode_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        uint32_t                n_comp,
        const pulsar_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        (comp_kv_pack && (comp_kv_f16 || comp_kv_fp8 || head_dim <= PULSAR_ATTN_PACK_NROT ||
         ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0)) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp *
         (comp_kv_fp8 ? PULSAR_FP8_KV_ROWBYTES(head_dim)
                      : comp_kv_pack ? PULSAR_ATTN_PACK_ROWBYTES(head_dim)
                                     : head_dim * sizeof(float))) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              0,
                                                                              comp_kv_fp8,
                                                                              comp_kv_pack,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim,
                                                                              raw_f16,
                                                                              NULL, NULL, NULL, 0, 1);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "pulsar: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 0, comp_kv_fp8, comp_kv_pack,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim, raw_f16,
                                                 NULL, NULL, NULL, 0, 1);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}




int pulsar_gpu_attention_prefill_raw_heads_tensor(pulsar_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const pulsar_gpu_tensor *q, const pulsar_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim, uint32_t raw_f16) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_window_attn = getenv("PULSAR_CUDA_NO_WINDOW_ATTENTION") != NULL;
    static const int force_window_attn = getenv("PULSAR_CUDA_WINDOW_ATTENTION") != NULL;
    static const int no_cublas_attn = getenv("PULSAR_CUDA_NO_CUBLAS_ATTENTION") != NULL;
    if (n_tokens > 1 && head_dim == 512 &&
        !no_window_attn &&
        (force_window_attn || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim,
                                                                   raw_f16);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        !no_cublas_attn) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        /* cuBLAS consumes f32 operands: in f16 raw mode, expand the raw rows
         * into an f32 staging region first (pure conversion, exact values, so
         * the gemms see the identical matrix the f32 path reads). */
        const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
        const uint64_t kv_offset = raw_f16 ? ((kv_count * sizeof(float) + 255u) & ~255ull) : 0;
        const uint64_t tmp_bytes = kv_offset + out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = (float *)((char *)tmp + kv_offset);
        float *out_tmp = (float *)((char *)tmp + kv_offset + out_offset);
        const float *kv_mat = (const float *)raw_kv->ptr;
        if (raw_f16) {
            attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                    tmp,
                    (const float *)raw_kv->ptr,
                    (const float *)raw_kv->ptr,
                    n_tokens,
                    0,
                    head_dim,
                    1);
            if (!cuda_ok(cudaGetLastError(), "attention raw f16 expand launch")) return 0;
            kv_mat = tmp;
        }
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv_mat,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv_mat,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim, raw_f16);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}


static int attention_decode_batch_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        const pulsar_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                non_causal,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    /* Descriptor (banked) mode: both per-row arrays or neither; the KV
     * operands are whole bank pools, so byte bounds scale by n_banks and the
     * uint32 row ABI (seq*cap + local) must not overflow.  The scalar
     * n_raw/raw_start are IGNORED and NOT validated in this mode (pass 0):
     * the raw span and ring start are per-row derived in the kernels from
     * positions[t].  raw_cap must still be the true per-bank ring capacity
     * (it addresses the ring and sizes the byte bound); scalar n_comp is the
     * cross-bank superset clamp (comp_cap is the per-bank row stride, so it
     * must cover it).  All rejections here are fail-loud: a silent 0 return
     * looks like a generic launch failure to the driver. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (n_comp != 0 && comp_cap < n_comp) ||
         /* Per-row derivation clamps the raw span to `window`, and the
          * kernels' raw_rows scratch holds PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP
          * rows.  A zero (unbounded) or over-cap window would be silently
          * truncated to the OLDEST rows — fail loud instead. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked decode attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    const uint32_t kernel_n_banks = descr ? n_banks : 1u;
    /* comp_kv is the per-bank comp cache. With split allocations + a base-pointer
     * table (increment 2a) each bank is its OWN comp_cap-row allocation, and
     * comp_kv is bank 0's — so the byte bound is per-bank (comp_cap), not
     * n_banks*comp_cap. Without a table (legacy contiguous), it spans all banks. */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < kv_banks * raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        (comp_kv_pack && (comp_kv_f16 || comp_kv_fp8 || head_dim <= PULSAR_ATTN_PACK_NROT ||
         ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0)) ||
        (n_comp && comp_kv->bytes < comp_rows_min *
         (comp_kv_fp8 ? PULSAR_FP8_KV_ROWBYTES(head_dim)
                      : comp_kv_pack ? PULSAR_ATTN_PACK_ROWBYTES(head_dim)
                                     : head_dim * sizeof(float))) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    /* Per-bank comp base-pointer table (descriptor mode only; NULL → the kernel's
     * scalar base + seq_id*comp_cap fallback, bit-identical to the contiguous pool). */
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              non_causal,
                                                                              comp_kv_fp8,
                                                                              comp_kv_pack,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim,
                                                                              raw_f16,
                                                                              positions_ptr,
                                                                              seq_id_ptr,
                                                                              comp_bank_ptrs_ptr,
                                                                              comp_cap,
                                                                              kernel_n_banks);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "pulsar: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_window_attn = getenv("PULSAR_CUDA_NO_WINDOW_ATTENTION") != NULL;
    static const int force_window_attn = getenv("PULSAR_CUDA_WINDOW_ATTENTION") != NULL;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !no_window_attn &&
        (force_window_attn || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   non_causal,
                                                                   comp_kv_fp8,
                                                                   comp_kv_pack,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim,
                                                                   raw_f16,
                                                                   positions_ptr,
                                                                   seq_id_ptr,
                                                                   comp_bank_ptrs_ptr,
                                                                   comp_cap,
                                                                   kernel_n_banks);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, non_causal, comp_kv_fp8, comp_kv_pack, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim, raw_f16,
                                                 positions_ptr, seq_id_ptr, comp_bank_ptrs_ptr, comp_cap, kernel_n_banks);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}



int pulsar_gpu_attention_decode_raw_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        uint32_t                raw_f16,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, 0, 0, NULL, 0, non_causal, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim, raw_f16,
                                      positions, seq_id, NULL /* raw path: no comp */, comp_cap, n_banks);
}



int pulsar_gpu_attention_decode_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        const pulsar_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        uint32_t                raw_f16,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_kv_fp8, comp_kv_pack, comp_mask, use_comp_mask, non_causal,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim, raw_f16,
                                      positions, seq_id, comp_bank_ptrs, comp_cap, n_banks);
}



int pulsar_gpu_attention_indexed_mixed_batch_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        const pulsar_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    /* Descriptor (banked) mode: same contract as attention_decode_batch_launch
     * (scalar n_raw/raw_start ignored and unvalidated, raw_cap must be the true
     * per-bank ring capacity, rejections fail-loud).  NOTE: banked rows are NO
     * LONGER forced onto the generic per-(row,head) kernel — 0e643eb taught the
     * heads8-online fast variant the banked descriptors, and the dispatch below
     * routes banked rows to it (see the rationale at the dispatch site).  That
     * recovered ~5x on ratio-4 banked prefill; do not "restore" the old
     * single-bank-only restriction. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || raw_cap == 0 || ratio == 0 ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         comp_cap < n_comp ||
         /* raw_rows scratch bound — see attention_decode_batch_launch. */
         window == 0u || window > PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP ||
         (uint64_t)n_banks * raw_cap > 4294967296ull ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked indexed attention rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u raw_cap=%u comp_cap=%u n_comp=%u window=%u ratio=%u)\n",
                n_tokens, n_banks, raw_cap, comp_cap, n_comp, window, ratio);
        return 0;
    }
    const uint64_t kv_banks = descr ? n_banks : 1u;
    /* Per-bank split: comp_kv is bank 0's comp_cap-row allocation when the base-
     * pointer table is present (see attention_decode_batch_launch). */
    const uint64_t comp_rows_min = (descr && comp_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    if (comp_kv_f16 || comp_kv_fp8 ||
        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 ||
        (!descr && (n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap)) ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < kv_banks * raw_cap * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        (comp_kv_pack && (head_dim <= PULSAR_ATTN_PACK_NROT ||
         ((head_dim - PULSAR_ATTN_PACK_NROT) % PULSAR_FP8_KV_BLOCK) != 0)) ||
        comp_kv->bytes < comp_rows_min * (comp_kv_pack ? PULSAR_ATTN_PACK_ROWBYTES(head_dim)
                                                       : head_dim * sizeof(float)) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    const void * const *comp_bank_ptrs_ptr =
        (descr && comp_bank_ptrs) ? (const void * const *)comp_bank_ptrs->ptr : NULL;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_indexed_topk_sort = getenv("PULSAR_CUDA_NO_INDEXED_TOPK_SORT") != NULL;
    static const int no_indexed_heads8 = getenv("PULSAR_CUDA_NO_INDEXED_HEADS8") != NULL;
    static const int twopass_requested = getenv("PULSAR_CUDA_INDEXED_TWOPASS") != NULL;
    /* Kill-switch for the single-token (decode) heads8-online route added below;
     * set it to restore the pre-change generic per-(row,head) decode kernel. */
    static const int no_indexed_decode_heads8 =
        getenv("PULSAR_CUDA_NO_INDEXED_DECODE_HEADS8") != NULL;
    /* The sort stays OFF for n_tokens == 1.  It is a pure locality optimization:
     * attention_indexed_mixed_heads8_online_kernel reads topk[] in whatever order
     * it is given, clamps each id against visible_comp, and folds rows through an
     * online softmax that is correct for ANY permutation — there is no dedup, no
     * monotonicity assumption, and no binary search anywhere in it.  At decode the
     * whole 512-row compressed scan is L2-resident, so ascending row addresses buy
     * nothing while the bitonic sort costs ~4.2 us per token. */
    if (n_tokens > 1u && top_k == 512u &&
        !no_indexed_topk_sort) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    /* The heads8-online kernel is banked-aware (it takes the descriptor
     * arrays), so banked rows route here too — recovering the ~5x it holds over
     * the generic per-(row,head) kernel that banked mode was previously forced
     * onto.  The rb4 twopass variant stays single-bank-only, so a banked launch
     * always takes the online branch (never rb4).
     *
     * Scope guard: packed comp routes to the online kernel ONLY in banked mode
     * (descr) or on a single-token (decode) row.  Multi-token classic (non-descr)
     * paths keep their exact prior kernel choice — f32 -> online, pack -> generic.
     *
     * THE n_tokens == 1 CARVE-OUT CHANGES BEHAVIOUR, DELIBERATELY.  Single-token
     * decode used to fall through to attention_indexed_mixed_kernel at
     * grid(1, n_head): 64 blocks, each dequantizing the SAME compressed rows for
     * one head.  Routing it here instead gives grid(1, n_head/16): each block
     * stages a row into shared memory once and reuses it for 16 heads, amortizing
     * the pack dequant instructions and the load latency across the group.  This
     * is a LATENCY / instruction-amortization win, NOT a bandwidth win — the
     * indexer caps the compressed scan at top_k=512 rows (~0.46 MB per layer),
     * which sits entirely in GB10's 24 MiB L2, so the reads the old kernel
     * repeated were already L2 hits.  Do not expect it to scale with context.
     *
     * It is NOT bit-exact against the generic kernel: the online softmax folds
     * rows in a different order, so decode logits drift (top-1 held everywhere
     * sampled; top-10 reorders and a greedy stream diverges within a token or
     * two).  Each build stays perfectly deterministic run to run.  Prefill is
     * untouched — it already took this kernel. */
    if ((n_tokens > 1 || !no_indexed_decode_heads8) && head_dim == 512 && top_k <= 512u &&
        (descr || !comp_kv_pack || n_tokens == 1u) &&
        !no_indexed_heads8) {
        /* rb4 twopass has no pack support, so pack (and banked) always take the
         * online branch. */
        if (descr || comp_kv_pack || !twopass_requested) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim,
                                                                               raw_f16,
                                                                               comp_kv_pack,
                                                                               positions_ptr,
                                                                               seq_id_ptr,
                                                                               comp_bank_ptrs_ptr,
                                                                               comp_cap,
                                                                               descr ? n_banks : 1u);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim,
                                                                 raw_f16);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim,
                                                  raw_f16,
                                                  comp_kv_pack,
                                                  positions_ptr,
                                                  seq_id_ptr,
                                                  comp_bank_ptrs_ptr,
                                                  comp_cap,
                                                  descr ? n_banks : 1u);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}



static int attention_prefill_mixed_launch(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        const pulsar_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * (raw_f16 ? sizeof(__half) : sizeof(float)) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_window_attn = getenv("PULSAR_CUDA_NO_WINDOW_ATTENTION") != NULL;
    static const int force_window_attn = getenv("PULSAR_CUDA_WINDOW_ATTENTION") != NULL;
    static const int no_cublas_attn = getenv("PULSAR_CUDA_NO_CUBLAS_ATTENTION") != NULL;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !no_window_attn &&
        (force_window_attn || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim,
                                                                   raw_f16);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        !no_cublas_attn) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim,
                raw_f16);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim, raw_f16);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}



int pulsar_gpu_attention_prefill_static_mixed_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16) {
    if (comp_kv_f16 || comp_kv_fp8) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim, raw_f16);
}



int pulsar_gpu_attention_prefill_masked_mixed_heads_tensor(
        pulsar_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *raw_kv,
        const pulsar_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        const pulsar_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim, raw_f16);
}

