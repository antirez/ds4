/* V4.1-specific operations. Included after the shared CUDA kernels. */
#include "ds4_deepseek41_gpu.h"

static bool dsv41_has_floats(const ds4_gpu_tensor *t, uint64_t count) {
    return t && t->ptr && count <= t->bytes / sizeof(float);
}

__device__ static float dsv41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ static float dsv41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) +
        ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__global__ static void dsv41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = dsv41_bf16(x[i]);
}

__global__ static void dsv41_quantize_kernel(float *x, uint32_t width,
                                            uint32_t rows, uint32_t mode) {
    const uint32_t block = mode == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t group = (uint64_t)blockIdx.x * 8u + threadIdx.x / 32u;
    if (group >= (uint64_t)(width / block) * rows) return;
    const uint64_t i = group * block + lane;
    const float value = lane < block ? dsv41_bf16(x[i]) : 0.0f;
    const float peak = __shfl_sync(0xffffffffu, warp_max_f32(fabsf(value)), 0);
    float scale, quantized;
    if (mode == DS4_V41_FP8_E8M0) {
        scale = dsv41_pow2_ceil(__fmul_rn(fmaxf(peak, 1.0e-4f), 1.0f / 448.0f));
        quantized = dsv4_e4m3fn_dequant_dev(__fdiv_rn(fabsf(value), scale));
    } else {
        scale = mode == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(__fdiv_rn(fmaxf(peak, 0.01171875f), 6.0f)) :
            dsv41_pow2_ceil(__fmul_rn(fmaxf(peak, 0x1.8p-124f), 1.0f / 6.0f));
        quantized = dsv4_e2m1fn_dequant_dev(__fdiv_rn(fabsf(value), scale));
    }
    if (lane < block) x[i] = dsv41_bf16(__fmul_rn(copysignf(quantized, value), scale));
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width,
                                       uint32_t rows, ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) || !dsv41_has_floats(x, count) ||
        (count + 255u) / 256u > INT32_MAX) return 0;
    if (format == DS4_V41_BF16)
        dsv41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, count);
    else
        dsv41_quantize_kernel<<<(unsigned)((count / block + 7u) / 8u), 256, 0, cuda_decode_stream()>>>(
            (float *)x->ptr, width, rows, format);
    return cuda_ok(cudaGetLastError(), "V4.1 activation rounding");
}

/* Keep the legacy hc_expand_kernel's initial multiply and ascending source
 * accumulation. The plain MAC expressions intentionally retain the build's
 * contraction policy; forcing FMA would change --fmad=false builds. One
 * thread reuses the block and residual loads for all four output streams. */
__global__ static void dsv41_hc_expand_bf16_kernel(
        float *out, const float *block, const float *add,
        const float *residual, const float *split, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * 5120u) return;
    const uint32_t d = i % 5120u, row = i / 5120u;
    float b = block[i];
    if (add) b = __fadd_rn(b, add[i]);
    b = dsv41_bf16(b);
    const uint64_t base = (uint64_t)row * 20480u + d;
    const float r0 = residual[base];
    const float r1 = residual[base + 5120u];
    const float r2 = residual[base + 10240u];
    const float r3 = residual[base + 15360u];
    const float *post = split + (uint64_t)row * 24u + 4u;
    const float *comb = post + 4u;
    for (uint32_t h = 0u; h < 4u; h++) {
        float acc = __fmul_rn(b, post[h]);
        acc += comb[h] * r0;
        acc += comb[h + 4u] * r1;
        acc += comb[h + 8u] * r2;
        acc += comb[h + 12u] * r3;
        out[base + (uint64_t)h * 5120u] = dsv41_bf16(acc);
    }
}

extern "C" int ds4_gpu_dsv41_hc_expand_bf16(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *block, const ds4_gpu_tensor *add,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split,
        uint32_t rows) {
    const uint64_t n = (uint64_t)rows * 5120u;
    if (!rows || n > UINT32_MAX || g_quality_mode || g_n_gpus != 1) return 0;
    const ds4_gpu_tensor *tensors[] = {out, block, add ? add : block, residual, split};
    const uint64_t counts[] = {n * 4u, n, n, n * 4u, (uint64_t)rows * 24u};
    for (uint32_t j = 0u; j < 5u; j++) {
        if (!dsv41_has_floats(tensors[j], counts[j]) || tensors[j]->device_id < -1)
            return 0;
        const uintptr_t p = (uintptr_t)tensors[j]->ptr;
        if ((p & 15u) || counts[j] * sizeof(float) > UINTPTR_MAX - p) return 0;
    }
    const uintptr_t dest = (uintptr_t)out->ptr;
    const uint64_t dest_bytes = counts[0] * sizeof(float);
    const int tier = ds4_tensor_device_idx(out);
    if (tier < 0 || tier >= g_n_gpus) return 0;
    for (uint32_t j = 1u; j < 5u; j++) {
        const uintptr_t src = (uintptr_t)tensors[j]->ptr;
        const uint64_t src_bytes = counts[j] * sizeof(float);
        if (ds4_tensor_device_idx(tensors[j]) != tier ||
            (dest <= src ? src - dest < dest_bytes : dest - src < src_bytes)) return 0;
    }
    int device = -1;
    if (!cuda_ok(cudaGetDevice(&device), "V4.1 HC device") ||
        device != g_gpu[tier].device_id) return 0;
    /* No allocation, weight resolution, device changes, or synchronization:
     * tensor lifetimes and stream ordering remain with the caller. */
    dsv41_hc_expand_bf16_kernel<<<(unsigned)((n + 255u) / 256u), 256,
                                 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)block->ptr,
        add ? (const float *)add->ptr : nullptr,
        (const float *)residual->ptr, (const float *)split->ptr, rows);
    return cuda_ok(cudaGetLastError(), "V4.1 HC expand BF16");
}

extern "C" int ds4_gpu_dsv41_shared_join(void) {
    if (!g_dsv41_shared.pending) return 1;
    g_dsv41_shared.pending = false;
    const bool ok = cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(),
        g_dsv41_shared.done, 0), "shared expert join");
    if (!ok && !g_decode_graph_capturing)
        (void)cudaStreamSynchronize(g_dsv41_shared.stream);
    return ok;
}

extern "C" int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp) {
    if (!ds4_gpu_device_is_spark()) return 0;
    if (g_dsv41_shared.active || g_dsv41_shared.pending || !width || !hidden ||
        !dsv41_has_floats(x, width) || !dsv41_has_floats(out, width) ||
        !dsv41_has_floats(gate, hidden) || !dsv41_has_floats(up, hidden) ||
        !dsv41_has_floats(mid, hidden)) return -1;
    const uint64_t blocks = ((uint64_t)std::max(width, hidden) + 31u) / 32u;
    if (blocks * 36u > CUDA_DSV41_SHARED_SCRATCH ||
        !cuda_dsv41_shared_prepare()) return 0;
    const cudaStream_t main = cuda_decode_stream();
    if (!cuda_ok(cudaEventRecord(g_dsv41_shared.ready, main), "shared expert ready") ||
        !cuda_ok(cudaStreamWaitEvent(g_dsv41_shared.stream, g_dsv41_shared.ready, 0),
                 "shared expert input wait")) return -1;
    g_dsv41_shared.active = true;
    const bool ok =
        ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                   gate_offset, width, hidden, x, 1) &&
        ds4_gpu_dsv41_quantize(gate, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                   up_offset, width, hidden, x, 1) &&
        ds4_gpu_dsv41_quantize(up, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_swiglu_tensor(mid, gate, up, hidden, clamp, 1.0f) &&
        ds4_gpu_dsv41_quantize(mid, hidden, 1, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                   down_offset, hidden, width, mid, 1) &&
        ds4_gpu_dsv41_quantize(out, width, 1, DS4_V41_BF16);
    g_dsv41_shared.active = false;
    if (!cuda_ok(cudaEventRecord(g_dsv41_shared.done, g_dsv41_shared.stream),
                 "shared expert done")) {
        if (!g_decode_graph_capturing) (void)cudaStreamSynchronize(g_dsv41_shared.stream);
        return -1;
    }
    g_dsv41_shared.pending = true;
    if (!ok) {
        (void)ds4_gpu_dsv41_shared_join();
        return -1;
    }
    return 1;
}

struct dsv41_rope_args {
    uint32_t width, heads, rows, start, stride, inverse;
    float frequencies[32];
};

__global__ static void dsv41_rope_kernel(float *x, dsv41_rope_args a) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t head = (uint64_t)blockIdx.x * 8u + threadIdx.x / 32u;
    if (head >= (uint64_t)a.heads * a.rows) return;
    const uint32_t row = head / a.heads;
    const float theta = __fmul_rn(float(a.start + row * a.stride), a.frequencies[lane]);
    /* Avoid fast-math range reduction at long absolute positions. */
    const float c = (float)cos((double)theta);
    const float s = (a.inverse ? -1.0f : 1.0f) * (float)sin((double)theta);
    const uint64_t i = head * a.width + a.width - 64u + lane * 2u;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(__fsub_rn(__fmul_rn(re, c), __fmul_rn(im, s)));
    x[i + 1u] = dsv41_bf16(__fadd_rn(__fmul_rn(re, s), __fmul_rn(im, c)));
}

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width,
                                          uint32_t heads, uint32_t rows, uint32_t start,
                                          uint32_t stride, bool compressed, bool inverse) {
    if (width < 64u || !heads || !rows || rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT64_MAX / width ||
        !dsv41_has_floats(x, (uint64_t)width * heads * rows) ||
        ((uint64_t)heads * rows + 7u) / 8u > INT32_MAX) return 0;
    struct frequencies {
        float value[2][32];
        frequencies() {
            for (int kind = 0; kind < 2; kind++) {
                const float base = kind ? 160000.0f : 10000.0f;
                const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
                const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
                for (int i = 0; i < 32; i++) {
                    float f = 1.0f / powf(base, (float)i / 32.0f);
                    if (kind) {
                        const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                        const float smooth = 1.0f - ramp;
                        f = (f / 16.0f) * (1.0f - smooth) + f * smooth;
                    }
                    value[kind][i] = f;
                }
            }
        }
    };
    static const frequencies table;
    dsv41_rope_args args = {width, heads, rows, start, stride, inverse, {0}};
    memcpy(args.frequencies, table.value[compressed ? 1 : 0], sizeof(args.frequencies));
    dsv41_rope_kernel<<<(unsigned)(((uint64_t)heads * rows + 7u) / 8u), 256, 0, cuda_decode_stream()>>>(
        (float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                   uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1u, compressed, inverse);
}

__global__ static void dsv41_engram_kernel(float *residual, const float *kv,
                                           const float *qw, const float *kw,
                                           const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t row = blockIdx.x, head = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    if (mask && !mask[row]) return;
    const uint64_t offset = ((uint64_t)row * 4u + head) * width;
    const uint64_t key = ((uint64_t)row * 5u + head) * width;
    const uint64_t value = ((uint64_t)row * 5u + 4u) * width;
    float h2 = 0, k2 = 0, dot = 0;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i], k = dsv41_bf16(kv[key + i]);
        const uint32_t wi = head * width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (qw[wi] * kw[wi]) * k;
    }
    h2 = __shfl_sync(0xffffffffu, warp_sum_f32(h2), 0);
    k2 = __shfl_sync(0xffffffffu, warp_sum_f32(k2), 0);
    dot = __shfl_sync(0xffffffffu, warp_sum_f32(dot), 0) *
        rsqrtf(h2 / width + eps) * rsqrtf(k2 / width + eps) * rsqrtf(float(width));
    const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot)));
    for (uint32_t i = lane; i < width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] + gate * dsv41_bf16(kv[value + i]));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                         const ds4_gpu_tensor *qw, const ds4_gpu_tensor *kw,
                                         const ds4_gpu_tensor *mask, uint32_t width,
                                         uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || rows > INT32_MAX || !isfinite(eps) || eps <= 0 ||
        count > UINT64_MAX / 5u || !dsv41_has_floats(residual, count * 4u) ||
        !dsv41_has_floats(kv, count * 5u) || !dsv41_has_floats(qw, (uint64_t)width * 4u) ||
        !dsv41_has_floats(kw, (uint64_t)width * 4u) || (mask && mask->bytes < rows)) return 0;
    dsv41_engram_kernel<<<rows, 128, 0, cuda_decode_stream()>>>((float *)residual->ptr,
        (const float *)kv->ptr, (const float *)qw->ptr, (const float *)kw->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void dsv41_pool_kernel(float *out, const float *kv, const float *scores,
                                         const float *prev_kv, const float *prev_scores,
                                         uint32_t width, uint32_t pairs, uint32_t tail) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)width * pairs) return;
    const uint32_t col = i % width;
    const int64_t row = (int64_t)(i / width) * 2 - tail;
    const uint64_t b = (uint64_t)(row + 1) * width + col;
    const float ka = row < 0 ? prev_kv[col] : kv[(uint64_t)row * width + col];
    const float sa = row < 0 ? prev_scores[col] : scores[(uint64_t)row * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak), eb = expf(sb - peak);
    out[i] = dsv41_bf16(__fdiv_rn(__fadd_rn(__fmul_rn(ka, ea), __fmul_rn(kv[b], eb)), ea + eb));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores, ds4_gpu_tensor *prev_kv,
                                   ds4_gpu_tensor *prev_scores, uint32_t width,
                                   uint32_t rows, uint32_t start) {
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || rows > UINT32_MAX - start ||
        !dsv41_has_floats(kv, count) || !dsv41_has_floats(scores, count) ||
        !dsv41_has_floats(prev_kv, width) || !dsv41_has_floats(prev_scores, width) ||
        (pairs && !dsv41_has_floats(out, (uint64_t)width * pairs)) ||
        ((uint64_t)width * pairs + 255u) / 256u > INT32_MAX) return 0;
    if (pairs) {
        dsv41_pool_kernel<<<(unsigned)(((uint64_t)width * pairs + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)prev_kv->ptr, (const float *)prev_scores->ptr, width, pairs, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 pair pooling")) return 0;
    }
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    const uint64_t bytes = (uint64_t)width * sizeof(float);
    return last_even < start ||
        (ds4_gpu_tensor_copy(prev_kv, 0, kv, (last_even - start) * bytes, bytes) &&
         ds4_gpu_tensor_copy(prev_scores, 0, scores, (last_even - start) * bytes, bytes));
}

__global__ static void dsv41_candidates_kernel(float *out, const float *scores,
                                               const float *mask, uint32_t width,
                                               uint32_t rows, uint32_t start, uint32_t ratio) {
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)out_width * rows) return;
    const uint32_t row = i / out_width, col = i % out_width;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (mask) {
        out[i] = col < visible && mask[(uint64_t)row * blocks + col / 8u] == 0.0f ? scores[i] : -INFINITY;
    } else {
        float best = -INFINITY;
        for (uint32_t j = col * 8u; j < min(visible, (col + 1u) * 8u); j++)
            best = fmaxf(best, scores[(uint64_t)row * width + j]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[i] = best;
    }
}

static int dsv41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                            const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
                            uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint64_t count = (uint64_t)(mask ? width : blocks) * rows;
    if (!dsv41_has_floats(scores, (uint64_t)width * rows) || !dsv41_has_floats(out, count) ||
        (mask && !dsv41_has_floats(mask, (uint64_t)blocks * rows)) ||
        (count + 255u) / 256u > INT32_MAX) return 0;
    dsv41_candidates_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)scores->ptr, mask ? (const float *)mask->ptr : NULL,
        width, rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse candidates");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return dsv41_candidates(out, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return mask && dsv41_candidates(scores, scores, mask, width, rows, start, ratio);
}

__global__ static void dsv41_carry_kernel(uint32_t *packed, float *plain,
                                          uint32_t width, uint32_t rows,
                                          uint32_t words, uint32_t format, bool pack) {
    const uint32_t lanes = ((width + 31u) / 32u) * 32u;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = i / lanes, col = i % lanes;
    if (row >= rows) return;
    if (format == DS4_V41_CARRY_BF16) {
        if (col >= width) return;
        uint16_t *p = (uint16_t *)(packed + (uint64_t)row * words);
        if (pack) p[col] = __float_as_uint(plain[(uint64_t)row * width + col]) >> 16u;
        else plain[(uint64_t)row * width + col] = __uint_as_float((uint32_t)p[col] << 16u);
    } else if (pack) {
        const bool allowed = col < width && plain[(uint64_t)row * width + col] == 0.0f;
        const uint32_t bits = __ballot_sync(0xffffffffu, allowed);
        if (!(col & 31u)) packed[(uint64_t)row * words + col / 32u] = bits;
    } else if (col < width) {
        plain[(uint64_t)row * width + col] = packed[(uint64_t)row * words + col / 32u] &
            (1u << (col & 31u)) ? 0.0f : -INFINITY;
    }
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t offset,
                                        ds4_gpu_tensor *plain, uint32_t width,
                                        uint32_t rows, uint32_t format, bool pack) {
    if (!width || width > UINT32_MAX - 31u || !rows || rows > UINT32_MAX - offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1u) / 2u : (width + 31u) / 32u;
    const uint64_t count = (uint64_t)((width + 31u) / 32u) * 32u * rows;
    if (!dsv41_has_floats(packed, (uint64_t)(offset + rows) * words) ||
        !dsv41_has_floats(plain, (uint64_t)rows * width) || (count + 255u) / 256u > INT32_MAX) return 0;
    dsv41_carry_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (uint32_t *)packed->ptr + (uint64_t)offset * words, (float *)plain->ptr,
        width, rows, words, format, pack);
    return cuda_ok(cudaGetLastError(), "V4.1 compact carry");
}

__global__ static void dsv41_gather_kernel(float *out, const float *source,
                                          const int32_t *ids, uint32_t source_rows) {
    const int32_t id = ids[blockIdx.x];
    for (uint32_t col = threadIdx.x; col < 512u; col += blockDim.x)
        out[(uint64_t)blockIdx.x * 512u + col] = id >= 0 && (uint32_t)id < source_rows ?
            source[(uint64_t)id * 512u + col] : NAN;
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u || selected_rows > source_rows ||
        !dsv41_has_floats(source, (uint64_t)source_rows * 512u) ||
        !dsv41_has_floats(out, (uint64_t)selected_rows * 512u) || !dsv41_has_floats(ids, selected_rows)) return 0;
    dsv41_gather_kernel<<<selected_rows, 256, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)source->ptr, (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

/* A warp owns one score. Preserve the 128-lane reduction tree, but keep
 * all partials in registers instead of synchronizing a block per head. */
__global__ static void dsv41_index_scores_kernel(float *scores, const float *q,
        const float *weights, const float *keys, uint32_t width,
        uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t row = blockIdx.y, lane = threadIdx.x & 31u;
    if (key >= width) return;
    if (key >= (start + row + 1u) / ratio) {
        if (!lane) scores[(uint64_t)row * width + key] = -INFINITY;
        return;
    }
    const float *k = keys + (uint64_t)key * 128u + lane;
    const float k0 = k[0], k1 = k[32], k2 = k[64], k3 = k[96];
    float total = 0;
    for (uint32_t h = 0; h < 32u; h++) {
        const float *x = q + ((uint64_t)row * 32u + h) * 128u + lane;
        const float a = __fadd_rn(__fmul_rn(x[0], k0), __fmul_rn(x[64], k2));
        const float b = __fadd_rn(__fmul_rn(x[32], k1), __fmul_rn(x[96], k3));
        float dot = __fadd_rn(a, b);
        for (uint32_t stride = 16u; stride; stride >>= 1u)
            dot += __shfl_down_sync(0xffffffffu, dot, stride);
        total += fmaxf(dot, 0.0f) * weights[(uint64_t)row * 32u + h];
    }
    if (!lane) scores[(uint64_t)row * width + key] = total * (1.0f / 64.0f);
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                                   const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                                   const ds4_gpu_tensor *keys, uint32_t source_rows,
                                                   uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || source_rows > INT32_MAX || !rows ||
        rows > 65535u || rows > UINT32_MAX - start || (start + rows) / ratio > source_rows ||
        !dsv41_has_floats(scores, (uint64_t)source_rows * rows) ||
        !dsv41_has_floats(q, (uint64_t)rows * 32u * 128u) ||
        !dsv41_has_floats(weights, (uint64_t)rows * 32u) ||
        !dsv41_has_floats(keys, (uint64_t)source_rows * 128u)) return 0;
    /* FP4 with power-of-two scales can exceed FP16's exponent range.
     * Do not pass these already-quantized values through the GLM FP16 tile. */
    dsv41_index_scores_kernel<<<dim3((source_rows + 7u) / 8u, rows), 256, 0, cuda_decode_stream()>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const float *)keys->ptr, source_rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 index scores");
}

extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }
extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                          const ds4_gpu_tensor *, uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                   const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                   const ds4_gpu_tensor *, uint32_t, uint32_t,
                                                   uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                uint32_t width, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        (start + 1u) / ratio < 1024u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 512u) ||
        !dsv41_has_floats(scores, (uint64_t)rows * width)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        ds4_gpu_tensor src = *scores, dst = *selected;
        src.ptr = (float *)src.ptr + (uint64_t)row * width;
        src.bytes = (uint64_t)visible * sizeof(float);
        dst.ptr = (uint32_t *)dst.ptr + (uint64_t)row * 512u;
        dst.bytes = 512u * sizeof(uint32_t);
        if (!ds4_gpu_indexer_topk_tensor(&dst, &src, visible, 1u, 512u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map,
                                             uint64_t model_size, uint64_t weight_offset,
                                             uint32_t width, uint32_t outputs, uint32_t rows,
                                             const ds4_gpu_tensor *in) {
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !dsv41_has_floats(in, (uint64_t)width * rows) ||
        !dsv41_has_floats(out, (uint64_t)outputs * rows)) return 0;
    if (rows > 1u && outputs <= 128u && g_cublas_ready && !g_quality_mode &&
        !getenv("DS4_CUDA_SERIAL_F16_MATMUL") && !getenv("DS4_CUDA_NO_F16_CUBLAS_ONE") &&
        !getenv("DS4_CUDA_SERIAL_ROUTER") && !getenv("DS4_CUDA_F16_SMALL_OUT")) {
        if (width > INT32_MAX || outputs > INT32_MAX || weight_offset > model_size ||
            outputs > (model_size - weight_offset) / sizeof(__half) / width) return 0;
        const uint64_t bytes = (uint64_t)width * outputs * sizeof(__half);
        const int tier = ds4_tensor_device_idx(out);
        if (ds4_tensor_device_idx(in) != tier) return 0;
        const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, bytes, tier,
                                               "V4.1 F16 vector batch");
        const uint64_t count = (uint64_t)rows * width;
        __half *x = (__half *)cuda_tmp_alloc_on(tier, count * sizeof(__half), "F16 vector batch inputs");
        if (!w || !x) return 0;
        f32_to_f16_kernel<<<(count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
            x, (const float *)in->ptr, count);
        if (!cuda_ok(cudaGetLastError(), "F16 vector batch convert")) return 0;
        const float alpha = 1.0f, beta = 0.0f;
        /* Small output projections benefit from one conversion launch. Larger
         * GEMVs are faster when input conversion is interleaved with each row.
         * Keep the one-row cuBLAS reduction in both cases. */
        for (uint32_t row = 0; row < rows; row++) {
            float *y = (float *)out->ptr + (uint64_t)row * outputs;
            const __half *xr = x + (uint64_t)row * width;
            const cublasStatus_t status = cublasGemmEx(cuda_cublas_for_tier(tier),
                    CUBLAS_OP_T, CUBLAS_OP_N, outputs, 1, width, &alpha,
                    w, CUDA_R_16F, width, xr, CUDA_R_16F, width,
                    &beta, y, CUDA_R_32F, outputs, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(status, "F16 vector batch")) return 0;
        }
        return 1;
    }
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = *in, y = *out;
        x.ptr = (float *)x.ptr + (uint64_t)row * width;
        x.bytes = (uint64_t)width * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)row * outputs;
        y.bytes = (uint64_t)outputs * sizeof(float);
        if (!ds4_gpu_matmul_f16_tensor(&y, model_map, model_size, weight_offset,
                                       width, outputs, &x, 1u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                                                    const void *model_map, uint64_t model_size,
                                                    uint64_t a, uint64_t b,
                                                    const ds4_gpu_tensor *heads, uint32_t rows) {
    if (!rows || !dsv41_has_floats(out, (uint64_t)rows * 5120u) ||
        !dsv41_has_floats(low, (uint64_t)rows * 8192u) ||
        !dsv41_has_floats(heads, (uint64_t)rows * 32768u)) return 0;
    /* Quantization launches one grid-y entry per row and head group. */
    for (uint32_t first = 0; first < rows; ) {
        const uint32_t count = min(rows - first, 65535u / 8u);
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)first * 32768u;
        x.bytes = (uint64_t)count * 32768u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)first * 8192u;
        y.bytes = (uint64_t)count * 8192u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_rows_exact_tensor(&y, model_map, model_size,
                a, 4096u, 1024u, 8u, 0u, 8u, &x, count)) return 0;
        first += count;
    }
    return ds4_gpu_dsv41_quantize(low, 8192u, rows, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model_map, model_size,
            b, 8192u, 5120u, low, rows);
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t a, uint64_t b,
        const ds4_gpu_tensor *heads, uint32_t rows, uint32_t rank) {
    const uint64_t shard_bytes = UINT64_C(4) * 1024 * (4096 / 32 * 34);
    if (rank > 1 || !rows || a > model_size || 2u * shard_bytes > model_size - a ||
        !dsv41_has_floats(out, (uint64_t)rows * 5120u) ||
        !dsv41_has_floats(low, (uint64_t)rows * 4096u) ||
        !dsv41_has_floats(heads, (uint64_t)rows * 16384u)) return 0;
    for (uint32_t first = 0; first < rows;) {
        const uint32_t count = min(rows - first, 65535u / 4u);
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)first * 16384u;
        x.bytes = (uint64_t)count * 16384u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)first * 4096u;
        y.bytes = (uint64_t)count * 4096u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_rows_exact_tensor(&y, model_map, model_size,
                a + rank * shard_bytes, 4096u, 1024u, 4u, 0u, 4u, &x, count)) return 0;
        first += count;
    }
    return ds4_gpu_dsv41_quantize(low, 4096u, rows, DS4_V41_BF16) &&
        ds4_gpu_matmul_q8_0_kslice_rows_tensor(out, model_map, model_size,
            b, 8192u, 5120u, rank * 4096u, 4096u, low, rows);
}

/* Repeat the native decode Q4 dot unchanged over token/group rows. The
 * full weight-row stride is retained for TP output-B column slices. */
__global__ static void dsv41_output_q4_native_kernel(
        float *out, const char *weights, const cuda_block_q8_K *xq,
        uint32_t outputs, uint32_t blocks, uint32_t full_blocks,
        uint32_t block0, uint32_t groups, bool output_bf16) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= groups * outputs) return;
    const uint32_t token = blockIdx.y;
    const cuda_block_q8_K *xr = xq +
        ((uint64_t)token * groups + row / outputs) * blocks;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(
        weights + (uint64_t)row * full_blocks * sizeof(cuda_block_q4_K)) + block0;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 8u)
        acc += dev_dot_q4_K_q8_K_block(wr + b, xr + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u)
        out[(uint64_t)token * groups * outputs + row] = output_bf16 ? dsv41_bf16(acc) : acc;
}

static int dsv41_output_q4_rows(
        float *out, const char *weights, const float *input,
        void *scratch, uint64_t scratch_bytes, uint32_t outputs,
        uint32_t width, uint32_t full_width, uint32_t k0,
        uint32_t groups, uint32_t rows, bool use_mmq, bool output_bf16) {
    if (use_mmq && width == full_width)
        return ds4_mmq_q4_K_decode_samples(weights, input, out,
            scratch, (size_t)scratch_bytes, (int)outputs, (int)width,
            (int)rows, (int)groups, output_bf16, cuda_decode_stream()) == 0;
    const uint32_t blocks = width / CUDA_QK_K;
    q8_K_quantize_kernel<<<dim3(blocks, rows * groups), 256, 0, cuda_decode_stream()>>>(
        (cuda_block_q8_K *)scratch, input, width, rows * groups);
    if (!cuda_ok(cudaGetLastError(), "V4.1 output Q4 quantize")) return 0;
    dsv41_output_q4_native_kernel<<<dim3(groups * outputs / 32u, rows), 256,
                                    0, cuda_decode_stream()>>>(
        out, weights, (const cuda_block_q8_K *)scratch, outputs, blocks,
        full_width / CUDA_QK_K, k0 / CUDA_QK_K, groups, output_bf16);
    return cuda_ok(cudaGetLastError(), "V4.1 output Q4 rows");
}

static int dsv41_output_q8_rows(
        float *out, const char *weights, const float *input, void *scratch,
        uint32_t outputs, uint32_t width, uint32_t full_width, uint32_t k0,
        uint32_t groups, uint32_t rows) {
    const uint32_t blocks = width / 32u;
    const uint64_t qbytes = (uint64_t)rows * groups * width;
    int8_t *xq = (int8_t *)scratch;
    float *scales = (float *)((char *)scratch + qbytes);
    const dim3 qgrid(blocks, rows * groups);
    if (groups > 1u)
        ds4_cuda_launch_q8_0_group_slice_quantize(
            cuda_q8_quant_warp_reduce_enabled(), qgrid, cuda_decode_stream(),
            xq, scales, input, width, blocks, groups, 0u, groups);
    else
        ds4_cuda_launch_q8_0_quantize(
            cuda_q8_quant_warp_reduce_enabled(), qgrid, cuda_decode_stream(),
            xq, scales, input, width, blocks);
    if (!cuda_ok(cudaGetLastError(), "V4.1 output Q8 quantize")) return 0;
    const dim3 grid(groups * outputs / 8u, rows);
    const unsigned char *w = (const unsigned char *)weights;
    const int dp4a = cuda_q8_use_dp4a();
    if (groups > 1u)
        grouped_q8_0_a_preq_warp8_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            out, w, xq, scales, width, outputs, groups, rows, blocks, dp4a);
    else
        matmul_q8_0_kslice_preq_warp8_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            out, w, xq, scales, width, outputs, full_width / 32u,
            k0 / 32u, blocks, dp4a);
    return cuda_ok(cudaGetLastError(), "V4.1 output Q8 rows");
}

static bool dsv41_output_range(const void *ptr, uint64_t bytes) {
    return ptr && bytes <= UINTPTR_MAX - (uintptr_t)ptr;
}

static bool dsv41_output_overlap(const void *a, uint64_t a_bytes,
                                  const void *b, uint64_t b_bytes) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

extern "C" int ds4_gpu_dsv41_attention_output_typed_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t a, uint64_t b,
        uint32_t a_type, uint32_t b_type, const ds4_gpu_tensor *heads,
        uint32_t rows, uint32_t tp_world, uint32_t tp_rank) {
    if (!rows || (tp_world != 1u && tp_world != 2u) || tp_rank >= tp_world ||
        (a_type != 8u && a_type != 12u) || (b_type != 8u && b_type != 12u) ||
        !dsv41_output_range(model_map, model_size)) return 0;
    const uint32_t groups = 8u / tp_world;
    const uint32_t low_width = groups * 1024u, heads_width = groups * 4096u;
    const uint64_t a_row = a_type == 8u ? 128u * 34u : 16u * sizeof(cuda_block_q4_K);
    const uint64_t b_row = b_type == 8u ? 256u * 34u : 32u * sizeof(cuda_block_q4_K);
    const uint64_t a_bytes = 8192u * a_row, b_bytes = 5120u * b_row;
    const uint64_t heads_bytes = (uint64_t)rows * heads_width * sizeof(float);
    const uint64_t low_bytes = (uint64_t)rows * low_width * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * 5120u * sizeof(float);
    if (a > model_size || a_bytes > model_size - a ||
        b > model_size || b_bytes > model_size - b ||
        !dsv41_has_floats(heads, heads_bytes / sizeof(float)) ||
        !dsv41_has_floats(low, low_bytes / sizeof(float)) ||
        !dsv41_has_floats(out, out_bytes / sizeof(float))) return 0;
    const void *buffers[] = {heads->ptr, low->ptr, out->ptr};
    const uint64_t sizes[] = {heads_bytes, low_bytes, out_bytes};
    const char *a_map = (const char *)model_map + a;
    const char *b_map = (const char *)model_map + b;
    for (uint32_t i = 0u; i < 3u; i++) {
        if (((uintptr_t)buffers[i] & (sizeof(float) - 1u)) ||
            !dsv41_output_range(buffers[i], sizes[i]) ||
            dsv41_output_overlap(buffers[i], sizes[i], a_map, a_bytes) ||
            dsv41_output_overlap(buffers[i], sizes[i], b_map, b_bytes)) return 0;
        for (uint32_t j = 0u; j < i; j++)
            if (dsv41_output_overlap(buffers[i], sizes[i], buffers[j], sizes[j])) return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    int device = -1;
    if (tier < 0 || tier >= g_n_gpus || out->device_id < -1 ||
        low->device_id < -1 || heads->device_id < -1 ||
        ds4_tensor_device_idx(heads) != tier || ds4_tensor_device_idx(low) != tier ||
        !cuda_ok(cudaGetDevice(&device), "V4.1 output device") ||
        device != g_gpu[tier].device_id) return 0;
    /* Typed Q4 batches are prefill work; decode capture uses the scalar API.
     * Reject capture before weight resolution, MMQ init, or scratch growth. */
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if ((a_type == 12u || b_type == 12u) &&
        (!cuda_ok(cudaStreamIsCapturing(cuda_decode_stream(), &capture),
                  "V4.1 output capture status") ||
         capture != cudaStreamCaptureStatusNone)) return 0;
    const uint64_t local_a_bytes = (uint64_t)low_width * a_row;
    const uint64_t local_a = a + (uint64_t)tp_rank * local_a_bytes;
    const char *wa = cuda_resolve_weight_ptr(model_map, local_a, local_a_bytes,
                                            tier, "V4.1 output A");
    const char *wb = cuda_resolve_weight_ptr(model_map, b, b_bytes, tier, "V4.1 output B");
    if (!wa || !wb || ((uintptr_t)wa & (a_type == 12u ? 3u : 1u)) ||
        ((uintptr_t)wb & (b_type == 12u ? 3u : 1u)) ||
        !dsv41_output_range(wa, local_a_bytes) || !dsv41_output_range(wb, b_bytes)) return 0;
    for (uint32_t i = 0u; i < 3u; i++)
        if (dsv41_output_overlap(buffers[i], sizes[i], wa, local_a_bytes) ||
            dsv41_output_overlap(buffers[i], sizes[i], wb, b_bytes)) return 0;

    if (a_type == 8u && b_type == 8u) {
        if (rows > 65535u) return 0;
        return tp_world == 2u ?
            ds4_gpu_dsv41_attention_output_tp_batch(out, low, model_map, model_size,
                a, b, heads, rows, tp_rank) :
            ds4_gpu_dsv41_attention_output_batch(out, low, model_map, model_size,
                a, b, heads, rows);
    }

    /* One bounded arena is reused in stream order. It fits Q8_K, canonical
     * Q8_1, and Q8_0+F32 scales for A or B, including packed TP activations. */
    const uint32_t chunk_rows = min(rows, 64u);
    const uint64_t block_scratch = std::max((uint64_t)sizeof(cuda_block_q8_K), UINT64_C(288));
    const uint64_t scratch_bytes = (uint64_t)chunk_rows * groups * 16u * block_scratch;
    const bool use_mmq = cuda_use_mmq() != 0;
    /* MMQ's lazy initialization visits physical device zero. Restore the
     * validated execution device before touching its scratch or launching. */
    int dispatch_device = -1;
    if (!cuda_ok(cudaGetDevice(&dispatch_device), "V4.1 output MMQ device") ||
        (dispatch_device != device &&
         !cuda_ok(cudaSetDevice(device), "V4.1 output restore device"))) return 0;
    if (!use_mmq && g_cuda_test_q4_mmq_strict) return 0;
    for (uint32_t first = 0u; first < rows;) {
        /* Scalar Q8 B shares this arena; refresh after every nested dispatch
         * so growing its workspace cannot leave the next chunk a stale view. */
        void *scratch = cuda_tmp_alloc_on(tier, scratch_bytes, "V4.1 output batch activations");
        if (!dsv41_output_range(scratch, scratch_bytes) || ((uintptr_t)scratch & 15u) ||
            dsv41_output_overlap(scratch, scratch_bytes, wa, local_a_bytes) ||
            dsv41_output_overlap(scratch, scratch_bytes, wb, b_bytes)) return 0;
        for (uint32_t i = 0u; i < 3u; i++)
            if (dsv41_output_overlap(scratch, scratch_bytes, buffers[i], sizes[i])) return 0;
        const uint32_t count = min(rows - first, chunk_rows);
        const float *x = (const float *)heads->ptr + (uint64_t)first * heads_width;
        float *l = (float *)low->ptr + (uint64_t)first * low_width;
        float *y = (float *)out->ptr + (uint64_t)first * 5120u;
        const int a_ok = a_type == 12u ?
            dsv41_output_q4_rows(l, wa, x, scratch, scratch_bytes,
                1024u, 4096u, 4096u, 0u, groups, count, use_mmq, true) :
            dsv41_output_q8_rows(l, wa, x, scratch,
                1024u, 4096u, 4096u, 0u, groups, count);
        if (!a_ok) return 0;
        /* Q4 rounds only after its complete reduction (after sanitize on
         * MMVQ). The mixed Q8-A path retains the established separate pass. */
        if (a_type == 8u) {
            dsv41_bf16_kernel<<<(uint64_t)count * low_width / 256u, 256,
                                0, cuda_decode_stream()>>>(l, (uint64_t)count * low_width);
            if (!cuda_ok(cudaGetLastError(), "V4.1 output low BF16")) return 0;
        }
        if (b_type == 8u && tp_world == 1u) {
            /* Scalar Q8 B may use resident aligned weights. Keep its exact
             * dispatcher until batching that representation is validated. */
            for (uint32_t row = 0u; row < count; row++) {
                ds4_gpu_tensor input_row = *low, output_row = *out;
                input_row.ptr = l + (uint64_t)row * low_width;
                input_row.bytes = (uint64_t)low_width * sizeof(float);
                output_row.ptr = y + (uint64_t)row * 5120u;
                output_row.bytes = 5120u * sizeof(float);
                if (!ds4_gpu_matmul_q8_0_tensor(&output_row, model_map, model_size,
                        b, low_width, 5120u, &input_row, 1u)) return 0;
            }
            first += count;
            continue;
        }
        const int b_ok = b_type == 12u ?
            dsv41_output_q4_rows(y, wb, l, scratch, scratch_bytes,
                5120u, low_width, 8192u, tp_rank * low_width, 1u, count, use_mmq, false) :
            dsv41_output_q8_rows(y, wb, l, scratch,
                5120u, low_width, 8192u, tp_rank * low_width, 1u, count);
        if (!b_ok) return 0;
        first += count;
    }
    /* Output stays F32: TP reduction and its BF16 boundary belong to caller. */
    return 1;
}
