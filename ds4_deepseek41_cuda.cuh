/* V4.1-specific operations. Included after the shared CUDA kernels. */
#include "ds4_deepseek41_gpu.h"

#ifdef __HIP_PLATFORM_AMD__
static cudaStream_t cuda_decode_stream(void) { return nullptr; }
#endif

__device__ static float dsv41_broadcast(float value) {
#ifdef __HIP_PLATFORM_AMD__
    return __shfl(value, 0, 32);
#else
    return __shfl_sync(0xffffffffu, value, 0);
#endif
}

static bool dsv41_has_floats(const ds4_gpu_tensor *t, uint64_t count) {
    return t && t->ptr && count <= t->bytes / sizeof(float);
}

__device__ static float dsv41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

#ifndef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!map || !width || !rows || !dsv41_has_floats(out, count) ||
        !dsv41_has_floats(x, count) || offset > size ||
        (uint64_t)width * 4u > size - offset) return 0;
    const float *w = (const float *)cuda_resolve_weight_ptr(map, offset,
        (uint64_t)width * 4u, ds4_tensor_device_idx(out), "V4.1 norm");
    if (!w) return 0;
    rms_norm_weight_kernel<true><<<rows, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, w, width, rows, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 norm");
}

extern "C" int ds4_gpu_dsv41_hc_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows,
        uint32_t weight_stride) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || (weight_stride != 4 && weight_stride != 24) ||
        (count + 255u) / 256u > INT32_MAX || !dsv41_has_floats(out, count) ||
        !dsv41_has_floats(x, count * 4u) ||
        !dsv41_has_floats(weights, (uint64_t)(rows - 1u) * weight_stride + 4u)) return 0;
    hc_weighted_sum_kernel<true><<<(count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, (const float *)weights->ptr,
        width, 4u, rows, weight_stride);
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 HC sum");
}

extern "C" int ds4_gpu_dsv41_hc_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *block,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split,
        uint32_t width, uint32_t rows) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || count > (uint64_t)INT32_MAX * 64u ||
        !dsv41_has_floats(out, count * 4u) || !dsv41_has_floats(residual, count * 4u) ||
        !dsv41_has_floats(block, count) || !dsv41_has_floats(split, (uint64_t)rows * 24u)) return 0;
    const float *sp = (const float *)split->ptr;
    hc_expand_kernel<true><<<(count * 4u + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)block->ptr, NULL, NULL,
        (const float *)residual->ptr, sp + 4u, sp + 8u,
        width, 4u, rows, 24u, 24u, 0, 0);
    return cuda_ok(cudaGetLastError(), "V4.1 BF16 HC expansion");
}
#endif

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
    const float peak = dsv41_broadcast(warp_max_f32(fabsf(value)));
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

#ifndef __HIP_PLATFORM_AMD__
extern "C" int ds4_gpu_dsv41_shared_join(void) {
    if (!g_dsv41_shared.pending) return 1;
    g_dsv41_shared.pending = false;
    const bool ok = cuda_ok(cudaStreamWaitEvent(cuda_decode_stream(),
        g_dsv41_shared.done, 0), "shared expert join");
    if (!ok && !g_decode_graph_capturing)
        (void)cudaStreamSynchronize(g_dsv41_shared.stream);
    return ok;
}

__global__ static void dsv41_shared_gate_up_kernel(
        float *gate, float *up, float *mid,
        const unsigned char *wg, const unsigned char *wu,
        const int8_t *xq, const float *xs,
        uint32_t blocks, uint32_t hidden, float clamp, int dp4a, const int *enabled) {
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= hidden) return;
    if (enabled && !*enabled) {
        if (lane == 0) mid[row] = 0.0f;
        return;
    }
    float g = 0, u = 0;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint64_t off = ((uint64_t)row * blocks + b) * 34u;
        const int8_t *xb = xq + b * 32u;
        const int dg = dot_i8_block((const int8_t *)(wg + off + 2u), xb, 32, dp4a);
        const int du = dot_i8_block((const int8_t *)(wu + off + 2u), xb, 32, dp4a);
        g += __half2float(*(const __half *)(wg + off)) * xs[b] * (float)dg;
        u += __half2float(*(const __half *)(wu + off)) * xs[b] * (float)du;
    }
    g = dsv41_bf16(warp_sum_f32(g));
    u = dsv41_bf16(warp_sum_f32(u));
    if (lane == 0) {
        gate[row] = g;
        up[row] = u;
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fminf(fmaxf(u, -clamp), clamp);
        }
        mid[row] = dsv41_bf16((g / (1.0f + expf(-g))) * u);
    }
}

__global__ static void dsv41_shared_down_kernel(float *out, const unsigned char *w,
        const int8_t *xq, const float *xs, uint32_t blocks, uint32_t width, int dp4a,
        const int *enabled) {
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= width) return;
    if (enabled && !*enabled) {
        if (lane == 0) out[row] = 0.0f;
        return;
    }
    float sum = 0;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint64_t off = ((uint64_t)row * blocks + b) * 34u;
        const int dot = dot_i8_block((const int8_t *)(w + off + 2u), xq + b * 32u, 32, dp4a);
        sum += __half2float(*(const __half *)(w + off)) * xs[b] * (float)dot;
    }
    sum = warp_sum_f32(sum);
    if (lane == 0) out[row] = dsv41_bf16(sum);
}

static bool dsv41_shared_fused(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t gate_off, uint64_t up_off,
        uint64_t down_off, uint32_t width, uint32_t hidden, float clamp,
        const int *enabled) {
    const uint64_t gu_bytes = (uint64_t)hidden * (width / 32u) * 34u;
    const uint64_t down_bytes = (uint64_t)width * (hidden / 32u) * 34u;
    if (!map || gate_off > size || gu_bytes > size - gate_off ||
        up_off > size || gu_bytes > size - up_off ||
        down_off > size || down_bytes > size - down_off) return false;
    const auto *wg = (const unsigned char *)cuda_resolve_weight_ptr(map, gate_off, gu_bytes, 0, "shared gate");
    const auto *wu = (const unsigned char *)cuda_resolve_weight_ptr(map, up_off, gu_bytes, 0, "shared up");
    const auto *wd = (const unsigned char *)cuda_resolve_weight_ptr(map, down_off, down_bytes, 0, "shared down");
    if (!wg || !wu || !wd) return false;
    auto *xq = (int8_t *)g_dsv41_shared.scratch;
    auto *xs = (float *)((char *)g_dsv41_shared.scratch + std::max(width, hidden));
    const cudaStream_t stream = cuda_decode_stream();
    const int dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<width / 32u, 32, 0, stream>>>(
        xq, xs, (const float *)x->ptr, width, width / 32u);
    dsv41_shared_gate_up_kernel<<<(hidden + 7u) / 8u, 256, 0, stream>>>(
        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
        wg, wu, xq, xs, width / 32u, hidden, clamp, dp4a, enabled);
    quantize_q8_0_f32_kernel<<<hidden / 32u, 32, 0, stream>>>(
        xq, xs, (const float *)mid->ptr, hidden, hidden / 32u);
    dsv41_shared_down_kernel<<<(width + 7u) / 8u, 256, 0, stream>>>(
        (float *)out->ptr, wd, xq, xs, hidden / 32u, width, dp4a, enabled);
    return cuda_ok(cudaGetLastError(), "V4.1 fused shared expert");
}

__device__ static uint32_t dsv41_shared_owner(const int32_t *selected,
        uint32_t experts, uint32_t active, uint32_t tie) {
    uint32_t left = 0;
    for (uint32_t i = 0; i < active; i++) left += (uint32_t)selected[i] < experts / 2u;
    return left * 2u == active ? tie : left * 2u < active ? 0u : 1u;
}

__global__ static void dsv41_shared_owner_kernel(int *enabled, const int32_t *selected,
        uint32_t experts, uint32_t active, uint32_t rank, uint32_t tie) {
    *enabled = rank == dsv41_shared_owner(selected, experts, active, tie);
}

__global__ static void dsv41_shared_mask_kernel(float *shared, const int32_t *selected,
        uint32_t width, uint32_t experts, uint32_t active, uint32_t rank, uint32_t tie) {
    const uint32_t row = blockIdx.x;
    __shared__ bool enabled;
    if (!threadIdx.x) enabled = rank == dsv41_shared_owner(selected + row * active, experts, active, tie);
    __syncthreads();
    if (!enabled) for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        shared[(uint64_t)row * width + i] = 0.0f;
}

extern "C" int ds4_gpu_dsv41_shared_mask(ds4_gpu_tensor *shared, const ds4_gpu_tensor *selected,
        uint32_t width, uint32_t rows, uint32_t experts, uint32_t active,
        uint32_t rank, uint32_t tie_owner) {
    if (!width || !rows || rows > 8u || !experts || experts % 2u ||
        !active || active > experts || rank > 1u || tie_owner > 1u ||
        !dsv41_has_floats(shared, (uint64_t)width * rows) ||
        !dsv41_has_floats(selected, (uint64_t)active * rows)) return 0;
    dsv41_shared_mask_kernel<<<rows, 256, 0, cuda_decode_stream()>>>(
        (float *)shared->ptr, (const int32_t *)selected->ptr,
        width, experts, active, rank, tie_owner);
    return cuda_ok(cudaGetLastError(), "V4.1 batched shared owner");
}

extern "C" int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp,
        const ds4_v41_shared_owner *owner) {
    if (!ds4_gpu_device_is_spark()) return 0;
    if (g_dsv41_shared.active || g_dsv41_shared.pending || !width || !hidden ||
        !dsv41_has_floats(x, width) || !dsv41_has_floats(out, width) ||
        !dsv41_has_floats(gate, hidden) || !dsv41_has_floats(up, hidden) ||
        !dsv41_has_floats(mid, hidden)) return -1;
    const uint64_t blocks = ((uint64_t)std::max(width, hidden) + 31u) / 32u;
    if (owner && (!owner->experts || owner->experts % 2u || !owner->active ||
        owner->active > owner->experts || owner->rank > 1u || owner->tie_owner > 1u ||
        !dsv41_has_floats(owner->selected, owner->active))) return -1;
    if (blocks * 36u > CUDA_DSV41_SHARED_SCRATCH - sizeof(int) ||
        !cuda_dsv41_shared_prepare()) return 0;
    const cudaStream_t main = cuda_decode_stream();
    const bool fused = width % 32u == 0 && hidden % 32u == 0 &&
        !getenv("DS4_CUDA_DISABLE_V41_SHARED_FUSION");
    if (owner && !fused) return 0;
    int *enabled = owner ? (int *)((char *)g_dsv41_shared.scratch +
                                  CUDA_DSV41_SHARED_SCRATCH - sizeof(int)) : NULL;
    /* Decide before the owned-expert filter mutates selected on the main
     * stream. The ready event publishes this decision to the side stream. */
    if (owner) {
        dsv41_shared_owner_kernel<<<1, 1, 0, main>>>(enabled,
            (const int32_t *)owner->selected->ptr, owner->experts, owner->active,
            owner->rank, owner->tie_owner);
        if (!cuda_ok(cudaGetLastError(), "V4.1 shared expert owner")) return -1;
    }
    if (!cuda_ok(cudaEventRecord(g_dsv41_shared.ready, main), "shared expert ready") ||
        !cuda_ok(cudaStreamWaitEvent(g_dsv41_shared.stream, g_dsv41_shared.ready, 0),
                 "shared expert input wait")) return -1;
    g_dsv41_shared.active = true;
    const bool ok = fused ? dsv41_shared_fused(out, gate, up, mid, x, model_map,
        model_size, gate_offset, up_offset, down_offset, width, hidden, clamp, enabled) :
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

#endif

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
    h2 = dsv41_broadcast(warp_sum_f32(h2));
    k2 = dsv41_broadcast(warp_sum_f32(k2));
    dot = dsv41_broadcast(warp_sum_f32(dot)) *
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
                                               uint32_t rows, uint32_t start, uint32_t ratio,
                                               uint32_t mask_stride) {
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)out_width * rows) return;
    const uint32_t row = i / out_width, col = i % out_width;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (mask) {
        out[i] = col < visible && mask[(uint64_t)row * mask_stride + col / 8u] == 0.0f ? scores[i] : -INFINITY;
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
                            uint32_t start, uint32_t ratio, uint32_t mask_stride) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint64_t count = (uint64_t)(mask ? width : blocks) * rows;
    if (!mask_stride) mask_stride = blocks;
    if (mask_stride < blocks ||
        !dsv41_has_floats(scores, (uint64_t)width * rows) || !dsv41_has_floats(out, count) ||
        (mask && !dsv41_has_floats(mask, (uint64_t)mask_stride * (rows - 1u) + blocks)) ||
        (count + 255u) / 256u > INT32_MAX) return 0;
    dsv41_candidates_kernel<<<(unsigned)((count + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)scores->ptr, mask ? (const float *)mask->ptr : NULL,
        width, rows, start, ratio, mask_stride);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse candidates");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return dsv41_candidates(out, scores, NULL, width, rows, start, ratio, 0);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio,
                                              uint32_t mask_stride) {
    return mask && dsv41_candidates(scores, scores, mask, width, rows, start, ratio, mask_stride);
}

extern "C" int ds4_gpu_dsv41_candidate_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *blocks,
                                                  uint32_t width, uint32_t rows, uint32_t start) {
    if (!rows || start > UINT32_MAX - 8u || rows > UINT32_MAX - 8u - start ||
        (start + rows + 7u) / 8u > width || (start + 8u) / 8u <= 2048u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 2048u) ||
        !dsv41_has_floats(blocks, (uint64_t)rows * width)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 8u) / 8u;
        ds4_gpu_tensor src = *blocks, dst = *selected;
        src.ptr = (float *)src.ptr + (uint64_t)row * width;
        src.bytes = (uint64_t)visible * sizeof(float);
        dst.ptr = (uint32_t *)dst.ptr + (uint64_t)row * 2048u;
        dst.bytes = 2048u * sizeof(uint32_t);
        if (!ds4_gpu_indexer_topk_tensor(&dst, &src, visible, 1u, 2048u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_candidate_mask_batch(ds4_gpu_tensor *mask, const ds4_gpu_tensor *selected,
                                                  uint32_t width, uint32_t rows, uint32_t mask_stride) {
    if (!rows || mask_stride < width ||
        !dsv41_has_floats(selected, (uint64_t)rows * 2048u) ||
        !dsv41_has_floats(mask, (uint64_t)mask_stride * (rows - 1u) + width)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor src = *selected, dst = *mask;
        src.ptr = (uint32_t *)src.ptr + (uint64_t)row * 2048u;
        src.bytes = 2048u * sizeof(uint32_t);
        dst.ptr = (float *)dst.ptr + (uint64_t)row * mask_stride;
        dst.bytes = (uint64_t)width * sizeof(float);
        if (!ds4_gpu_dsv4_topk_mask_tensor(&dst, &src, width, 1u, 2048u)) return 0;
    }
    return 1;
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
#ifdef __HIP_PLATFORM_AMD__
        const uint32_t bits = (uint32_t)(__ballot(allowed) >> ((threadIdx.x % warpSize) / 32u * 32u));
#else
        const uint32_t bits = __ballot_sync(0xffffffffu, allowed);
#endif
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
        dot = warp_sum_f32(dot);
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
extern "C" int ds4_gpu_dsv41_project_pair_q8(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t,
                                             uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
                                             const ds4_gpu_tensor *) { return 0; }
extern "C" int ds4_gpu_dsv41_shared_swiglu(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *,
                                           uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                                           const ds4_gpu_tensor *, float, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_matmul_expand_rows(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t,
                                                uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t,
                                                uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_norm_pair_rows(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t,
                                            ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t,
                                            const void *, uint64_t, float, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_rope_quantize(const ds4_gpu_tensor *, ds4_gpu_tensor *, uint64_t, uint32_t,
                                           uint32_t, bool, ds4_v41_activation_format) { return 0; }
extern "C" int ds4_gpu_dsv41_project_f16_bf16(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t,
                                              uint32_t, const ds4_gpu_tensor *) { return 0; }
extern "C" int ds4_gpu_dsv41_attention_decode(ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
                                              const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
                                              uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                              ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_project_q(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                       const ds4_gpu_tensor *, uint32_t, bool) { return 0; }
extern "C" int ds4_gpu_dsv41_attention_low(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                           uint32_t, ds4_gpu_tensor *, uint32_t, bool) { return 0; }
extern "C" int ds4_gpu_dsv41_hc_block_input_rows(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
                                                 ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                                 const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                                                 uint32_t, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_hc_project(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
                                        uint32_t, uint32_t, float) { return 0; }
extern "C" int ds4_gpu_dsv41_router_rows(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                         const void *, uint64_t, uint64_t, uint32_t, uint32_t, float,
                                         uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_matmul_f32_rows(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t,
                                             const ds4_gpu_tensor *, uint32_t) { return 0; }
extern "C" int ds4_gpu_dsv41_hc_input(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *,
                                      const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t,
                                      uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float,
                                      float) { return 0; }

extern "C" int ds4_gpu_dsv41_indexer_topk_tensor(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                 uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    return ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k);
}

#ifdef __HIP_PLATFORM_AMD__
__global__ static void dsv41_causal_scores_kernel(float *out, const float *scores,
        uint32_t width, uint32_t stride, uint32_t start, uint32_t ratio) {
    const uint32_t row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) out[(uint64_t)row * width + col] =
        col < (start + row + 1u) / ratio ? scores[(uint64_t)row * stride + col] : -INFINITY;
}
#endif

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                                uint32_t width, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        (start + 1u) / ratio <= 512u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 512u) ||
        !dsv41_has_floats(scores, (uint64_t)rows * width)) return 0;
#ifdef __HIP_PLATFORM_AMD__
    if (rows > 1u) {
        if (rows > 65535u) return 0;
        /* The generic sorter already batches rows. Pack their causal prefixes
         * without changing the caller's scores or using its sort scratch. */
        const uint32_t packed_width = (start + rows) / ratio;
        ds4_gpu_tensor *causal = ds4_gpu_tensor_alloc((uint64_t)rows * packed_width * sizeof(float));
        if (!causal) return 0;
        dsv41_causal_scores_kernel<<<dim3((packed_width + 255u) / 256u, rows), 256>>>(
            (float *)causal->ptr, (const float *)scores->ptr, packed_width, width, start, ratio);
        const bool ok = cuda_ok(cudaGetLastError(), "V4.1 causal scores") &&
            ds4_gpu_indexer_topk_tensor(selected, causal, packed_width, rows, 512u);
        ds4_gpu_tensor_free(causal);
        return ok;
    }
#endif
    uint32_t row = 0;
#ifndef __HIP_PLATFORM_AMD__
    /* Batch independent causal rows, splitting only at a sort-width boundary. */
    while (row < rows && (start + row + 1u) / ratio <= 4096u) {
        const uint32_t visible = (start + row + 1u) / ratio;
        const uint32_t sort = visible <= 1024u ? 1024u : visible <= 2048u ? 2048u : 4096u;
        const uint32_t available = (sort + 1u) * ratio - 1u - (start + row);
        const uint32_t count = rows - row < available ? rows - row : available;
        uint32_t *dst = (uint32_t *)selected->ptr + (uint64_t)row * 512u;
        const float *src = (const float *)scores->ptr + (uint64_t)row * width;
        if (sort == 1024u)
            indexer_topk_pow2_kernel<1024, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        else if (sort == 2048u)
            indexer_topk_pow2_kernel<2048, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        else
            indexer_topk_pow2_kernel<4096, true><<<count, 1024>>>(
                dst, src, width, count, 512u, start + row, ratio);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal topk batch")) return 0;
        row += count;
    }
    if (rows - row >= 2u && getenv("DS4_CUDA_NO_TOPK_STREAM") == NULL) {
        indexer_topk_stream512_kernel<true><<<rows - row, 512>>>(
            (uint32_t *)selected->ptr + (uint64_t)row * 512u,
            (const float *)scores->ptr + (uint64_t)row * width,
            width, rows - row, 512u, start + row, ratio);
        return cuda_ok(cudaGetLastError(), "V4.1 wide causal topk batch");
    }
#endif
    for (; row < rows; row++) {
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

__global__ static void dsv41_indexer_all_kernel(int32_t *ids, uint32_t rows,
                                                uint32_t start, uint32_t ratio) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = i / 512u, col = i % 512u;
    if (row >= rows) return;
    const uint32_t visible = (start + row + 1u) / ratio;
    if (col < min(visible, 512u)) ids[i] = (int32_t)col;
}

extern "C" int ds4_gpu_dsv41_indexer_all_batch(ds4_gpu_tensor *selected, uint32_t rows,
                                               uint32_t start, uint32_t ratio) {
    if (!ratio || !rows || rows > UINT32_MAX - start || (start + rows) / ratio > 512u ||
        !dsv41_has_floats(selected, (uint64_t)rows * 512u)) return 0;
    dsv41_indexer_all_kernel<<<(unsigned)(((uint64_t)rows * 512u + 255u) / 256u), 256, 0,
        cuda_decode_stream()>>>((int32_t *)selected->ptr, rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 indexer select all");
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map,
                                             uint64_t model_size, uint64_t weight_offset,
                                             uint32_t width, uint32_t outputs, uint32_t rows,
                                             const ds4_gpu_tensor *in) {
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !dsv41_has_floats(in, (uint64_t)width * rows) ||
        !dsv41_has_floats(out, (uint64_t)outputs * rows)) return 0;
#ifndef __HIP_PLATFORM_AMD__
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
#endif
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
#ifdef __HIP_PLATFORM_AMD__
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = *heads, y = *low;
        x.ptr = (float *)x.ptr + (uint64_t)row * 32768u;
        x.bytes = 32768u * sizeof(float);
        y.ptr = (float *)y.ptr + (uint64_t)row * 8192u;
        y.bytes = 8192u * sizeof(float);
        if (!ds4_gpu_attention_output_low_q8_tensor(&y, model_map, model_size,
                a, 4096u, 1024u, 8u, &x)) return 0;
    }
#else
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
#endif
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

/* Metal folds these bf16 roundings into the producing kernels; here the
 * standalone rounding pass follows (or precedes) the unchanged operator. */
static uint32_t ds4_gpu_dsv41_rows_of(const ds4_gpu_tensor *t, uint32_t width) {
    const uint64_t bytes = ds4_gpu_tensor_bytes(t);
    const uint64_t rows = width ? bytes / ((uint64_t)width * sizeof(float)) : 0;
    return rows ? (uint32_t)rows : 1u;
}

extern "C" int ds4_gpu_rms_norm_weight_bf16_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    return ds4_gpu_rms_norm_weight_tensor(out, x, model_map, model_size, weight_offset, n, eps) &&
           ds4_gpu_dsv41_quantize(out, n, 1, DS4_V41_BF16);
}

extern "C" int ds4_gpu_matmul_q8_0_bf16_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok) &&
           ds4_gpu_dsv41_quantize(out, (uint32_t)out_dim, (uint32_t)n_tok, DS4_V41_BF16);
}

extern "C" int ds4_gpu_matmul_q8_0_bf16io_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_dsv41_quantize(x, (uint32_t)in_dim, (uint32_t)n_tok, DS4_V41_BF16) &&
           ds4_gpu_matmul_q8_0_bf16_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_hc_weighted_sum_bf16_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights,
        uint32_t n_embd, uint32_t n_hc) {
    return ds4_gpu_hc_weighted_sum_tensor(out, residual_hc, weights, n_embd, n_hc) &&
           ds4_gpu_dsv41_quantize(out, n_embd, ds4_gpu_dsv41_rows_of(out, n_embd), DS4_V41_BF16);
}

extern "C" int ds4_gpu_hc_expand_split_bf16_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc) &&
           ds4_gpu_dsv41_quantize(out_hc, n_hc * n_embd, ds4_gpu_dsv41_rows_of(out_hc, n_hc * n_embd), DS4_V41_BF16);
}

/* block_out is scratch owned by the caller: the sum lands in it before the expansion. */
extern "C" int ds4_gpu_hc_expand_split_add_bf16_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    ds4_gpu_tensor *block = (ds4_gpu_tensor *)block_out;
    const uint32_t rows = ds4_gpu_dsv41_rows_of(block_out, n_embd);
    return ds4_gpu_add_tensor(block, block_out, block_add, (uint64_t)n_embd * rows) &&
           ds4_gpu_dsv41_quantize(block, n_embd, rows, DS4_V41_BF16) &&
           ds4_gpu_hc_expand_split_bf16_tensor(out_hc, block_out, residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_dsv41_rope_bf16(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                       uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_quantize(x, width * heads, rows, DS4_V41_BF16) &&
           ds4_gpu_dsv41_rope(x, width, heads, rows, start, compressed, inverse);
}
