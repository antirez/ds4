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
