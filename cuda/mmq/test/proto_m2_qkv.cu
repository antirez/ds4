/* proto_m2_qkv.cu — M2-Inc2 prototypes: QKV-post chain fusions.
 *
 * Three kill-switch-ready fusions over the decode QKV-post pool, each proven
 * BIT-EXACT here against verbatim copies of the production kernels:
 *
 *   2a  dsv4_qkv_rms_norm_rows + q8_1 emission of the q row (kills the q_b
 *       mmvq quantize prelude, quantize_q8_1 grid=4).
 *   2b  head_rms_norm + rope_tail(q) fused, device-scalars pos source (the
 *       existing fused kernel is dead code because it bakes host pos0 into
 *       captures; this is its capture-safe twin).
 *   2c  rope_tail(kv) + fp8_kv_quantize + store_raw_kv fused into ONE
 *       8-block kernel: rope touches only the 64-float rotary tail, fp8
 *       covers only the 448 nope elems (disjoint), and the production fp8
 *       kernel's 5.9 us is 7 SEQUENTIAL 64-elem groups in a single 64-thread
 *       block — the groups are independent, so they become 7 parallel blocks
 *       (+1 rope/tail block).  Raw-store f16 round-trip preserved.
 *
 * Build: nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_m2_qkv.cu -o proto_qkv
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define N_HEAD   64u
#define HEAD_DIM 512u
#define N_ROT    64u
#define Q_RANK   1024u
#define KV_DIM   512u
#define RAW_CAP  1024u
#define RMS_EPS  1e-6f

/* mirror of the production decode-scalars substrate (only pos0/raw_row read) */
struct ds4_decode_scalars {
    uint32_t pos0, raw_row, raw_start, n_raw, n_comp, emit_phase,
             comp_row, index_row, flags, token;
};

typedef struct { __half2 ds; int8_t qs[32]; } block_q8_1_t;

/* ============ verbatim production device helpers ============ */
__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}
__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

/* ============ verbatim production kernels (references) ============ */
__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out, const float *q, const float *q_w, uint32_t q_n,
        float *kv_out, const float *kv, const float *kv_w, uint32_t kv_n,
        uint32_t rows, float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__global__ static void rope_tail_scalars_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        const struct ds4_decode_scalars * __restrict__ scalars,
        int32_t pos_offset, uint32_t pos_stride, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;
    const uint32_t pos0 = (uint32_t)((int32_t)scalars->pos0 + pos_offset);
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;
    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ static void fp8_kv_quantize_kernel(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        unsigned char * __restrict__ codes_base, float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_stride = (uint64_t)n_nope + (uint64_t)n_rot * sizeof(float);
    const uint64_t scale_stride = (uint64_t)(n_nope / 64u);
    float *xr = x + (uint64_t)row * head_dim;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * row_stride : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * scale_stride : (float *)0;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            if (codes_row) codes_row[off + tid] = 0; /* unused in decode path */
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    (void)row_stride;
}

__global__ static void store_raw_kv_batch_kernel(
        float *raw, const float *kv,
        uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim,
        const int32_t * __restrict__ positions, const int32_t * __restrict__ seq_id,
        const struct ds4_decode_scalars * __restrict__ s_override) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row;
    if (s_override != NULL && n_tokens == 1u) {
        row = s_override->raw_row;
    } else {
        uint32_t pos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t seq_base = seq_id ? (uint32_t)seq_id[t] * raw_cap : 0u;
        row = seq_base + (pos % raw_cap);
    }
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

/* q8_1 reference (vendored quantize_q8_1 semantics, contiguous case) */
__global__ static void ref_quantize_q8_1_kernel(block_q8_1_t *y, const float *x, uint32_t n) {
    const uint32_t i = blockIdx.x * 32u + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
        sum += __shfl_xor_sync(0xffffffffu, sum, off, 32);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? (int8_t)0 : (int8_t)roundf(xi / d);
    y[blockIdx.x].qs[threadIdx.x] = q;
    if (threadIdx.x == 0) y[blockIdx.x].ds = __floats2half2_rn(d, sum);
}

/* ============ 2a: qkv_rms_norm_rows + q8_1(q row) ============ */
__global__ static void dsv4_qkv_rms_norm_rows_q81_kernel(
        float *q_out, const float *q, const float *q_w, uint32_t q_n,
        float *kv_out, const float *kv, const float *kv_w, uint32_t kv_n,
        uint32_t rows, float eps, block_q8_1_t *q81) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
    /* q8_1 emission of the q row (row 0 decode): re-read the just-written
     * orow values (identical bits) in warp-contiguous granules. */
    if (which == 0u && q81 != NULL && row == 0u && (n & 31u) == 0u) {
        __syncthreads();
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t warp = threadIdx.x >> 5u;
        const uint32_t nwarp = blockDim.x >> 5u;
        for (uint32_t qb = warp; qb < n / 32u; qb += nwarp) {
            const float v = orow[qb * 32u + lane];
            float amax = fabsf(v);
            float s = v;
            for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
                s += __shfl_xor_sync(0xffffffffu, s, off, 32);
            }
            const float d1 = amax / 127.0f;
            const int8_t qv = amax == 0.0f ? (int8_t)0 : (int8_t)roundf(v / d1);
            q81[qb].qs[lane] = qv;
            if (lane == 0u) q81[qb].ds = __floats2half2_rn(d1, s);
        }
    }
}

/* ============ 2b: head_rms + rope_tail(q), scalars pos ============ */
__global__ static void head_rms_norm_rope_tail_scalars_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        const struct ds4_decode_scalars * __restrict__ scalars,
        int32_t pos_offset, uint32_t pos_stride, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }
    const uint32_t pos0 = (uint32_t)((int32_t)scalars->pos0 + pos_offset);
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

/* ============ 2c: rope_tail(kv) + fp8 quantize + raw store ============ */
/* Grid: (n_nope/64 + 1) blocks x 64 threads.  Blocks [0, n_nope/64) each own
 * one independent fp8 group (identical 64-wide shared-tree reduce as the
 * production kernel, so scales and rounded values are bit-exact) and store
 * their post-quant region to the raw row.  The last block ropes the rotary
 * tail (32 pairs, verbatim rope_tail_scalars math at n_head=1/t=0) and
 * stores the rotated tail.  Disjoint regions -> no cross-block ordering. */
__global__ static void kv_rope_fp8_store_scalars_kernel(
        float *kv, float *raw, uint32_t raw_cap, uint32_t head_dim, uint32_t n_rot,
        const struct ds4_decode_scalars * __restrict__ scalars,
        int32_t pos_offset, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = scalars->raw_row;
    float *raw_row = raw + (uint64_t)(row % raw_cap) * head_dim;
    if (b < n_nope / 64u) {
        /* fp8 group b: verbatim group math from fp8_kv_quantize_kernel */
        const uint32_t off = b * 64u;
        float v = 0.0f;
        if (off + tid < n_nope) v = kv[off + tid];
        __shared__ float scratch[64];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            const float qv = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            kv[off + tid] = qv;
            raw_row[off + tid] = __half2float(__float2half(qv));
        }
    } else {
        /* rope the rotary tail then store it */
        const uint32_t pos0 = (uint32_t)((int32_t)scalars->pos0 + pos_offset);
        float corr0 = 0.0f, corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            float denom = 2.0f * logf(freq_base);
            corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
            corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
            corr0 = fmaxf(0.0f, corr0);
            corr1 = fminf((float)(n_rot - 1), corr1);
        }
        float *tail = kv + n_nope;
        if (tid < n_rot / 2u) {
            const uint32_t i = tid * 2u;
            float theta_extrap = (float)pos0 * powf(freq_base, -((float)i) / (float)n_rot);
            float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            float c = cosf(theta) * mscale;
            float s = sinf(theta) * mscale;
            if (inverse) s = -s;
            float x0 = tail[i];
            float x1 = tail[i + 1];
            const float r0 = x0 * c - x1 * s;
            const float r1 = x0 * s + x1 * c;
            tail[i] = r0;
            tail[i + 1] = r1;
            raw_row[n_nope + i] = __half2float(__float2half(r0));
            raw_row[n_nope + i + 1] = __half2float(__float2half(r1));
        }
    }
}

/* ============ harness ============ */
static float *dmalloc_f(uint64_t n) { float *p; CK(cudaMalloc(&p, n * sizeof(float))); return p; }
static void fill_rand(float *d, uint64_t n, unsigned seed) {
    float *h = (float *)malloc(n * sizeof(float));
    srand(seed);
    for (uint64_t i = 0; i < n; i++) h[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
    CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}
static int cmp_dev(const char *what, const float *a, const float *b, uint64_t n) {
    float *ha = (float *)malloc(n * sizeof(float)), *hb = (float *)malloc(n * sizeof(float));
    CK(cudaMemcpy(ha, a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb, b, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = memcmp(ha, hb, n * sizeof(float)) != 0;
    if (bad) {
        uint64_t k = 0;
        for (; k < n; k++) if (ha[k] != hb[k]) break;
        printf("  %s: MISMATCH at %llu (%a vs %a)\n", what, (unsigned long long)k, ha[k], hb[k]);
    } else printf("  %s: bit-identical (%llu floats)\n", what, (unsigned long long)n);
    free(ha); free(hb);
    return bad;
}

int main() {
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    int total_bad = 0;

    /* rope parameter sets: plain and yarn-extended */
    struct RP { float fb, fs, ef, af, bf, bs; uint32_t ctxo; const char *name; };
    RP rps[2] = {
        { 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 0u, "plain" },
        { 10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f, 4096u, "yarn" },
    };
    uint32_t poss[3] = { 0u, 999u, 32767u };

    ds4_decode_scalars hs;
    memset(&hs, 0, sizeof(hs));
    ds4_decode_scalars *ds;
    CK(cudaMalloc(&ds, sizeof(hs)));

    /* ---- 2a ---- */
    {
        float *q_in = dmalloc_f(Q_RANK), *kv_in = dmalloc_f(KV_DIM);
        float *qw = dmalloc_f(Q_RANK), *kvw = dmalloc_f(KV_DIM);
        float *qo_r = dmalloc_f(Q_RANK), *kvo_r = dmalloc_f(KV_DIM);
        float *qo_f = dmalloc_f(Q_RANK), *kvo_f = dmalloc_f(KV_DIM);
        block_q8_1_t *q81_r, *q81_f;
        CK(cudaMalloc(&q81_r, (Q_RANK / 32) * sizeof(block_q8_1_t)));
        CK(cudaMalloc(&q81_f, (Q_RANK / 32) * sizeof(block_q8_1_t)));
        fill_rand(q_in, Q_RANK, 11); fill_rand(kv_in, KV_DIM, 12);
        fill_rand(qw, Q_RANK, 13); fill_rand(kvw, KV_DIM, 14);
        dim3 g(1, 2, 1);
        dsv4_qkv_rms_norm_rows_kernel<<<g, 256, 0, st>>>(qo_r, q_in, qw, Q_RANK, kvo_r, kv_in, kvw, KV_DIM, 1, RMS_EPS);
        ref_quantize_q8_1_kernel<<<Q_RANK / 32, 32, 0, st>>>(q81_r, qo_r, Q_RANK);
        dsv4_qkv_rms_norm_rows_q81_kernel<<<g, 256, 0, st>>>(qo_f, q_in, qw, Q_RANK, kvo_f, kv_in, kvw, KV_DIM, 1, RMS_EPS, q81_f);
        CK(cudaStreamSynchronize(st));
        printf("2a qkv_rms + q8_1 emission:\n");
        total_bad += cmp_dev("q_out", qo_r, qo_f, Q_RANK);
        total_bad += cmp_dev("kv_out", kvo_r, kvo_f, KV_DIM);
        char *h1 = (char *)malloc((Q_RANK / 32) * sizeof(block_q8_1_t));
        char *h2 = (char *)malloc((Q_RANK / 32) * sizeof(block_q8_1_t));
        CK(cudaMemcpy(h1, q81_r, (Q_RANK / 32) * sizeof(block_q8_1_t), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2, q81_f, (Q_RANK / 32) * sizeof(block_q8_1_t), cudaMemcpyDeviceToHost));
        int bad = memcmp(h1, h2, (Q_RANK / 32) * sizeof(block_q8_1_t)) != 0;
        printf("  q8_1 codes: %s\n", bad ? "MISMATCH" : "bit-identical");
        total_bad += bad;
        free(h1); free(h2);
    }

    /* ---- 2b ---- */
    for (int r = 0; r < 2; r++) for (int p = 0; p < 3; p++) {
        RP rp = rps[r];
        hs.pos0 = poss[p];
        CK(cudaMemcpy(ds, &hs, sizeof(hs), cudaMemcpyHostToDevice));
        const uint64_t qn = (uint64_t)N_HEAD * HEAD_DIM;
        float *qa = dmalloc_f(qn), *qb = dmalloc_f(qn);
        fill_rand(qa, qn, 100 + r * 10 + p);
        CK(cudaMemcpy(qb, qa, qn * sizeof(float), cudaMemcpyDeviceToDevice));
        head_rms_norm_kernel<<<N_HEAD, 256, 0, st>>>(qa, 1, N_HEAD, HEAD_DIM, RMS_EPS);
        uint32_t pairs = N_HEAD * (N_ROT / 2);
        rope_tail_scalars_kernel<<<(pairs + 255) / 256, 256, 0, st>>>(qa, 1, N_HEAD, HEAD_DIM, N_ROT,
                ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
        head_rms_norm_rope_tail_scalars_kernel<<<N_HEAD, 256, 0, st>>>(qb, 1, N_HEAD, HEAD_DIM, N_ROT,
                ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs, RMS_EPS);
        CK(cudaStreamSynchronize(st));
        printf("2b head_rms+rope (%s, pos=%u):\n", rp.name, poss[p]);
        total_bad += cmp_dev("q", qa, qb, qn);
        CK(cudaFree(qa)); CK(cudaFree(qb));
    }

    /* ---- 2c ---- */
    for (int r = 0; r < 2; r++) for (int p = 0; p < 3; p++) {
        RP rp = rps[r];
        hs.pos0 = poss[p];
        hs.raw_row = poss[p] % RAW_CAP;
        CK(cudaMemcpy(ds, &hs, sizeof(hs), cudaMemcpyHostToDevice));
        float *kva = dmalloc_f(KV_DIM), *kvb = dmalloc_f(KV_DIM);
        float *rawa = dmalloc_f((uint64_t)RAW_CAP * HEAD_DIM), *rawb = dmalloc_f((uint64_t)RAW_CAP * HEAD_DIM);
        CK(cudaMemset(rawa, 0, (uint64_t)RAW_CAP * HEAD_DIM * sizeof(float)));
        CK(cudaMemset(rawb, 0, (uint64_t)RAW_CAP * HEAD_DIM * sizeof(float)));
        fill_rand(kva, KV_DIM, 200 + r * 10 + p);
        CK(cudaMemcpy(kvb, kva, KV_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
        /* reference chain: rope(kv) -> fp8 -> store */
        uint32_t pairs = 1 * (N_ROT / 2);
        rope_tail_scalars_kernel<<<(pairs + 255) / 256, 256, 0, st>>>(kva, 1, 1, HEAD_DIM, N_ROT,
                ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
        fp8_kv_quantize_kernel<<<1, 64, 0, st>>>(kva, 1, HEAD_DIM, N_ROT, NULL, NULL);
        store_raw_kv_batch_kernel<<<(HEAD_DIM + 255) / 256, 256, 0, st>>>(rawa, kva, RAW_CAP, 0, 1, HEAD_DIM, NULL, NULL, ds);
        /* fused */
        kv_rope_fp8_store_scalars_kernel<<<(HEAD_DIM - N_ROT) / 64 + 1, 64, 0, st>>>(kvb, rawb, RAW_CAP, HEAD_DIM, N_ROT,
                ds, 0, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
        CK(cudaStreamSynchronize(st));
        printf("2c kv rope+fp8+store (%s, pos=%u, row=%u):\n", rp.name, poss[p], hs.raw_row);
        total_bad += cmp_dev("kv", kva, kvb, KV_DIM);
        total_bad += cmp_dev("raw row", rawa + (uint64_t)hs.raw_row * HEAD_DIM,
                             rawb + (uint64_t)hs.raw_row * HEAD_DIM, HEAD_DIM);
        CK(cudaFree(kva)); CK(cudaFree(kvb)); CK(cudaFree(rawa)); CK(cudaFree(rawb));
    }

    /* ---- timing: reference chains vs fused, 43-layer sweeps ---- */
    {
        const int REPS = 500, LAYERS = 43;
        hs.pos0 = 1000u; hs.raw_row = 1000u % RAW_CAP;
        CK(cudaMemcpy(ds, &hs, sizeof(hs), cudaMemcpyHostToDevice));
        RP rp = rps[1];
        const uint64_t qn = (uint64_t)N_HEAD * HEAD_DIM;
        float *q = dmalloc_f(qn), *kv = dmalloc_f(KV_DIM);
        float *raw = dmalloc_f((uint64_t)RAW_CAP * HEAD_DIM);
        float *qi = dmalloc_f(Q_RANK), *kvi = dmalloc_f(KV_DIM);
        float *qw = dmalloc_f(Q_RANK), *kvw = dmalloc_f(KV_DIM);
        float *qo = dmalloc_f(Q_RANK), *kvo = dmalloc_f(KV_DIM);
        block_q8_1_t *q81; CK(cudaMalloc(&q81, (Q_RANK / 32) * sizeof(block_q8_1_t)));
        fill_rand(q, qn, 31); fill_rand(kv, KV_DIM, 32);
        fill_rand(qi, Q_RANK, 33); fill_rand(kvi, KV_DIM, 34);
        fill_rand(qw, Q_RANK, 35); fill_rand(kvw, KV_DIM, 36);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        uint32_t pairs_q = N_HEAD * (N_ROT / 2), pairs_kv = N_ROT / 2;
        dim3 g2(1, 2, 1);
        float ms;
        /* reference chain */
        for (int w = 0; w < 2; w++) {
            if (w) CK(cudaEventRecord(e0, st));
            for (int rr = 0; rr < (w ? REPS : 10); rr++) for (int l = 0; l < LAYERS; l++) {
                dsv4_qkv_rms_norm_rows_kernel<<<g2, 256, 0, st>>>(qo, qi, qw, Q_RANK, kvo, kvi, kvw, KV_DIM, 1, RMS_EPS);
                ref_quantize_q8_1_kernel<<<Q_RANK / 32, 32, 0, st>>>(q81, qo, Q_RANK);
                head_rms_norm_kernel<<<N_HEAD, 256, 0, st>>>(q, 1, N_HEAD, HEAD_DIM, RMS_EPS);
                rope_tail_scalars_kernel<<<(pairs_q + 255) / 256, 256, 0, st>>>(q, 1, N_HEAD, HEAD_DIM, N_ROT, ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
                rope_tail_scalars_kernel<<<1, 256, 0, st>>>(kv, 1, 1, HEAD_DIM, N_ROT, ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
                fp8_kv_quantize_kernel<<<1, 64, 0, st>>>(kv, 1, HEAD_DIM, N_ROT, NULL, NULL);
                store_raw_kv_batch_kernel<<<(HEAD_DIM + 255) / 256, 256, 0, st>>>(raw, kv, RAW_CAP, 0, 1, HEAD_DIM, NULL, NULL, ds);
                (void)pairs_kv;
            }
            if (w) { CK(cudaEventRecord(e1, st)); CK(cudaStreamSynchronize(st)); }
            else CK(cudaStreamSynchronize(st));
        }
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("timing reference chain (7 launches): %.2f us/layer (x43 = %.3f ms/step)\n",
               1000.0f * ms / (REPS * LAYERS), ms / REPS);
        /* fused chain */
        for (int w = 0; w < 2; w++) {
            if (w) CK(cudaEventRecord(e0, st));
            for (int rr = 0; rr < (w ? REPS : 10); rr++) for (int l = 0; l < LAYERS; l++) {
                dsv4_qkv_rms_norm_rows_q81_kernel<<<g2, 256, 0, st>>>(qo, qi, qw, Q_RANK, kvo, kvi, kvw, KV_DIM, 1, RMS_EPS, q81);
                head_rms_norm_rope_tail_scalars_kernel<<<N_HEAD, 256, 0, st>>>(q, 1, N_HEAD, HEAD_DIM, N_ROT, ds, 0, 1u, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs, RMS_EPS);
                kv_rope_fp8_store_scalars_kernel<<<(HEAD_DIM - N_ROT) / 64 + 1, 64, 0, st>>>(kv, raw, RAW_CAP, HEAD_DIM, N_ROT, ds, 0, rp.ctxo, 0, rp.fb, rp.fs, rp.ef, rp.af, rp.bf, rp.bs);
            }
            if (w) { CK(cudaEventRecord(e1, st)); CK(cudaStreamSynchronize(st)); }
            else CK(cudaStreamSynchronize(st));
        }
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("timing fused chain (3 launches):   %.2f us/layer (x43 = %.3f ms/step)\n",
               1000.0f * ms / (REPS * LAYERS), ms / REPS);
    }

    printf(total_bad ? "PROTO_M2_QKV FAILED (%d)\n" : "PROTO_M2_QKV_DONE all bit-exact\n", total_bad);
    return total_bad != 0;
}
