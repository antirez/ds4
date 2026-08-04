/* proto_m2_comp.cu — M2-Inc5 prototype: fused compressor pair+store.
 * Replaces the per-compressor-event 5-launch chain
 *   matmul_f16_splitk(kv) + combine + matmul_f16_splitk(sc) + combine
 * + compressor_store
 * with ONE cooperative kernel, at the three decode widths
 *   out 1024 (primary ratio-4), out 512 (primary ratio-2),
 *   out 256 (indexer ratio-4, head_dim 128); in_dim 4096, ksplit 2/4/8.
 *
 * Gate: kv_cur/sc_cur/state_kv/state_score all BIT-IDENTICAL to the ref
 * chain across widths x ape f16/f32 x pos0 sweep; capture==eager; timing.
 *
 * Bit-exactness: phase 1 replicates the ref split-K tile exactly like
 * M2-Inc3 (proto_m2_router.cu), generalized to seg in {2048,1024,512}:
 * a lane's A = seg/256 accumulators hold chunks lane+32a (identical values
 * at identical lanes to ref warps a), each reduced by the verbatim
 * warp_sum_f32; the ref's cross-warp shfl_down tree over [s0..s7,0-padded]
 * imposes the parenthesization
 *   A=2: s0+s1
 *   A=4: (s0+s2)+(s1+s3)
 *   A=8: ((s0+s4)+(s2+s6))+((s1+s5)+(s3+s7))
 * (identical up to x+0.0f==x on exact zeros).  The tail combines partials
 * in kseg order (verbatim combine kernel) and applies the verbatim store
 * body (state copy + ape add) to the just-combined values.
 *
 * Build (GB10):
 *   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_m2_comp.cu -o proto_m2_comp
 */
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define N_IN 4096u

/* ------------------------------------------------------------------ */
/* Vendored production device functions + kernels (ds4_cuda.cu, verbatim) */

struct ds4_decode_scalars {
    uint32_t pos0;
    uint32_t raw_row;
    uint32_t raw_start;
    uint32_t n_raw;
    uint32_t n_comp;
    uint32_t emit_phase;
    uint32_t comp_row;
    uint32_t index_row;
    uint32_t flags;
    uint32_t token;
};

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float block_reduce_sum_256(float v) {
    v = warp_sum_f32(v);
    __shared__ float s[8];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0u) {
        r = (lane < 8u) ? s[lane] : 0.0f;
        r += __shfl_down_sync(0xffffffffu, r, 4);
        r += __shfl_down_sync(0xffffffffu, r, 2);
        r += __shfl_down_sync(0xffffffffu, r, 1);
    }
    return r;
}

__global__ static void matmul_f16_splitk_kernel(
        float *partial,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t ksplit,
        int vec_ok) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t kseg = blockIdx.y;
    if (row >= out_dim) return;
    uint64_t seg = (in_dim + (uint64_t)ksplit - 1u) / (uint64_t)ksplit;
    seg = (seg + 7u) & ~(uint64_t)7u;
    const uint64_t k0 = (uint64_t)kseg * seg;
    if (k0 >= in_dim) {
        if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = 0.0f;
        return;
    }
    uint64_t k1 = k0 + seg;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const uint32_t tid = threadIdx.x;
    const uint32_t nt = blockDim.x;
    float sum = 0.0f;
    if (vec_ok) {
        const uint64_t vend = k0 + ((k1 - k0) & ~(uint64_t)7u);
        for (uint64_t i = k0 + (uint64_t)tid * 8u; i < vend; i += (uint64_t)nt * 8u) {
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                 + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        for (uint64_t i = vend + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    } else {
        for (uint64_t i = k0 + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    }
    sum = block_reduce_sum_256(sum);
    if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = sum;
}

__global__ static void matmul_f16_splitk_combine_kernel(
        float *out,
        const float *partial,
        uint64_t out_dim,
        uint32_t ksplit) {
    const uint64_t row = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (row >= out_dim) return;
    const float *p = partial + row * (uint64_t)ksplit;
    float s = 0.0f;
    for (uint32_t k = 0u; k < ksplit; k++) s += p[k];
    out[row] = s;
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        const struct ds4_decode_scalars * __restrict__ s_override) {
    if (s_override) {
        pos0 = s_override->pos0;
    }
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

/* ------------------------------------------------------------------ */
/* FUSED: pair matmul + combine + store as one cooperative kernel.
 * Template KS = ksplit in {2,4,8}; SEG = 4096/KS; A = SEG/256 accumulators
 * per lane. */

template <uint32_t KS>
__global__ static void comp_pair_store_fused_kernel(
        float *kv_cur,            /* [width] */
        float *sc_cur,            /* [width] */
        float *partials,          /* [2*width*KS] dedicated scratch */
        float *state_kv,
        float *state_score,
        const __half *w_kv,       /* [width][4096] */
        const __half *w_sc,       /* [width][4096] */
        const float *x,           /* [4096] */
        const void *ape_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t width,           /* == coff*head_dim == out_dim */
        uint32_t pos0,
        const struct ds4_decode_scalars * __restrict__ s_override) {
    constexpr uint32_t SEG = N_IN / KS;
    constexpr uint32_t A = SEG / 256u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;

    /* Phase 1: one warp per (matmul m, row, kseg) tile. */
    const uint32_t tiles_per_m = width * KS;
    const uint32_t nwarp = gridDim.x * (blockDim.x >> 5u);
    for (uint32_t t = blockIdx.x * (blockDim.x >> 5u) + warp;
         t < 2u * tiles_per_m; t += nwarp) {
        const uint32_t m = t < tiles_per_m ? 0u : 1u;
        const uint32_t tt = m ? t - tiles_per_m : t;
        const uint32_t row = tt / KS;
        const uint32_t kseg = tt % KS;
        const uint64_t k0 = (uint64_t)kseg * SEG;
        const __half *wr = (m ? w_sc : w_kv) + (uint64_t)row * N_IN;
        float s[A];
        #pragma unroll
        for (uint32_t a = 0; a < A; a++) {
            const uint64_t i = k0 + ((uint64_t)lane + 32u * a) * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            float sum = 0.0f;
            sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                 + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            s[a] = warp_sum_f32(sum);
        }
        /* ref block_reduce_sum_256 cross-warp tree over [s0..s(A-1), 0...]:
         * shfl_down 4,2,1 -> fixed parenthesization (zero-adds are identity). */
        float partial;
        if (KS == 8u) {          /* A == 2 */
            partial = s[0] + s[1 % A];
        } else if (KS == 4u) {   /* A == 4 */
            partial = (s[0] + s[2 % A]) + (s[1 % A] + s[3 % A]);
        } else {                 /* KS == 2, A == 8 */
            partial = ((s[0] + s[4 % A]) + (s[2 % A] + s[6 % A]))
                    + ((s[1 % A] + s[5 % A]) + (s[3 % A] + s[7 % A]));
        }
        if (lane == 0u) partials[(uint64_t)m * tiles_per_m + (uint64_t)row * KS + kseg] = partial;
    }

    cooperative_groups::this_grid().sync();

    /* Phase 2: combine (kseg ascending, verbatim) + verbatim store body. */
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n2 = 2u * width;
    if (gid >= n2) return;
    const uint32_t m = gid < width ? 0u : 1u;
    const uint32_t row = m ? gid - width : gid;
    const float *p = partials + (uint64_t)m * tiles_per_m + (uint64_t)row * KS;
    float s = 0.0f;
    for (uint32_t k = 0u; k < KS; k++) s += p[k];
    uint32_t p0 = s_override ? s_override->pos0 : pos0;
    const uint32_t pos_mod = p0 % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    if (m == 0u) {
        kv_cur[row] = s;
        state_kv[(uint64_t)dst_row * width + row] = s;
    } else {
        sc_cur[row] = s;
        state_score[(uint64_t)dst_row * width + row] =
            s + model_scalar_dev(ape_map, ape_offset, ape_type, (uint64_t)pos_mod * width + row);
    }
}

/* ------------------------------------------------------------------ */
/* Host harness */

struct cfg_t {
    const char *name;
    uint32_t head_dim, ratio, width, ksplit;
};
static const cfg_t CFGS[3] = {
    { "primary-r4 (1024,k2)", 512, 4, 1024, 2 },
    { "primary-r2 (512,k4)",  512, 2, 512,  4 },
    { "index-r4   (256,k8)",  128, 4, 256,  8 },
};

static void fill_rand(float *p, size_t n, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f * scale;
    }
}

struct bufs {
    __half *w_kv, *w_sc;
    float *x, *kv_cur, *sc_cur, *state_kv, *state_score, *part_ref, *part_fu;
    char *ape;             /* f32 or f16 depending on ape_type */
    ds4_decode_scalars *scalars;
};

static void *fused_ptr(uint32_t ks) {
    if (ks == 2u) return (void *)comp_pair_store_fused_kernel<2u>;
    if (ks == 4u) return (void *)comp_pair_store_fused_kernel<4u>;
    return (void *)comp_pair_store_fused_kernel<8u>;
}

static void run_ref(const bufs &b, const cfg_t &c, uint32_t ape_type, uint32_t pos0, int via_scalars) {
    dim3 sg1(c.width, c.ksplit, 1);
    matmul_f16_splitk_kernel<<<sg1, 256>>>(b.part_ref, b.w_kv, b.x, N_IN, c.width, c.ksplit, 1);
    matmul_f16_splitk_combine_kernel<<<(c.width + 255u) / 256u, 256>>>(b.kv_cur, b.part_ref, c.width, c.ksplit);
    matmul_f16_splitk_kernel<<<sg1, 256>>>(b.part_ref, b.w_sc, b.x, N_IN, c.width, c.ksplit, 1);
    matmul_f16_splitk_combine_kernel<<<(c.width + 255u) / 256u, 256>>>(b.sc_cur, b.part_ref, c.width, c.ksplit);
    compressor_store_kernel<<<(c.width + 255u) / 256u, 256>>>(
            b.kv_cur, b.sc_cur, b.state_kv, b.state_score,
            b.ape, 0, ape_type, c.head_dim, c.ratio,
            via_scalars ? 0xdeadbeefu : pos0, 1,
            via_scalars ? b.scalars : NULL);
    CHECK(cudaGetLastError());
}

static int run_fused(const bufs &b, const cfg_t &c, uint32_t ape_type, uint32_t pos0, int via_scalars,
                     unsigned grid_blocks, cudaStream_t stream) {
    float *kv_cur = b.kv_cur, *sc_cur = b.sc_cur, *partials = b.part_fu;
    float *state_kv = b.state_kv, *state_score = b.state_score, *x = b.x;
    const __half *w_kv = b.w_kv, *w_sc = b.w_sc;
    const void *ape = b.ape;
    uint64_t ape_off = 0;
    uint32_t at = ape_type, hd = c.head_dim, ra = c.ratio, wi = c.width;
    uint32_t p0 = via_scalars ? 0xdeadbeefu : pos0;
    const ds4_decode_scalars *sc_ptr = via_scalars ? b.scalars : NULL;
    void *args[] = {
        (void *)&kv_cur, (void *)&sc_cur, (void *)&partials,
        (void *)&state_kv, (void *)&state_score,
        (void *)&w_kv, (void *)&w_sc, (void *)&x,
        (void *)&ape, (void *)&ape_off, (void *)&at,
        (void *)&hd, (void *)&ra, (void *)&wi, (void *)&p0, (void *)&sc_ptr };
    cudaError_t err = cudaLaunchCooperativeKernel(
            fused_ptr(c.ksplit), dim3(grid_blocks, 1, 1), dim3(256, 1, 1), args, 0, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "coop launch failed (%s, %u blk): %s\n", c.name, grid_blocks, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

int main() {
    int dev = 0, coop = 0, n_sm = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
    printf("coop=%d n_sm=%d\n", coop, n_sm);
    if (!coop) return 1;
    unsigned cap[3];
    for (int i = 0; i < 3; i++) {
        int bps = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fused_ptr(CFGS[i].ksplit), 256, 0));
        cap[i] = (unsigned)(bps * n_sm);
        printf("  %s: blocks/sm=%d cap=%u\n", CFGS[i].name, bps, cap[i]);
    }

    const uint32_t WMAX = 1024;
    bufs b = {};
    CHECK(cudaMalloc(&b.w_kv, (size_t)WMAX * N_IN * sizeof(__half)));
    CHECK(cudaMalloc(&b.w_sc, (size_t)WMAX * N_IN * sizeof(__half)));
    CHECK(cudaMalloc(&b.x, N_IN * sizeof(float)));
    CHECK(cudaMalloc(&b.kv_cur, WMAX * sizeof(float)));
    CHECK(cudaMalloc(&b.sc_cur, WMAX * sizeof(float)));
    CHECK(cudaMalloc(&b.state_kv, 8u * WMAX * sizeof(float)));
    CHECK(cudaMalloc(&b.state_score, 8u * WMAX * sizeof(float)));
    CHECK(cudaMalloc(&b.part_ref, (size_t)WMAX * 8u * sizeof(float)));
    CHECK(cudaMalloc(&b.part_fu, 2u * (size_t)WMAX * 8u * sizeof(float)));
    CHECK(cudaMalloc(&b.ape, 4u * WMAX * sizeof(float)));
    CHECK(cudaMalloc(&b.scalars, sizeof(ds4_decode_scalars)));

    float *hbuf = (float *)malloc((size_t)WMAX * N_IN * sizeof(float));
    __half *hh = (__half *)malloc((size_t)WMAX * N_IN * sizeof(__half));
    float *hape = (float *)malloc(4u * WMAX * sizeof(float));
    __half *hape16 = (__half *)malloc(4u * WMAX * sizeof(__half));

    const size_t OUTB = WMAX * sizeof(float);
    const size_t STB = 8u * WMAX * sizeof(float);
    float *ref_out = (float *)malloc(2 * OUTB + 2 * STB);
    float *fu_out = (float *)malloc(2 * OUTB + 2 * STB);

    int total_bad = 0, n_cases = 0;
    const unsigned grids[] = { 48u, 96u, 128u };
    const uint32_t poss[] = { 0u, 1u, 2u, 3u, 7u, 1023u };
    for (int ci = 0; ci < 3; ci++) {
        const cfg_t &c = CFGS[ci];
        for (unsigned seed = 1; seed <= 3; seed++) {
            fill_rand(hbuf, (size_t)c.width * N_IN, seed * 7919u + ci, 0.05f);
            for (size_t i = 0; i < (size_t)c.width * N_IN; i++) hh[i] = __float2half(hbuf[i]);
            CHECK(cudaMemcpy(b.w_kv, hh, (size_t)c.width * N_IN * sizeof(__half), cudaMemcpyHostToDevice));
            fill_rand(hbuf, (size_t)c.width * N_IN, seed * 104729u + ci, 0.05f);
            for (size_t i = 0; i < (size_t)c.width * N_IN; i++) hh[i] = __float2half(hbuf[i]);
            CHECK(cudaMemcpy(b.w_sc, hh, (size_t)c.width * N_IN * sizeof(__half), cudaMemcpyHostToDevice));
            fill_rand(hbuf, N_IN, seed * 31u, 1.0f);
            CHECK(cudaMemcpy(b.x, hbuf, N_IN * sizeof(float), cudaMemcpyHostToDevice));
            fill_rand(hape, (size_t)c.ratio * c.width, seed * 71u, 0.5f);
            for (size_t i = 0; i < (size_t)c.ratio * c.width; i++) hape16[i] = __float2half(hape[i]);
            for (uint32_t ape_type = 0; ape_type <= 1; ape_type++) {
                if (ape_type == 0) CHECK(cudaMemcpy(b.ape, hape, (size_t)c.ratio * c.width * sizeof(float), cudaMemcpyHostToDevice));
                else CHECK(cudaMemcpy(b.ape, hape16, (size_t)c.ratio * c.width * sizeof(__half), cudaMemcpyHostToDevice));
                for (uint32_t pi = 0; pi < 6; pi++) {
                    const uint32_t pos0 = poss[pi];
                    const int via_scalars = (int)(pi & 1u);
                    ds4_decode_scalars sc = {};
                    sc.pos0 = pos0;
                    CHECK(cudaMemcpy(b.scalars, &sc, sizeof(sc), cudaMemcpyHostToDevice));
                    /* ref */
                    CHECK(cudaMemset(b.kv_cur, 0xCB, OUTB));
                    CHECK(cudaMemset(b.sc_cur, 0xCB, OUTB));
                    CHECK(cudaMemset(b.state_kv, 0xCB, STB));
                    CHECK(cudaMemset(b.state_score, 0xCB, STB));
                    run_ref(b, c, ape_type, pos0, via_scalars);
                    CHECK(cudaDeviceSynchronize());
                    CHECK(cudaMemcpy(ref_out, b.kv_cur, OUTB, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(ref_out + WMAX, b.sc_cur, OUTB, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(ref_out + 2 * WMAX, b.state_kv, STB, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(ref_out + 10 * WMAX, b.state_score, STB, cudaMemcpyDeviceToHost));
                    for (unsigned g : grids) {
                        if (g > cap[ci]) continue;
                        CHECK(cudaMemset(b.kv_cur, 0xCB, OUTB));
                        CHECK(cudaMemset(b.sc_cur, 0xCB, OUTB));
                        CHECK(cudaMemset(b.state_kv, 0xCB, STB));
                        CHECK(cudaMemset(b.state_score, 0xCB, STB));
                        if (!run_fused(b, c, ape_type, pos0, via_scalars, g, 0)) return 1;
                        CHECK(cudaDeviceSynchronize());
                        CHECK(cudaMemcpy(fu_out, b.kv_cur, OUTB, cudaMemcpyDeviceToHost));
                        CHECK(cudaMemcpy(fu_out + WMAX, b.sc_cur, OUTB, cudaMemcpyDeviceToHost));
                        CHECK(cudaMemcpy(fu_out + 2 * WMAX, b.state_kv, STB, cudaMemcpyDeviceToHost));
                        CHECK(cudaMemcpy(fu_out + 10 * WMAX, b.state_score, STB, cudaMemcpyDeviceToHost));
                        n_cases++;
                        if (memcmp(ref_out, fu_out, 2 * OUTB + 2 * STB)) {
                            total_bad++;
                            printf("MISMATCH %s seed%u ape%u pos%u g%u\n", c.name, seed, ape_type, pos0, g);
                            for (int i = 0; i < 6; i++)
                                printf("  kv[%d] ref=%.9g fu=%.9g  sc ref=%.9g fu=%.9g\n", i,
                                       ref_out[i], fu_out[i], ref_out[WMAX + i], fu_out[WMAX + i]);
                        }
                    }
                }
            }
        }
    }
    printf("parity: %d/%d cases bit-identical\n", n_cases - total_bad, n_cases);
    if (total_bad) return 1;

    /* capture/replay bit-identity (one config) */
    {
        const cfg_t &c = CFGS[2];
        const unsigned g = cap[2] < 128u ? cap[2] : 128u;
        ds4_decode_scalars sc = {};
        sc.pos0 = 5;
        CHECK(cudaMemcpy(b.scalars, &sc, sizeof(sc), cudaMemcpyHostToDevice));
        if (!run_fused(b, c, 1, 0, 1, g, 0)) return 1;
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(ref_out, b.kv_cur, OUTB, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(ref_out + 2 * WMAX, b.state_score, STB, cudaMemcpyDeviceToHost));
        cudaStream_t cs;
        CHECK(cudaStreamCreate(&cs));
        cudaGraph_t graph;
        cudaGraphExec_t gexec;
        CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        if (!run_fused(b, c, 1, 0, 1, g, cs)) return 1;
        CHECK(cudaStreamEndCapture(cs, &graph));
        CHECK(cudaGraphInstantiate(&gexec, graph, NULL, NULL, 0));
        for (int r = 0; r < 3; r++) {
            CHECK(cudaMemset(b.kv_cur, 0xCB, OUTB));
            CHECK(cudaGraphLaunch(gexec, cs));
            CHECK(cudaStreamSynchronize(cs));
            CHECK(cudaMemcpy(fu_out, b.kv_cur, OUTB, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(fu_out + 2 * WMAX, b.state_score, STB, cudaMemcpyDeviceToHost));
            if (memcmp(ref_out, fu_out, OUTB) || memcmp(ref_out + 2 * WMAX, fu_out + 2 * WMAX, STB)) {
                printf("capture-replay MISMATCH\n");
                return 1;
            }
        }
        printf("capture/replay: bit-identical (3 replays)\n");
        CHECK(cudaGraphExecDestroy(gexec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(cs));
    }

    /* timing: rotating 16-layer weight sets (L2-defeated) per config */
    {
        const int NL = 16, ITERS = 2000, WARM = 200;
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        float ms;
        for (int ci = 0; ci < 3; ci++) {
            const cfg_t &c = CFGS[ci];
            __half *wl[NL][2];
            for (int l = 0; l < NL; l++)
                for (int m = 0; m < 2; m++) {
                    CHECK(cudaMalloc(&wl[l][m], (size_t)c.width * N_IN * sizeof(__half)));
                    CHECK(cudaMemcpy(wl[l][m], m ? b.w_sc : b.w_kv,
                                     (size_t)c.width * N_IN * sizeof(__half), cudaMemcpyDeviceToDevice));
                }
            bufs bl = b;
            for (int it = 0; it < WARM; it++) {
                bl.w_kv = wl[it % NL][0]; bl.w_sc = wl[it % NL][1];
                run_ref(bl, c, 1, 3, 1);
            }
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(e0));
            for (int it = 0; it < ITERS; it++) {
                bl.w_kv = wl[it % NL][0]; bl.w_sc = wl[it % NL][1];
                run_ref(bl, c, 1, 3, 1);
            }
            CHECK(cudaEventRecord(e1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            printf("%s ref 5-launch : %8.3f us/event\n", c.name, ms * 1000.0f / ITERS);
            for (unsigned g : grids) {
                if (g > cap[ci]) continue;
                for (int it = 0; it < WARM; it++) {
                    bl.w_kv = wl[it % NL][0]; bl.w_sc = wl[it % NL][1];
                    run_fused(bl, c, 1, 3, 1, g, 0);
                }
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(e0));
                for (int it = 0; it < ITERS; it++) {
                    bl.w_kv = wl[it % NL][0]; bl.w_sc = wl[it % NL][1];
                    run_fused(bl, c, 1, 3, 1, g, 0);
                }
                CHECK(cudaEventRecord(e1));
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                printf("%s fused g=%3u  : %8.3f us/event\n", c.name, g, ms * 1000.0f / ITERS);
            }
            for (int l = 0; l < NL; l++)
                for (int m = 0; m < 2; m++) CHECK(cudaFree(wl[l][m]));
        }
    }

    printf("PROTO_M2_COMP OK\n");
    return 0;
}
