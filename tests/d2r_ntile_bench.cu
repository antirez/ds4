/* D2R N-tile microbench -- standalone, no model load, no GPU lock beyond its own run.
 *
 * BUILD + RUN (not wired into the Makefile on purpose: it is an experiment harness,
 * and the Makefile is shared with a parallel branch):
 *     nvcc -O3 --use_fast_math -arch=sm_120f -o /tmp/d2r_ntile_bench tests/d2r_ntile_bench.cu
 *     flock -w 3600 temp/gpu.lock /tmp/d2r_ntile_bench 2048 5
 * Peak device alloc ~4.8 GB (unified memory), so drop_caches first on GB10.
 *
 * WHAT IT ANSWERS (2026-07-16, temp/d2r-microbench.md):  the MXFP4 expert stages'
 * token ("N") tile is 8 for gate/up and 16 for down because tokens x K-blocks x 292 B
 * must fit the 48 KB static __shared__ cap -- not because of registers (72/96 of 255)
 * and not by design.  MXFP4 inherited the widths from the IQ2 kernels (commit 7c92951,
 * "mirrors the iq2 gate/down").  Measured on GB10: gate/up NT 8 -> 16, staged in
 * DYNAMIC smem, is 2.30x and BIT-EXACT; down NT 16 -> 32 is 1.00x (already past its
 * knee); NT 32/64 REGRESS on gate/up (register spill).  That 2.30x is ~= the whole
 * projected D2R tensor-core win, which is why D2R was redirected rather than built.
 *
 * Replicates moe_gate_up_mid_mxfp4_expert_tile8_rowspan_kernel and
 * moe_down_mxfp4_expert_tile16_row2048_kernel from src/cuda/pulsar_cuda_moe.cu
 * verbatim, but templated on the N (token) tile width, to answer:
 *   does widening the N tile capture the D2R win without a tensor core?
 *
 * Shapes pinned to v5mx (DeepSeek-V4-Flash REAP25): n_embd=4096, n_ff_exp=2048,
 * 192 experts, 6 used, 2048 tokens => 12288 pairs, 64 pairs/expert.
 *
 * The K reduction (blocks b, strided by 8 lanes) and the 3-level float fold are
 * IDENTICAL for every NT: level 1 = sb ascending, level 2 = facc per b within a
 * lane, level 3 = quarter_warp_sum over 8 lanes.  Widening NT adds tokens, which
 * are independent output columns.  So every NT must be bit-identical -- checked
 * here against NT=8 (gate/up) / NT=16 (down) on real output, and by the in-tree
 * `make cuda-prefill-gate` when wired into the engine.
 *
 * THREE TRAPS THIS HARNESS HIT, EACH OF WHICH SILENTLY INVERTS THE ANSWER.  If you
 * fork this file, keep all three guards:
 *
 *  1. DECODE MUST MATCH PRODUCTION.  ***STALE AS OF 5c86e4d — READ THIS FIRST.***
 *     When this harness was written, production's dev_e2m1_x2 computed the e2m1
 *     magnitude ARITHMETICALLY, and DECODE=0 below is a verbatim copy of that.
 *     Production has since adopted the packed-shift select this harness measured
 *     as DECODE=1's `dev_e2m1_x2_lut` (two 32-bit immediates + a shift -- NOT the
 *     memory table described below).  So the mapping is now inverted: **DECODE=1
 *     is production and DECODE=0 is the retired form.**  Any future sweep taken
 *     against DECODE=0 as "the baseline" mis-attributes decode cost -- which is
 *     exactly the failure this guard exists to prevent.  Re-copy from production
 *     before trusting a new number.
 *
 *     The original warning, still true and still the reason DECODE is a knob:
 *     replacing the decode with a `static const int8_t lut[16]` is numerically
 *     identical but nvcc lowers it to GLOBAL gather loads, cutting the decode
 *     from 43% to 13% of the stream --
 *     i.e. benchmarking a kernel we do not ship, against which the N-tile looks
 *     worthless.  Verified: with the arithmetic decode this bench's NT=8 kernel is
 *     11,688 SASS instructions vs the shipped kernel's 11,671 (0.15%).  That match is
 *     the harness's correctness gate; check it after any edit.
 *
 *  2. NO RUNTIME-BOUNDED LOOPS OVER THE TOKEN ARRAYS.  `for (p = 0; p < n; p++)` with
 *     runtime n makes ptxas put ys[]/facc[]/xqb[] in LOCAL memory (2.5 KB stack frames
 *     at NT=64), so the sweep measures local-memory traffic, not tile width.  Every
 *     such loop here is `#pragma unroll for (p = 0; p < NT; p++) { if (p >= n) break; }`
 *     -- compile-time bound, uniform predicate.
 *
 *  3. THE BIT-EXACT CHECK CAN PASS VACUOUSLY.  Buffers are memset to 0 before each run,
 *     so two kernels that both wrote nothing memcmp equal and report "YES".
 *     assert_ref_nontrivial() proves the reference is populated first.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define QK_K 256

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} cuda_block_q8_K;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s @ %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

/* ---- verbatim from src/cuda/pulsar_cuda_moe.cu ---- */

/* VERBATIM from src/cuda/pulsar_cuda_moe.cu:184.  Do NOT "improve" this into a table:
 * a `static const int8_t lut[16]` looks equivalent (and IS numerically -- the values are
 * exactly {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}) but nvcc lowers it to GLOBAL gather loads
 * (LDG.E.U8.CONSTANT), which is a different kernel with a different cost.  Measuring the
 * N-tile against a LUT baseline understates production's decode share by ~4x (43% -> 13%)
 * and would silently invalidate the whole sweep.  The LUT is benchmarked separately as
 * DECODE=1 below -- as a lever in its own right, not as the baseline. */
__device__ __forceinline__ static int dev_e2m1_x2_arith(unsigned nib) {
    const unsigned e = (nib >> 1) & 3u;
    const unsigned m = nib & 1u;
    const int mag = e ? (int)((1u << e) | (m << (e - 1u))) : (int)m;
    return (nib & 8u) ? -mag : mag;
}

__device__ __forceinline__ static int dev_e2m1_x2_lut(unsigned nib) {
    /* Same 16 values, packed into two 32-bit immediates and selected with a shift --
     * no memory operand, so no gather and no constant-bank serialisation. */
    const uint32_t w0 = 0x03020100u;      /* nib 0..3 -> 0,1,2,3   */
    const uint32_t w1 = 0x0C080604u;      /* nib 4..7 -> 4,6,8,12  */
    const unsigned k = nib & 7u;
    const uint32_t w = (k < 4u) ? w0 : w1;
    const int mag = (int)((w >> ((k & 3u) * 8u)) & 0xFFu);
    return (nib & 8u) ? -mag : mag;
}

template <int DECODE>
__device__ __forceinline__ static int dev_e2m1_x2(unsigned nib) {
    return DECODE ? dev_e2m1_x2_lut(nib) : dev_e2m1_x2_arith(nib);
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

/* Templated N-tile dot.  Structurally identical to dev_dot_mxfp4_q8_K_block8:
 * decode each weight nibble-group ONCE into wpack, dp4a against all NT tokens,
 * same per-subblock (scale*sumi) then 0.5*y->d order. */
template <uint32_t NT, int DECODE>
__device__ static void dev_dot_mxfp4_q8_K_blockN(
        const unsigned char *x,
        const cuda_block_q8_K *const *ys,
        uint32_t n,
        float acc[NT]) {
    float facc[NT];
    #pragma unroll
    for (uint32_t p = 0; p < NT; p++) facc[p] = 0.0f;
    #pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        const unsigned char *blk = x + (size_t)sb * 17;
        const float scale = __int_as_float((uint32_t)blk[0] << 23);
        int32_t wpack[8];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const unsigned char b0 = blk[1 + g * 2];
            const unsigned char b1 = blk[1 + g * 2 + 1];
            wpack[g] =
                  ((uint32_t)(uint8_t)dev_e2m1_x2<DECODE>(b0 & 0xF))
                | ((uint32_t)(uint8_t)dev_e2m1_x2<DECODE>(b0 >> 4)  << 8)
                | ((uint32_t)(uint8_t)dev_e2m1_x2<DECODE>(b1 & 0xF) << 16)
                | ((uint32_t)(uint8_t)dev_e2m1_x2<DECODE>(b1 >> 4)  << 24);
        }
        /* NOTE: unrolled over the COMPILE-TIME bound with a uniform predicate, not
         * over runtime `n`.  A runtime bound makes ptxas index ys[]/facc[] in LOCAL
         * memory (2.5 KB stack frames at NT=64) and we would be timing local-memory
         * traffic instead of tile width.  np is block-uniform so the predicate is a
         * uniform branch, not divergence. */
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) {
            if (p >= n) break;
            const int8_t *q8 = ys[p]->qs + sb * 32;
            int32_t sumi = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
                sumi = __dp4a(wpack[g], *(const int32_t *)(q8 + g * 4), sumi);
            facc[p] += scale * (float)sumi;
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < NT; p++) { if (p >= n) break; acc[p] += 0.5f * ys[p]->d * facc[p]; }
}

/* gate/up, templated on token tile NT and on the activation staging strategy:
 *   STAGE=0  no staging -- read activations from global (L1/L2).  No SMEM ceiling on NT.
 *   STAGE=1  production's strategy -- stage all NT*xq_blocks q8_K blocks.  SMEM = NT*16*292,
 *            which is what pins NT to 8 today (16*16*292 = 74752 > the 48 KB static cap).
 * K-chunked staging was tried and REJECTED: the accumulators gate[NT]/up[NT] are per-row,
 * so a K-chunk loop cannot be hoisted out of the row loop without keeping 32 rows' worth of
 * accumulators live.  Left inside, it restages every row (64x per block) and destroys the
 * very reuse staging exists for.  STAGE=0 tests the same hypothesis honestly: at NT=16 the
 * activations are 74752 B, which fits the 128 KB L1 when SMEM is unused. */
template <uint32_t ROW_SPAN, uint32_t NT, int STAGE, int DECODE>
__global__ static void gate_up_ntile_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    extern __shared__ cuda_block_q8_K sxq_dyn[];   /* [NT][16] when STAGE */
    uint32_t pair[NT];
    uint32_t tok[NT];
    uint32_t slot[NT];
    const cuda_block_q8_K *xqb[NT];
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) { pair[i] = 0; tok[i] = 0; slot[i] = 0; xqb[i] = NULL; }
    const uint32_t cnt = counts[expert];
    const uint32_t avail = (local_start < cnt) ? (cnt - local_start) : 0u;
    const uint32_t np = avail < NT ? avail : NT;
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) {
        if (i >= np) break;
        pair[i] = sorted_pairs[offsets[expert] + local_start + i];
        tok[i] = pair[i] / n_expert;
        slot[i] = pair[i] - tok[i] * n_expert;
        xqb[i] = xq + (uint64_t)tok[i] * xq_blocks;
    }
    /* STAGE=0 global; STAGE=1 staged with the compute pointer REASSIGNED into
     * xqb (which also held the global staging source, so it is generic -> the
     * activation reads lower to generic LD: this is the SHIPPED kernel); STAGE=2
     * staged with a SEPARATE pure-shared-provenance compute base csrc, which lets
     * nvcc infer the shared space and emit LDS.  csrc is assigned exactly one
     * address space per instantiation, never both -- the whole LDS trick. */
    const cuda_block_q8_K *csrc[NT];
    if (STAGE) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq_dyn[p * xq_blocks + b] = xqb[p][b];
        }
        __syncthreads();
        if (STAGE == 2) {
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) csrc[p] = sxq_dyn + p * xq_blocks;
        } else {
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) { if (p >= np) break; xqb[p] = sxq_dyn + p * xq_blocks; }
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) csrc[p] = xqb[p];
        }
    } else {
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) csrc[p] = xqb[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const unsigned char *gr = (const unsigned char *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const unsigned char *ur = (const unsigned char *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[NT], up[NT];
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) { gate[p] = 0.0f; up[p] = 0.0f; }
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            const cuda_block_q8_K *yb[NT];
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) { if (p >= np) break; yb[p] = csrc[p] + b; }
            dev_dot_mxfp4_q8_K_blockN<NT, DECODE>(gr + (uint64_t)b * 8u * 17u, yb, np, gate);
            dev_dot_mxfp4_q8_K_blockN<NT, DECODE>(ur + (uint64_t)b * 8u * 17u, yb, np, up);
        }
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) {
            if (p >= np) break;
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

/* down, templated on token tile NT. */
template <uint32_t NT, int STAGE, int DECODE>
__global__ static void down_ntile_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    extern __shared__ cuda_block_q8_K sxq_dyn2[];
    uint32_t pair[NT];
    const cuda_block_q8_K *xqb[NT];
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) { pair[i] = 0; xqb[i] = NULL; }
    const uint32_t cnt = counts[expert];
    const uint32_t avail = (local_start < cnt) ? (cnt - local_start) : 0u;
    const uint32_t np = avail < NT ? avail : NT;
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) {
        if (i >= np) break;
        pair[i] = sorted_pairs[offsets[expert] + local_start + i];
        xqb[i] = midq + (uint64_t)pair[i] * midq_blocks;
    }
    if (STAGE) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq_dyn2[p * midq_blocks + b] = xqb[p][b];
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) { if (p >= np) break; xqb[p] = sxq_dyn2 + p * midq_blocks; }
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const unsigned char *wr = (const unsigned char *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[NT];
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) acc[p] = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            const cuda_block_q8_K *yb[NT];
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) { if (p >= np) break; yb[p] = xqb[p] + b; }
            dev_dot_mxfp4_q8_K_blockN<NT, DECODE>(wr + (uint64_t)b * 8u * 17u, yb, np, acc);
        }
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) {
            if (p >= np) break;
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}

/* ---- host harness ---- */

static const uint32_t N_EXPERT_TOTAL = 192;   /* v5mx REAP25 routed layers */
static const uint32_t N_EXPERT_USED  = 6;
static const uint32_t IN_DIM  = 4096;
static const uint32_t MID_DIM = 2048;

struct Routing {
    uint32_t n_tokens, pair_count;
    uint32_t *d_sorted_pairs, *d_offsets, *d_counts;
};

/* Build a balanced routing: every expert gets exactly the same pair count.
 * (Production routing is data-dependent; balanced is the honest average case
 * and makes the tile-count arithmetic exact.) */
static void build_routing(Routing *r, uint32_t n_tokens) {
    r->n_tokens = n_tokens;
    r->pair_count = n_tokens * N_EXPERT_USED;
    uint32_t *h_pairs = (uint32_t *)malloc(r->pair_count * sizeof(uint32_t));
    uint32_t *h_counts = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    uint32_t *h_offsets = (uint32_t *)calloc(N_EXPERT_TOTAL + 1, sizeof(uint32_t));
    /* assign expert = (tok*USED + slot) % TOTAL -> perfectly balanced */
    uint32_t *exp_of_pair = (uint32_t *)malloc(r->pair_count * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            uint32_t e = (t * N_EXPERT_USED + s) % N_EXPERT_TOTAL;
            exp_of_pair[p] = e;
            h_counts[e]++;
        }
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) h_offsets[e + 1] = h_offsets[e] + h_counts[e];
    uint32_t *cursor = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            uint32_t e = exp_of_pair[p];
            /* pair id encodes (tok, slot) exactly as the engine does */
            h_pairs[h_offsets[e] + cursor[e]++] = t * N_EXPERT_USED + s;
        }
    CHECK(cudaMalloc(&r->d_sorted_pairs, r->pair_count * sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_offsets, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_counts, N_EXPERT_TOTAL * sizeof(uint32_t)));
    CHECK(cudaMemcpy(r->d_sorted_pairs, h_pairs, r->pair_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_offsets, h_offsets, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_counts, h_counts, N_EXPERT_TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(h_pairs); free(h_counts); free(h_offsets); free(cursor); free(exp_of_pair);
}

static uint32_t xrand(void);   /* fwd: defined below with rng_state */

/* IMBALANCED routing: each pair independently routed to a random expert, so
 * per-expert counts are uneven and NOT multiples of the tile width -> real
 * PARTIAL tiles (np < NT).  The d2r-microbench post-mortem records that BALANCED
 * routing (exactly 64 pairs/expert, every tile full) systematically OVERSTATES a
 * wide tile -- 2.30x microbench collapsed to 1.90x in-engine.  Partial tiles are
 * where the LDS win is smallest (fewer activation reads to convert per tile), so
 * this is the honest routing for the LD->LDS delta. */
static void build_routing_imbalanced(Routing *r, uint32_t n_tokens) {
    r->n_tokens = n_tokens;
    r->pair_count = n_tokens * N_EXPERT_USED;
    uint32_t *h_pairs = (uint32_t *)malloc(r->pair_count * sizeof(uint32_t));
    uint32_t *h_counts = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    uint32_t *h_offsets = (uint32_t *)calloc(N_EXPERT_TOTAL + 1, sizeof(uint32_t));
    uint32_t *exp_of_pair = (uint32_t *)malloc(r->pair_count * sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            uint32_t e = xrand() % N_EXPERT_TOTAL;   /* independent random route */
            exp_of_pair[p] = e;
            h_counts[e]++;
        }
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) h_offsets[e + 1] = h_offsets[e] + h_counts[e];
    uint32_t *cursor = (uint32_t *)calloc(N_EXPERT_TOTAL, sizeof(uint32_t));
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t s = 0; s < N_EXPERT_USED; s++) {
            uint32_t p = t * N_EXPERT_USED + s;
            uint32_t e = exp_of_pair[p];
            h_pairs[h_offsets[e] + cursor[e]++] = t * N_EXPERT_USED + s;
        }
    CHECK(cudaMalloc(&r->d_sorted_pairs, r->pair_count * sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_offsets, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_counts, N_EXPERT_TOTAL * sizeof(uint32_t)));
    CHECK(cudaMemcpy(r->d_sorted_pairs, h_pairs, r->pair_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_offsets, h_offsets, (N_EXPERT_TOTAL + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_counts, h_counts, N_EXPERT_TOTAL * sizeof(uint32_t), cudaMemcpyHostToDevice));
    /* partial-tile diagnostics at width 16 */
    uint32_t full = 0, partial = 0, empty = 0;
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) {
        if (h_counts[e] == 0) { empty++; continue; }
        full += h_counts[e] / 16u;
        if (h_counts[e] % 16u) partial++;
    }
    printf("# imbalanced routing @width16: %u full tiles, %u partial (np<16), %u empty experts\n\n",
           full, partial, empty);
    free(h_pairs); free(h_counts); free(h_offsets); free(cursor); free(exp_of_pair);
}

/* Tile list for a given tile width, mirroring the engine's build_tiles. */
struct Tiles { uint32_t *d_total, *d_experts, *d_starts; uint32_t n; };

static void build_tiles(Tiles *t, const Routing *r, uint32_t tile_w) {
    uint32_t *h_counts = (uint32_t *)malloc(N_EXPERT_TOTAL * sizeof(uint32_t));
    CHECK(cudaMemcpy(h_counts, r->d_counts, N_EXPERT_TOTAL * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t cap = 0;
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++) cap += (h_counts[e] + tile_w - 1) / tile_w;
    uint32_t *h_exp = (uint32_t *)malloc(cap * sizeof(uint32_t));
    uint32_t *h_start = (uint32_t *)malloc(cap * sizeof(uint32_t));
    uint32_t n = 0;
    for (uint32_t e = 0; e < N_EXPERT_TOTAL; e++)
        for (uint32_t s = 0; s < h_counts[e]; s += tile_w) { h_exp[n] = e; h_start[n] = s; n++; }
    t->n = n;
    CHECK(cudaMalloc(&t->d_total, sizeof(uint32_t)));
    CHECK(cudaMalloc(&t->d_experts, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&t->d_starts, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(t->d_total, &n, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->d_experts, h_exp, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(t->d_starts, h_start, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(h_counts); free(h_exp); free(h_start);
}

static void free_tiles(Tiles *t) {
    cudaFree(t->d_total); cudaFree(t->d_experts); cudaFree(t->d_starts);
}

static uint32_t rng_state = 12345u;
static uint32_t xrand(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

/* Occupancy + resource report for a kernel. */
template <typename K>
static void report_kernel(const char *name, K kern, uint32_t threads, size_t dyn_smem) {
    cudaFuncAttributes a;
    CHECK(cudaFuncGetAttributes(&a, (const void *)kern));
    int blocks = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (const void *)kern, threads, dyn_smem));
    int dev; CHECK(cudaGetDevice(&dev));
    int max_thr_sm; CHECK(cudaDeviceGetAttribute(&max_thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    printf("    %-28s regs=%-4d static_smem=%-7zu dyn_smem=%-7zu spill(st/ld)=%zu/%zu  blocks/SM=%d  occ=%.1f%%\n",
           name, a.numRegs, (size_t)a.sharedSizeBytes, dyn_smem,
           (size_t)a.localSizeBytes, (size_t)a.localSizeBytes,
           blocks, 100.0 * blocks * threads / max_thr_sm);
}

/* Guard against a VACUOUS bit-exactness PASS.  Every buffer is memset to 0 before its
 * kernel runs, so two kernels that both wrote NOTHING would memcmp equal and report
 * "YES" -- certifying a comparison that never happened.  Assert the reference buffer is
 * actually populated before trusting any byte-compare against it. */
static void assert_ref_nontrivial(const char *what, const float *d_buf, size_t n) {
    float *h = (float *)malloc(n * sizeof(float));
    CHECK(cudaMemcpy(h, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t nz = 0, nfin = 0;
    for (size_t i = 0; i < n; i++) { if (h[i] != 0.0f) nz++; if (isfinite(h[i])) nfin++; }
    printf("    [%s reference: %zu/%zu non-zero, %zu/%zu finite]%s\n", what, nz, n, nfin, n,
           nz == 0 ? "  ** REFERENCE ALL ZERO -- EVERY bitexact RESULT BELOW IS VACUOUS **" : "");
    free(h);
}

__global__ static void decode_selftest_kernel(int *out) {
    if (threadIdx.x < 16) {
        out[threadIdx.x] = dev_e2m1_x2_arith(threadIdx.x);
        out[16 + threadIdx.x] = dev_e2m1_x2_lut(threadIdx.x);
    }
}

/* The LUT decode is only admissible if it is numerically IDENTICAL to production's
 * arithmetic decode on all 16 nibbles.  Prove it before timing it. */
static void decode_selftest(void) {
    int *d; CHECK(cudaMalloc(&d, 32 * sizeof(int)));
    decode_selftest_kernel<<<1, 32>>>(d);
    CHECK(cudaDeviceSynchronize());
    int h[32]; CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16; i++) if (h[i] != h[16 + i]) bad++;
    printf("# decode self-test: arith = [");
    for (int i = 0; i < 16; i++) printf("%d%s", h[i], i < 15 ? "," : "");
    printf("]\n# decode self-test: %s (%d/16 mismatches)\n\n",
           bad ? "** LUT DIFFERS -- DECODE=1 ROWS ARE INVALID **" : "lut == arith on all 16 nibbles",
           bad);
    cudaFree(d);
}

int main(int argc, char **argv) {
    uint32_t n_tokens = argc > 1 ? (uint32_t)atoi(argv[1]) : 2048;
    int iters = argc > 2 ? atoi(argv[2]) : 5;

    int dev = 0; CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, dev));
    int smem_per_sm, smem_per_block_optin, regs_per_sm, max_thr_sm;
    CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CHECK(cudaDeviceGetAttribute(&smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CHECK(cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    CHECK(cudaDeviceGetAttribute(&max_thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    printf("# device: %s  sm_%d%d  SMs=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("# smem/SM=%d B  smem/block(optin)=%d B  regs/SM=%d  maxthreads/SM=%d\n",
           smem_per_sm, smem_per_block_optin, regs_per_sm, max_thr_sm);
    printf("# sizeof(cuda_block_q8_K)=%zu\n", sizeof(cuda_block_q8_K));
    decode_selftest();
    printf("# n_tokens=%u  experts=%u used=%u  pairs=%u  pairs/expert=%.1f  iters=%d\n\n",
           n_tokens, N_EXPERT_TOTAL, N_EXPERT_USED, n_tokens * N_EXPERT_USED,
           (double)(n_tokens * N_EXPERT_USED) / N_EXPERT_TOTAL, iters);

    const uint32_t xq_blocks = IN_DIM / QK_K;      /* 16 */
    const uint32_t midq_blocks = MID_DIM / QK_K;   /* 8 */
    const uint64_t gate_row_bytes = (uint64_t)IN_DIM * 17 / 32;    /* 2176 */
    const uint64_t gate_expert_bytes = gate_row_bytes * MID_DIM;
    const uint64_t down_row_bytes = (uint64_t)MID_DIM * 17 / 32;   /* 1088 */
    const uint64_t down_expert_bytes = down_row_bytes * IN_DIM;

    const uint64_t gate_bytes = gate_expert_bytes * N_EXPERT_TOTAL;
    const uint64_t down_bytes = down_expert_bytes * N_EXPERT_TOTAL;
    printf("# gate slab %.3f GB  up slab %.3f GB  down slab %.3f GB  (total %.3f GB)\n\n",
           gate_bytes/1e9, gate_bytes/1e9, down_bytes/1e9, (2.0*gate_bytes + down_bytes)/1e9);

    /* weights */
    char *d_gate, *d_up, *d_down;
    CHECK(cudaMalloc(&d_gate, gate_bytes));
    CHECK(cudaMalloc(&d_up, gate_bytes));
    CHECK(cudaMalloc(&d_down, down_bytes));
    {
        /* 2176 = one MXFP4 row (128 blocks x 17 B); a multiple of 17, and both slab
         * sizes are multiples of it, so block starts stay at i % 17 == 0 in every chunk. */
        size_t chunk = 2176u * 30000u;
        unsigned char *h = (unsigned char *)malloc(chunk);
        for (size_t i = 0; i < chunk; i++) h[i] = (unsigned char)(xrand() & 0xFF);
        /* E8M0 scale byte at each 17-byte block start: 2^(b-127) with b in [120,127]
         * => 2^-7..2^0, which keeps facc = scale*sumi (|sumi| <= 48768) finite and the
         * swiglu output in range. Random bytes here span 2^-127..2^128 -> inf/nan. */
        for (size_t i = 0; i < chunk; i += 17) h[i] = (unsigned char)(120u + (xrand() & 7u));
        for (size_t off = 0; off < gate_bytes; off += chunk) {
            size_t n = (gate_bytes - off < chunk) ? gate_bytes - off : chunk;
            CHECK(cudaMemcpy(d_gate + off, h, n, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_up + off, h, n, cudaMemcpyHostToDevice));
        }
        for (size_t off = 0; off < down_bytes; off += chunk) {
            size_t n = (down_bytes - off < chunk) ? down_bytes - off : chunk;
            CHECK(cudaMemcpy(d_down + off, h, n, cudaMemcpyHostToDevice));
        }
        free(h);
    }

    /* activations */
    cuda_block_q8_K *d_xq, *d_midq;
    CHECK(cudaMalloc(&d_xq, (size_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K)));
    CHECK(cudaMalloc(&d_midq, (size_t)n_tokens * N_EXPERT_USED * midq_blocks * sizeof(cuda_block_q8_K)));
    {
        size_t nx = (size_t)n_tokens * xq_blocks;
        cuda_block_q8_K *h = (cuda_block_q8_K *)malloc(nx * sizeof(cuda_block_q8_K));
        for (size_t i = 0; i < nx; i++) {
            h[i].d = 0.01f + 0.001f * (float)(xrand() % 100);
            for (int j = 0; j < QK_K; j++) h[i].qs[j] = (int8_t)(xrand() % 255 - 127);
            for (int j = 0; j < QK_K/16; j++) h[i].bsums[j] = (int16_t)(xrand() % 1000);
        }
        CHECK(cudaMemcpy(d_xq, h, nx * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice));
        free(h);
        size_t nm = (size_t)n_tokens * N_EXPERT_USED * midq_blocks;
        h = (cuda_block_q8_K *)malloc(nm * sizeof(cuda_block_q8_K));
        for (size_t i = 0; i < nm; i++) {
            h[i].d = 0.01f + 0.001f * (float)(xrand() % 100);
            for (int j = 0; j < QK_K; j++) h[i].qs[j] = (int8_t)(xrand() % 255 - 127);
            for (int j = 0; j < QK_K/16; j++) h[i].bsums[j] = (int16_t)(xrand() % 1000);
        }
        CHECK(cudaMemcpy(d_midq, h, nm * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice));
        free(h);
    }

    float *d_weights, *d_mid, *d_down_out, *d_ref;
    CHECK(cudaMalloc(&d_weights, (size_t)n_tokens * N_EXPERT_USED * sizeof(float)));
    {
        size_t n = (size_t)n_tokens * N_EXPERT_USED;
        float *h = (float *)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) h[i] = 0.1f + 0.001f * (float)(xrand() % 100);
        CHECK(cudaMemcpy(d_weights, h, n * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
    size_t mid_elems = (size_t)n_tokens * N_EXPERT_USED * MID_DIM;
    size_t down_elems = (size_t)n_tokens * N_EXPERT_USED * IN_DIM;
    CHECK(cudaMalloc(&d_mid, mid_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_down_out, down_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_ref, (mid_elems > down_elems ? mid_elems : down_elems) * sizeof(float)));

    Routing rt; build_routing(&rt, n_tokens);

    const float clamp = 7.0f;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    /* traffic model: each expert slab is re-read once per tile */
    #define GU_TRAFFIC(tiles) ((double)(tiles) * (double)gate_expert_bytes * 2.0)
    #define DN_TRAFFIC(tiles) ((double)(tiles) * (double)down_expert_bytes)

    printf("=== GATE/UP (in=%u -> mid=%u, xq_blocks=%u), ROW_SPAN=1024 ===\n", IN_DIM, MID_DIM, xq_blocks);
    printf("%-6s %-7s %-7s %-9s %-10s %-11s %-9s %-8s %s\n",
           "NT", "stage", "tiles", "ms", "GB moved", "eff GB/s", "vs NT8", "bitexact", "notes");

    double gu_base_ms = 0;

#define RUN_GU(NT, STAGE, DEC) do {                                                                   \
    Tiles tl; build_tiles(&tl, &rt, (NT));                                                        \
    size_t dyn = (STAGE) ? (size_t)(NT) * xq_blocks * sizeof(cuda_block_q8_K) : 0;                \
    auto kern = gate_up_ntile_kernel<1024, (NT), (STAGE), (DEC)>;                                        \
    cudaError_t at = cudaSuccess;                                                                 \
    if (dyn > 48u*1024u) at = cudaFuncSetAttribute((const void*)kern,                             \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);                               \
    (void)cudaGetLastError();  /* CLEAR: a failed setattr is sticky and would be misreported     \
                                * as the NEXT config's launch failure. */                        \
    if (at != cudaSuccess) {                                                                      \
        printf("%-6d %-7s %-7u %-9s %-10s %-11s %-9s %-8s SMEM %zu B > optin cap %d B\n",         \
               (NT), (STAGE)?"smem":"global", tl.n, "-", "-", "-", "-", "-", dyn, smem_per_block_optin); \
    } else {                                                                                      \
        dim3 g((MID_DIM + 1023u)/1024u, tl.n, 1);                                                 \
        CHECK(cudaMemset(d_mid, 0, mid_elems * sizeof(float)));                                   \
        kern<<<g, 256, dyn>>>(d_mid, d_gate, d_up, d_xq, rt.d_sorted_pairs, rt.d_offsets,         \
            rt.d_counts, tl.d_total, tl.d_experts, tl.d_starts, d_weights,                        \
            gate_expert_bytes, gate_row_bytes, xq_blocks, MID_DIM, N_EXPERT_USED, clamp);         \
        cudaError_t le = cudaGetLastError();                                                      \
        if (le != cudaSuccess) {                                                                  \
            printf("%-6d %-7s %-7u LAUNCH FAIL: %s\n", (NT), (STAGE)?"smem":"global", tl.n, cudaGetErrorString(le)); \
        } else {                                                                                  \
        CHECK(cudaDeviceSynchronize());                                                           \
        int bitex = -1;                                                                           \
        if ((NT) == 8 && (STAGE) == 1 && (DEC) == 0) { CHECK(cudaMemcpy(d_ref, d_mid, mid_elems*sizeof(float), cudaMemcpyDeviceToDevice)); \
            assert_ref_nontrivial("gate/up", d_ref, mid_elems); } \
        else {                                                                                    \
            float *a = (float*)malloc(mid_elems*sizeof(float)), *b = (float*)malloc(mid_elems*sizeof(float)); \
            CHECK(cudaMemcpy(a, d_mid, mid_elems*sizeof(float), cudaMemcpyDeviceToHost));         \
            CHECK(cudaMemcpy(b, d_ref, mid_elems*sizeof(float), cudaMemcpyDeviceToHost));         \
            bitex = memcmp(a, b, mid_elems*sizeof(float)) == 0 ? 1 : 0;                           \
            if (!bitex) { size_t nd=0; for(size_t i=0;i<mid_elems;i++) if(a[i]!=b[i]) nd++;        \
                printf("    [NT=%d differs in %zu/%zu elements]\n", (NT), nd, mid_elems); }        \
            free(a); free(b);                                                                     \
        }                                                                                         \
        CHECK(cudaEventRecord(ev0));                                                              \
        for (int it = 0; it < iters; it++)                                                        \
            kern<<<g, 256, dyn>>>(d_mid, d_gate, d_up, d_xq, rt.d_sorted_pairs, rt.d_offsets,     \
                rt.d_counts, tl.d_total, tl.d_experts, tl.d_starts, d_weights,                    \
                gate_expert_bytes, gate_row_bytes, xq_blocks, MID_DIM, N_EXPERT_USED, clamp);     \
        CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));                            \
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); ms /= iters;                    \
        double gb = GU_TRAFFIC(tl.n) / 1e9;                                                       \
        if ((NT) == 8 && (STAGE) == 1 && (DEC) == 0) gu_base_ms = ms;                                           \
        printf("%-6d %-7s %-7u %-9.2f %-10.2f %-11.1f %-9s %-8s %s\n", (NT), (STAGE)?"smem":"global",\
               tl.n, ms, gb, gb/(ms/1e3),                                                         \
               gu_base_ms>0 ? ({static char bf[16]; snprintf(bf,16,"%.2fx",gu_base_ms/ms); bf;}) : "-", \
               bitex<0?"base":(bitex?"YES":"** NO **"), (DEC)?"lut-decode":"");                                          \
        report_kernel(#NT "/" #STAGE "/dec" #DEC, kern, 256, dyn);                                            \
        }                                                                                         \
    }                                                                                             \
    free_tiles(&tl);                                                                              \
} while (0)

    RUN_GU(8, 1, 0);     /* <- the shipped kernel: NT=8, staged (generic LD), arith decode */
    RUN_GU(8, 2, 0);     /* NT=8, staged, LDS activation reads */
    RUN_GU(8, 0, 0);
    RUN_GU(16, 1, 0);    /* NT=16 staged, generic LD -- the CURRENTLY SHIPPED gate/up */
    RUN_GU(16, 2, 0);    /* NT=16 staged, LDS activation reads -- THE LEVER */
    RUN_GU(16, 0, 0);
    RUN_GU(32, 1, 0);
    RUN_GU(32, 0, 0);
    RUN_GU(64, 1, 0);
    RUN_GU(64, 0, 0);
    /* the decode itself as an independent lever, at the current and the best tile */
    RUN_GU(8, 1, 1);
    RUN_GU(16, 0, 1);
    RUN_GU(16, 1, 1);

    /* ---- the honest LD->LDS delta, under IMBALANCED routing (partial tiles) ---- */
    printf("\n=== GATE/UP, IMBALANCED routing (partial tiles; the honest LDS delta) ===\n");
    printf("%-6s %-7s %-7s %-9s %-10s %-11s %-9s %-8s %s\n",
           "NT", "stage", "tiles", "ms", "GB moved", "eff GB/s", "vs NT8", "bitexact", "notes");
    /* swap the routing under the macro (which references `rt`), refresh the base */
    cudaFree(rt.d_sorted_pairs); cudaFree(rt.d_offsets); cudaFree(rt.d_counts);
    build_routing_imbalanced(&rt, n_tokens);
    gu_base_ms = 0;
    RUN_GU(8, 1, 0);     /* NT=8 staged base under imbalanced routing (refreshes d_ref) */
    RUN_GU(16, 1, 0);    /* NT=16 staged, generic LD -- shipped */
    RUN_GU(16, 2, 0);    /* NT=16 staged, LDS -- the lever, honest routing */

    printf("\n=== DOWN (mid=%u -> out=%u, midq_blocks=%u) ===\n", MID_DIM, IN_DIM, midq_blocks);
    printf("%-6s %-7s %-7s %-9s %-10s %-11s %-9s %-8s %s\n",
           "NT", "stage", "tiles", "ms", "GB moved", "eff GB/s", "vs NT16", "bitexact", "notes");
    double dn_base_ms = 0;

#define RUN_DN(NT, STAGE, DEC) do {                                                                   \
    Tiles tl; build_tiles(&tl, &rt, (NT));                                                        \
    size_t dyn = (STAGE) ? (size_t)(NT) * midq_blocks * sizeof(cuda_block_q8_K) : 0;              \
    auto kern = down_ntile_kernel<(NT), (STAGE), (DEC)>;                                                 \
    cudaError_t at = cudaSuccess;                                                                 \
    if (dyn > 48u*1024u) at = cudaFuncSetAttribute((const void*)kern,                             \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);                               \
    (void)cudaGetLastError();  /* CLEAR: a failed setattr is sticky and would be misreported     \
                                * as the NEXT config's launch failure. */                        \
    if (at != cudaSuccess) {                                                                      \
        printf("%-6d %-7s %-7u %-9s %-10s %-11s %-9s %-8s SMEM %zu B > optin cap %d B\n",         \
               (NT), (STAGE)?"smem":"global", tl.n, "-", "-", "-", "-", "-", dyn, smem_per_block_optin); \
    } else {                                                                                      \
        dim3 g((IN_DIM + 2047u)/2048u, tl.n, 1);                                                  \
        CHECK(cudaMemset(d_down_out, 0, down_elems * sizeof(float)));                             \
        kern<<<g, 256, dyn>>>(d_down_out, d_down, d_midq, rt.d_sorted_pairs, rt.d_offsets,        \
            rt.d_counts, tl.d_total, tl.d_experts, tl.d_starts,                                   \
            down_expert_bytes, down_row_bytes, midq_blocks, IN_DIM, N_EXPERT_USED);               \
        cudaError_t le = cudaGetLastError();                                                      \
        if (le != cudaSuccess) {                                                                  \
            printf("%-6d %-7s %-7u LAUNCH FAIL: %s\n", (NT), (STAGE)?"smem":"global", tl.n, cudaGetErrorString(le)); \
        } else {                                                                                  \
        CHECK(cudaDeviceSynchronize());                                                           \
        int bitex = -1;                                                                           \
        if ((NT) == 16 && (STAGE) == 1 && (DEC) == 0) { CHECK(cudaMemcpy(d_ref, d_down_out, down_elems*sizeof(float), cudaMemcpyDeviceToDevice)); \
            assert_ref_nontrivial("down", d_ref, down_elems); } \
        else {                                                                                    \
            float *a = (float*)malloc(down_elems*sizeof(float)), *b = (float*)malloc(down_elems*sizeof(float)); \
            CHECK(cudaMemcpy(a, d_down_out, down_elems*sizeof(float), cudaMemcpyDeviceToHost));   \
            CHECK(cudaMemcpy(b, d_ref, down_elems*sizeof(float), cudaMemcpyDeviceToHost));        \
            bitex = memcmp(a, b, down_elems*sizeof(float)) == 0 ? 1 : 0;                          \
            if (!bitex) { size_t nd=0; for(size_t i=0;i<down_elems;i++) if(a[i]!=b[i]) nd++;       \
                printf("    [NT=%d differs in %zu/%zu elements]\n", (NT), nd, down_elems); }       \
            free(a); free(b);                                                                     \
        }                                                                                         \
        CHECK(cudaEventRecord(ev0));                                                              \
        for (int it = 0; it < iters; it++)                                                        \
            kern<<<g, 256, dyn>>>(d_down_out, d_down, d_midq, rt.d_sorted_pairs, rt.d_offsets,    \
                rt.d_counts, tl.d_total, tl.d_experts, tl.d_starts,                               \
                down_expert_bytes, down_row_bytes, midq_blocks, IN_DIM, N_EXPERT_USED);           \
        CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));                            \
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); ms /= iters;                    \
        double gb = DN_TRAFFIC(tl.n) / 1e9;                                                       \
        if ((NT) == 16 && (STAGE) == 1 && (DEC) == 0) dn_base_ms = ms;                                          \
        printf("%-6d %-7s %-7u %-9.2f %-10.2f %-11.1f %-9s %-8s %s\n", (NT), (STAGE)?"smem":"global",\
               tl.n, ms, gb, gb/(ms/1e3),                                                         \
               dn_base_ms>0 ? ({static char bf[16]; snprintf(bf,16,"%.2fx",dn_base_ms/ms); bf;}) : "-", \
               bitex<0?"base":(bitex?"YES":"** NO **"), (DEC)?"lut-decode":"");                                          \
        report_kernel(#NT "/" #STAGE "/dec" #DEC, kern, 256, dyn);                                            \
        }                                                                                         \
    }                                                                                             \
    free_tiles(&tl);                                                                              \
} while (0)

    RUN_DN(16, 1, 0);    /* <- the shipped kernel */
    RUN_DN(16, 0, 0);
    RUN_DN(32, 1, 0);
    RUN_DN(32, 0, 0);
    RUN_DN(64, 1, 0);
    RUN_DN(64, 0, 0);
    RUN_DN(16, 1, 1);
    RUN_DN(32, 0, 1);

    printf("\n# done\n");
    return 0;
}
