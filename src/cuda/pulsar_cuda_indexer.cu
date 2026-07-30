#include "pulsar_cuda_internal.h"

/* When set, the indexer compressed cache is stored MXKV-FP4-packed
 * (PULSAR_MXKV_FP4_ROWBYTES(128) = 68 B/row: 64 E2M1 nibble pairs low-nibble-first
 * + 4 E8M0 block-32 scales) instead of f32.  The cache rows are QAT-roundtripped
 * to exactly these values in both modes, so the scores are bit-identical; packed
 * mode only changes storage and read traffic.  head_dim must be 128. */
static int g_indexer_fp4 = 0;

void pulsar_gpu_indexer_set_fp4(int on) { g_indexer_fp4 = on ? 1 : 0; }

__device__ static inline float idx_e2m1_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static inline float idx_comp_load_dev(const float *index_comp, int fp4,
                                                 uint64_t row, uint32_t d,
                                                 uint32_t head_dim) {
    if (!fp4) return index_comp[row * head_dim + d];
    const uint8_t *r = (const uint8_t *)index_comp + row * PULSAR_MXKV_FP4_ROWBYTES(128u);
    /* 2^(e8-127) built directly as a float exponent — exact for e8 in [1,254],
     * and the pack amax floor (7e-38 -> e8 >= 2) rules out byte 0. ~20% cheaper
     * per scan call than exp2f in the occupancy-bound small-ctx regime. */
    const float scale = __uint_as_float((uint32_t)r[64u + (d >> 5u)] << 23);
    const uint8_t byte = r[d >> 1u];
    const uint8_t nib = (d & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xfu);
    const float mag = idx_e2m1_value_dev((int)(nib & 7u));
    return (nib & 8u) ? -mag * scale : mag * scale;
}



/* positions/seq_id/comp_cap/n_banks (descriptor-aware indexer kernels): the
 * same per-row multi-session banking as the decode attention kernels
 * (pulsar_cuda_attention.cu — design adapted from the MIT-licensed Entrpi/ds4
 * fork, v0.2 c71a49a; reimplemented, no code copied).  positions[t] is row
 * t's absolute query position, seq_id[t] its TRUE bank id; the row's visible
 * compressed count derives per row as (qpos+1)/ratio — the SAME emit-
 * inclusive rule the classic scalar path uses ((pos0+t+1)/ratio), because
 * the engine emits a step's compressed rows BEFORE the indexer scan reads
 * them.  Banked rows read the compressed cache at seq_id*comp_cap offsets;
 * the scalar n_comp is a cross-bank superset used only as the scan/grid
 * bound, never to address into a bank.  Rows past the row's own visible
 * frontier score -INFINITY so top-k never selects an unprimed or foreign
 * slot (the shallow-bank rows of a mixed batch see -INF for the superset
 * tail).  Dead rows (out-of-pool seq_id) fail visible: the whole score row
 * is -INFINITY, and the consuming attention kernel's own dead-row guard
 * zeroes that row's head outputs.  positions == NULL && seq_id == NULL
 * degenerates to the classic single-cache scalar path bit-exactly. */
__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        int fp4,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ index_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (seq_id && (uint32_t)seq_id[t] >= n_banks) {
        /* Dead/evicted row: fail-visible -INF score row (see header note). */
        if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
        return;
    }
    const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
    if (causal) {
        uint32_t n_visible = (qpos + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    /* Per-bank index base (see attention_decode_mixed_kernel): base-pointer table
     * → separate per-bank allocation at LOCAL row; NULL → classic seq_id*comp_cap. */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[t] : 0u;
    const float *index_src = index_bank_ptrs ? (const float *)index_bank_ptrs[sid_b] : index_comp;
    const uint64_t comp_base = index_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
            dot += qh[d] * idx_comp_load_dev(index_src, fp4, comp_base + c, d, head_dim);
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}



/* Descriptor-aware variant of the single-token fast tier: the banked entry
 * keeps this kernel (rather than forcing the generic per-(row,comp) kernel)
 * because its reduction order is what the classic single-token decode uses —
 * the PULSAR_DECODE_DESCR byte-gate and any per-session solo/banked comparison
 * depend on the banked n_tokens==1 scan being bit-identical to classic.
 * (Documented deviation from the Tier-2 spec's "generic only" note; the WMMA
 * multi-token tier does stay single-bank.)  Descriptor semantics match
 * indexer_scores_kernel (one row, so positions[0]/seq_id[0]). */
__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal,
        int fp4,
        const int32_t * __restrict__ positions,
        const int32_t * __restrict__ seq_id,
        const void * const * __restrict__ index_bank_ptrs,
        uint32_t comp_cap,
        uint32_t n_banks) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (seq_id && (uint32_t)seq_id[0] >= n_banks) {
        /* Dead/evicted row: fail-visible (see indexer_scores_kernel). */
        if (tid == 0) scores[c] = -INFINITY;
        return;
    }
    const uint32_t qpos = positions ? (uint32_t)positions[0] : pos0;
    if (causal) {
        const uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }
    /* Per-bank index base (see indexer_scores_kernel). */
    const uint32_t sid_b = seq_id ? (uint32_t)seq_id[0] : 0u;
    const float *index_src = index_bank_ptrs ? (const float *)index_bank_ptrs[sid_b] : index_comp;
    const uint64_t comp_base = index_bank_ptrs ? 0u
                             : (seq_id ? (uint64_t)sid_b * comp_cap : 0u);

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = idx_comp_load_dev(index_src, fp4, comp_base + c, tid, 128u);
    __syncthreads();

    /* Per-warp accumulation: warp w owns heads w, w+4, ..., w+60 and keeps a
     * private running sum, so the loop needs no block-wide synchronization
     * (the old version paid two __syncthreads() per 4-head group = 32 per
     * row). One sync at the end combines the four warp totals. The reduction
     * order changes by ~1 ULP vs the old serial accumulate, far below the
     * model's run-to-run float-atomic nondeterminism. */
    float wtotal = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) wtotal += fmaxf(dot, 0.0f) * weights[h] * scale;
    }
    if (lane == 0) partial[warp] = wtotal;
    __syncthreads();
    if (tid == 0) scores[c] = partial[0] + partial[1] + partial[2] + partial[3];
}



__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal,
        int fp4) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = idx_comp_load_dev(index_comp, fp4, comp, d, head_dim);
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}



/* Single-block argmax over n_vocab F32 logits. One block of 1024 threads
 * cooperatively scans the vocab, tracking a (best_v, best_idx) pair per
 * thread, then reduces in shared memory with value-keyed comparison.
 *
 * Tie-breaking: lower index wins, matching the host sample_argmax used by
 * the CPU reference path. Replaces the indexer-as-argmax workaround used
 * in the speculative top-id sites, which fell through to the legacy single-thread
 * indexer_topk_kernel at top_k=1, costing ~17.5 ms per call on n_vocab=129280. */
__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            /* Larger value wins; on exact ties prefer the lower index. */
            const bool take_right = (vr > vl) || (vr == vl && ir < il);
            if (take_right) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = sm_idx[0];
}



__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}



__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}



/* Apply one bitonic compare-exchange to a single element.
 *
 * Replicates EXACTLY the shared-memory network in indexer_topk_pow2_kernel:
 * for the pair (low = i, high = i^j) the original writes
 *     swap = desc ? better(b,a) : better(a,b)      (a = low, b = high)
 *     vals[low]  = swap ? b : a
 *     vals[high] = swap ? a : b
 * so the element at `low` takes the partner iff (desc ? better(p,o)
 * : better(o,p)) and the element at `high` takes it iff the mirrored
 * predicate holds.  Both members of a pair share the same `k` bit
 * (j < k), so `desc` can be computed from either index.  No comparison
 * result is reinterpreted, so this is bit-identical for every input,
 * including NaN and +-0. */
__device__ __forceinline__ static void topk_ce_apply(float &ov, uint32_t &oi,
                                                     float pv, uint32_t pi,
                                                     bool desc, bool is_low) {
    const bool take = (is_low == desc) ? topk_score_better(pv, pi, ov, oi)
                                       : topk_score_better(ov, oi, pv, pi);
    if (take) {
        ov = pv;
        oi = pi;
    }
}



/* Bitonic sort of SORT_N (score,index) pairs held in registers, 4 elements per
 * thread, launched with SORT_N/4 threads.  Element i = 4*threadIdx.x + m lives
 * in v[m]/x[m].
 *
 * Same network, same comparator, same pair directions as the all-shared
 * indexer_topk_pow2_kernel loop -- only the data movement changes:
 *   j <  4   partner is in the same thread   -> pure registers
 *   j <  128 partner is in the same warp     -> __shfl_xor_sync
 *   j >= 128 partner is in another warp      -> shared memory (vectorised)
 * That leaves only 15 of the 78 stages touching shared memory (and 30 of the
 * 78 barriers), which is what this kernel is actually bound by.
 *
 * On entry the input must already be in sv[]/sx[]; on exit the sorted result
 * is in the registers (shared memory contents are stale). */
template <uint32_t SORT_N>
__device__ __forceinline__ static void topk_bitonic_sort_regs4(
        float *sv, uint32_t *sx, float (&v)[4], uint32_t (&x)[4]) {
    const uint32_t tid = threadIdx.x;
    float4 *sv4 = reinterpret_cast<float4 *>(sv);
    uint4 *sx4 = reinterpret_cast<uint4 *>(sx);
    {
        const float4 lv = sv4[tid];
        const uint4 lx = sx4[tid];
        v[0] = lv.x; v[1] = lv.y; v[2] = lv.z; v[3] = lv.w;
        x[0] = lx.x; x[1] = lx.y; x[2] = lx.z; x[3] = lx.w;
    }

    #pragma unroll
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        #pragma unroll
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            if (j >= 128u) {
                __syncthreads();
                sv4[tid] = make_float4(v[0], v[1], v[2], v[3]);
                sx4[tid] = make_uint4(x[0], x[1], x[2], x[3]);
                __syncthreads();
                const uint32_t pt = tid ^ (j >> 2u);
                const float4 pv = sv4[pt];
                const uint4 px = sx4[pt];
                const float pva[4] = {pv.x, pv.y, pv.z, pv.w};
                const uint32_t pxa[4] = {px.x, px.y, px.z, px.w};
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], pva[m], pxa[m],
                                  (i & k) == 0u, (i & j) == 0u);
                }
            } else if (j >= 4u) {
                const uint32_t lane_mask = j >> 2u;
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const float pv = __shfl_xor_sync(0xffffffffu, v[m], lane_mask);
                    const uint32_t px = __shfl_xor_sync(0xffffffffu, x[m], lane_mask);
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], pv, px,
                                  (i & k) == 0u, (i & j) == 0u);
                }
            } else {
                const float ov[4] = {v[0], v[1], v[2], v[3]};
                const uint32_t oi[4] = {x[0], x[1], x[2], x[3]};
                #pragma unroll
                for (uint32_t m = 0u; m < 4u; m++) {
                    const uint32_t mm = m ^ j;
                    const uint32_t i = tid * 4u + m;
                    topk_ce_apply(v[m], x[m], ov[mm], oi[mm],
                                  (i & k) == 0u, (i & j) == 0u);
                }
            }
        }
    }
}



__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}



__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}



__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}



__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}



template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}



template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}



template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) out[i] = x[m];
    }
}



template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) selected[(uint64_t)t * top_k + i] = x[m];
    }
}



template <uint32_t SORT_N>
__launch_bounds__(1024) __global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    float v[4];
    uint32_t x[4];
    topk_bitonic_sort_regs4<SORT_N>(vals, idxs, v, x);

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    #pragma unroll
    for (uint32_t m = 0u; m < 4u; m++) {
        const uint32_t i = tid * 4u + m;
        if (i < top_k) dst[i] = x[m];
    }
}



__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

/* Configure indexer_topk_8192_cub_kernel's opt-in dynamic shared memory ONCE and
 * cache the result. The device, its max-optin-smem, and the per-function attribute
 * are process-invariant, so the cudaGetDevice + cudaDeviceGetAttribute +
 * cudaFuncSetAttribute triple must not run per launch (it did, per layer per token
 * at >=16k ctx — squarely on the decode launch floor). Returns the dynamic smem
 * byte count if the cub path is usable on this device, or 0 if the caller must fall
 * back to the pow2 kernel. Decode is single-threaded; a first-call race is benign
 * (cudaFuncSetAttribute is idempotent). */
static int indexer_topk_cub_smem(void) {
    static int cached = -1; /* -1 uninit, 0 unavailable, >0 smem bytes */
    if (cached >= 0) return cached;
    using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
    const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
    int dev = 0, max_optin_smem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&max_optin_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        max_optin_smem < smem ||
        cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
        cached = 0;
        return 0;
    }
    cached = smem;
    return smem;
}



static int indexer_scores_launch(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *index_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    const int fp4 = g_indexer_fp4;
    /* Descriptor (banked) mode: both per-row arrays or neither; the comp
     * cache operand is the whole bank pool, so the byte bound scales by
     * n_banks * comp_cap rows and the uint32 row ABI (seq*cap + local) must
     * not overflow.  Banked scans are always causal with a real ratio (the
     * per-row visible count derives from position).  The scalar n_comp is
     * the cross-bank superset: scan/grid bound and scores-row stride only,
     * never a bank address.  Rejections are fail-loud: a silent 0 return
     * looks like a generic launch failure to the driver. */
    const int descr = positions != NULL || seq_id != NULL;
    if (descr &&
        (!positions || !seq_id || n_banks == 0 || ratio == 0 || !causal ||
         comp_cap < n_comp ||
         positions->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         seq_id->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
         (uint64_t)n_banks * comp_cap > 4294967296ull)) {
        fprintf(stderr,
                "pulsar: banked indexer scores rejected: bad descriptor args "
                "(n_tokens=%u n_banks=%u comp_cap=%u n_comp=%u ratio=%u causal=%u)\n",
                n_tokens, n_banks, comp_cap, n_comp, ratio, causal);
        return 0;
    }
    /* Per-bank split: index_comp is bank 0's comp_cap-row allocation when the
     * base-pointer table is present (see attention_decode_batch_launch). */
    const uint64_t comp_rows_min = (descr && index_bank_ptrs) ? (uint64_t)comp_cap
                                 : descr ? (uint64_t)n_banks * comp_cap
                                         : (uint64_t)n_comp;
    const uint64_t comp_bytes = fp4
        ? comp_rows_min * PULSAR_MXKV_FP4_ROWBYTES(128u)
        : comp_rows_min * head_dim * sizeof(float);
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        (fp4 && head_dim != 128u) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < comp_bytes ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    const int32_t *positions_ptr = descr ? (const int32_t *)positions->ptr : NULL;
    const int32_t *seq_id_ptr = descr ? (const int32_t *)seq_id->ptr : NULL;
    const void * const *index_bank_ptrs_ptr =
        (descr && index_bank_ptrs) ? (const void * const *)index_bank_ptrs->ptr : NULL;
    const uint32_t kernel_n_banks = descr ? n_banks : 1u;
    static const int no_direct_one = getenv("PULSAR_CUDA_NO_INDEXER_DIRECT_ONE") != NULL;
    static const int no_wmma = getenv("PULSAR_CUDA_NO_INDEXER_WMMA") != NULL;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u && !no_direct_one) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0, fp4,
                                                         positions_ptr, seq_id_ptr,
                                                         index_bank_ptrs_ptr,
                                                         comp_cap, kernel_n_banks);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    /* The WMMA tier stays single-bank (like the reference design): banked
     * multi-token rows are forced onto the generic per-(comp,row) kernel. */
    if (!descr && !g_quality_mode && head_dim == 128u && n_head == 64u && !no_wmma) {
        dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
        indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0, fp4);
        return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0, fp4,
                                         positions_ptr, seq_id_ptr,
                                         index_bank_ptrs_ptr,
                                         comp_cap, kernel_n_banks);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}



int pulsar_gpu_indexer_score_one_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0,
                                 NULL, NULL, NULL, 0, 1);
}



int pulsar_gpu_indexer_scores_prefill_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1,
                                 NULL, NULL, NULL, 0, 1);
}



int pulsar_gpu_indexer_scores_decode_batch_tensor(
        pulsar_gpu_tensor       *scores,
        const pulsar_gpu_tensor *q,
        const pulsar_gpu_tensor *weights,
        const pulsar_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        const pulsar_gpu_tensor *positions,
        const pulsar_gpu_tensor *seq_id,
        const pulsar_gpu_tensor *index_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1,
                                 positions, seq_id, index_bank_ptrs, comp_cap, n_banks);
}



int pulsar_gpu_indexer_topk_tensor(
        pulsar_gpu_tensor       *selected,
        const pulsar_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_topk1024 = getenv("PULSAR_CUDA_NO_TOPK1024") != NULL;
    static const int no_topk2048 = getenv("PULSAR_CUDA_NO_TOPK2048") != NULL;
    static const int no_topk8192 = getenv("PULSAR_CUDA_NO_TOPK8192") != NULL;
    static const int no_topk_chunked = getenv("PULSAR_CUDA_NO_TOPK_CHUNKED") != NULL;
    if (top_k == 512u && n_comp <= 1024u &&
        !no_topk1024) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        !no_topk2048) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        !no_topk2048) {
        if (n_comp == 4096u) {
            const int smem = indexer_topk_cub_smem();
            if (smem > 0) {
                indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                             (const float *)scores->ptr,
                                                                             n_comp, n_tokens, top_k);
                return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        !no_topk2048 &&
        !no_topk8192) {
        if (n_comp > 4096u) {
            const int smem = indexer_topk_cub_smem();
            if (smem > 0) {
                indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                             (const float *)scores->ptr,
                                                                             n_comp, n_tokens, top_k);
                return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u && !no_topk2048 &&
        !no_topk_chunked) {
        /* Chunk width is chosen by batch shape, and ONLY by batch shape.
         *
         * A bitonic sort costs N*log2(N)*(log2(N)+1)/2 compare-exchanges, so
         * two 2048-chunks are cheaper in total work than one 4096-chunk
         * (2*66*2048 = 270k vs 78*4096 = 319k) and they run as two INDEPENDENT
         * blocks instead of one.  Which of those matters depends entirely on
         * n_tokens:
         *
         *   decode  (n_tokens == 1): the grid is (1, n_chunks) -- three blocks
         *     on a 48-SM GPU.  The stage is pure block latency, so halving the
         *     per-block network depth is the whole game.  Measured standalone at
         *     n_comp=9216 (the ctx-32k value): chunk stage 33.3 -> 22.1 us,
         *     whole chain 59.8 -> 50.4 us; in-engine decode_topk 70.2 -> 60.2 us.
         *
         *   prefill (n_tokens >= 2): the grid is (n_tokens, n_chunks) and is
         *     already saturated by token parallelism.  There is no latency to
         *     recover, so narrower chunks buy nothing and merely double the
         *     block count.  Keep 4096 there -- byte-for-byte the pre-existing
         *     prefill configuration, so this change structurally cannot regress
         *     prefill.
         *
         * Widening/narrowing is output-neutral: same comparator, same bitonic
         * network, same "top-512 of each chunk, then merge" reduction.
         * topk_score_better is a strict total order on (score, index) for every
         * NaN-free input (indices are distinct; FTZ makes +-0 and denormals
         * compare equal, and the -INFINITY/UINT32_MAX pads are literally
         * identical elements), so the selected top-512 sequence is uniquely
         * determined and independent of how the input is chunked.  Verified
         * bitwise over 400,896 standalone selections spanning uniform,
         * heavy-tie, all-equal, +-0-only, +-inf, denormal, coarse-quantised and
         * two-valued inputs at n_comp 8193..131072, and over 440,745,984
         * in-engine selections at ctx 36864: 0 differing.  NaN input is the one
         * exception -- NaN compares "not better" in both directions, so the
         * comparator is not an ordering there and NO restructuring can preserve
         * it; a NaN score already makes the selection arbitrary.
         *
         * Either width keeps the final merge inside SORT_N=4096: the tree loop
         * below reduces to at most PULSAR_CUDA_TOPK_MERGE_GROUP(=8) * 512 = 4096
         * candidates. */
        const uint32_t chunk_n = (n_tokens == 1u) ? 2048u : 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > PULSAR_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + PULSAR_CUDA_TOPK_MERGE_GROUP - 1u) / PULSAR_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        if (chunk_n == 2048u) {
            indexer_topk_chunk_pow2_kernel<2048><<<grid_chunks, 512>>>(cur,
                                                                        (const float *)scores->ptr,
                                                                        n_comp,
                                                                        n_tokens,
                                                                        top_k,
                                                                        candidate_stride);
        } else {
            indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                        (const float *)scores->ptr,
                                                                        n_comp,
                                                                        n_tokens,
                                                                        top_k,
                                                                        candidate_stride);
        }
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > PULSAR_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + PULSAR_CUDA_TOPK_MERGE_GROUP - 1u) / PULSAR_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    PULSAR_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}



int pulsar_gpu_argmax_tensor(
        pulsar_gpu_tensor       *out_idx,
        const pulsar_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}



int pulsar_gpu_dsv4_topk_mask_tensor(
        pulsar_gpu_tensor       *mask,
        const pulsar_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}


/* ===== MXFP8 (E4M3 + per-32 E8M0) weight matmul via cuBLASLt block-scaling.
 * Hardware MX tensor cores on GB10 (~2x FP16 prefill). Scale tensor uses the
 * Blackwell 128x4 tile swizzle (see blackwell_mx_scale_swizzle). ===== */
cublasLtHandle_t g_cublaslt = NULL;

