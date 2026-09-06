/* Standalone prototype kernels for the Qwen sparse-QSA GQA MMA attention.
 *
 * Compiled at runtime from source by speed-bench/gqa_mma_proto.m so every
 * measurement always reflects this file (no stale-binary artifacts).  The
 * variants exist to answer two questions the production tree could not:
 *   1. Does the 64-thread two-simdgroup LOCKSTEP organization (shared K/V
 *      staging, half the gather traffic of the per-simdgroup production
 *      kernel) beat the 113 ms/layer scalar kernel?
 *   2. Which phase (staging loads, matrix products, softmax roundtrip,
 *      loop/barrier skeleton) binds it?
 *
 * Numerics mirror kernel_qwen4_qsa_attention_gqa_mma_f32: F16 operands,
 * F32 accumulation, F32 softmax through threadgroup tiles, exact
 * production rank-to-token semantics (see QWEN38_PERF_HANDOFF item 20).
 */
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

static inline float qwen4_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct proto_args {
    uint queries;
    uint cache_cap;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint top_k;
    uint ratio;
    uint max_selected;
    uint debug;
};

constant constexpr uint GQA = 12u;
constant constexpr uint D = 256u;

template <int ORG, int BK, int DC, int SKIP>
static inline void gqa_mma_proto_body(
        constant proto_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        threadgroup uchar *tgm [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    /* ORG 0: per-simdgroup, 32 threads, grid (queries, 2*kv_heads),
     * eight head rows per group (the production structure).
     * ORG 1: lockstep, 64 threads, grid (queries, kv_heads), two
     * simdgroups of eight rows sharing every staged K/V tile. */
    constexpr uint ROWS = 8u;
    constexpr uint HPAD = ORG == 1 ? 16u : 8u;
    constexpr uint TPT = ORG == 1 ? 64u : 32u;
    constexpr uint NSG = ORG == 1 ? 2u : 1u;
    constexpr uint DCHUNKS = D / DC;
    constexpr uint TK = BK / 8u;
    constexpr uint OD = D / 8u;

    threadgroup half *qs = (threadgroup half *)tgm;
    threadgroup half *kvs = qs + HPAD * D;
    threadgroup float *s_tile = (threadgroup float *)(kvs + BK * DC);
    threadgroup half *p_tile = (threadgroup half *)(s_tile + HPAD * BK);
    threadgroup int *sel = (threadgroup int *)(p_tile + HPAD * BK);
    threadgroup float *stats = (threadgroup float *)(sel + BK);
    threadgroup float *rfac = stats + HPAD * 2u;
    threadgroup int *vote = (threadgroup int *)(rfac + HPAD);
    threadgroup float *ores = (threadgroup float *)kvs; /* overlay */

    const uint query = group.x;
    if (query >= args.queries) return;
    const uint kv_pair = group.y;
    uint kv_head, half_idx;
    if (ORG == 1) {
        if (kv_pair >= args.kv_heads) return;
        kv_head = kv_pair;
        half_idx = 0u;
    } else {
        if (kv_pair >= args.kv_heads * 2u) return;
        kv_head = kv_pair / 2u;
        half_idx = kv_pair % 2u;
    }
    const uint tid = (uint)sg * 32u + (uint)lane;
    const uint row_base = ORG == 1 ? (uint)sg * 8u : 0u;
    const uint valid_rows = ORG == 1
        ? ((uint)sg == 0u ? 8u : GQA - 8u)
        : (half_idx == 0u ? 8u : GQA - 8u);

    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count =
        min(min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;

    if (selected == 0u) {
        if (ORG == 1) {
            for (uint i = tid; i < GQA * D; i += TPT)
                out[((ulong)query * args.query_heads +
                     kv_head * GQA + i / D) * args.head_dim + (i % D)] = 0.0f;
        } else {
            for (uint i = tid; i < valid_rows * D; i += TPT) {
                const uint head = kv_head * GQA + half_idx * 8u + i / D;
                out[((ulong)query * args.query_heads + head) *
                    args.head_dim + (i % D)] = 0.0f;
            }
        }
        return;
    }

    /* Stage the query-head rows once as F16 (lockstep: all 16 rows). */
    for (uint i = tid; i < HPAD * (D / 8u); i += TPT) {
        const uint row = i / (D / 8u);
        const uint d8 = i % (D / 8u);
        const uint head_row = ORG == 1 ? row : half_idx * 8u + row;
        const uint head = kv_head * GQA + head_row;
        float4 f0 = 0.0f;
        float4 f1 = 0.0f;
        if (head_row < GQA) {
            const device const float *qrow = q +
                ((ulong)query * args.query_heads + head) * args.head_dim +
                d8 * 8u;
            f0 = *((device const float4 *)qrow);
            f1 = *((device const float4 *)(qrow + 4u));
        }
        qs[row * D + d8 * 8u + 0u] = half(f0.x);
        qs[row * D + d8 * 8u + 1u] = half(f0.y);
        qs[row * D + d8 * 8u + 2u] = half(f0.z);
        qs[row * D + d8 * 8u + 3u] = half(f0.w);
        qs[row * D + d8 * 8u + 4u] = half(f1.x);
        qs[row * D + d8 * 8u + 5u] = half(f1.y);
        qs[row * D + d8 * 8u + 6u] = half(f1.z);
        qs[row * D + d8 * 8u + 7u] = half(f1.w);
    }

    simdgroup_float8x8 ofrag[OD];
    for (uint dd = 0u; dd < OD; dd++)
        ofrag[dd] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint r = tid; r < HPAD; r += TPT) {
        stats[r * 2u] = -INFINITY;
        stats[r * 2u + 1u] = 0.0f;
    }

    if (SKIP == 3) {
        /* softmax-ablation: constant uniform P, written once. */
        for (uint i = tid; i < HPAD * BK; i += TPT)
            p_tile[i] = half(1.0f / (float)BK);
    }

    const float scale = rsqrt((float)args.head_dim);
    const uint n_tiles = (selected + BK - 1u) / BK;
    const ulong kv_base = (ulong)kv_head * args.head_dim;
    const ulong kv_token_stride = (ulong)args.kv_heads * args.head_dim;

    for (uint ktile = 0u; ktile < n_tiles; ktile++) {
        for (uint k = tid; k < BK; k += TPT) {
            const uint rank = ktile * BK + k;
            uint token = UINT_MAX;
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else if (rank < selected) {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            sel[k] = (token == UINT_MAX || token >= visible)
                ? -1 : (int)token;
        }
#define PROTO_GROUP_BARRIER() \
        threadgroup_barrier(mem_flags::mem_threadgroup)
        PROTO_GROUP_BARRIER();

        simdgroup_float8x8 sfrag[TK];
        for (uint ik = 0u; ik < TK; ik++)
            sfrag[ik] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint chunk = 0u; chunk < DCHUNKS; chunk++) {
            /* K is staged DIM-MAJOR (kvs[dim * BK + k]) so the QK operand
             * loads need no transpose. */
            for (uint i = tid; i < BK * (DC / 8u); i += TPT) {
                const uint k = i / (DC / 8u);
                const uint d8 = i % (DC / 8u);
                uint4 raw = 0u;
                if (SKIP != 1 && sel[k] >= 0) {
                    raw = *((device const uint4 *)(
                        key_cache +
                        (ulong)sel[k] * kv_token_stride + kv_base +
                        chunk * DC + d8 * 8u));
                }
                thread const ushort *elements =
                    (thread const ushort *)&raw;
                for (uint e = 0u; e < 8u; e++)
                    kvs[(d8 * 8u + e) * BK + k] = half(as_type<float>(
                        (uint)elements[e] << 16));
            }
            PROTO_GROUP_BARRIER();
            if (SKIP < 2) {
                for (uint dd8 = 0u; dd8 < DC / 8u; dd8++) {
                    simdgroup_half8x8 qf;
                    simdgroup_load(qf, qs + row_base * D + chunk * DC +
                                       dd8 * 8u, D, 0, false);
                    for (uint ik = 0u; ik < TK; ik++) {
                        simdgroup_half8x8 kt;
                        simdgroup_load(kt,
                                       kvs + (dd8 * 8u) * BK + ik * 8u,
                                       BK, 0, false);
                        simdgroup_multiply_accumulate(
                            sfrag[ik], qf, kt, sfrag[ik]);
                    }
                }
            }
            PROTO_GROUP_BARRIER();
        }

        if (SKIP == 5) {
            for (uint i = tid; i < HPAD * BK; i += TPT)
                p_tile[i] = half(1.0f / (float)BK);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (SKIP != 3 && SKIP != 5) {
            for (uint ik = 0u; ik < TK; ik++)
                simdgroup_store(sfrag[ik],
                                s_tile + row_base * BK + ik * 8u, BK, 0,
                                false);
        }
        if (SKIP != 5) simdgroup_barrier(mem_flags::mem_threadgroup);
        if (SKIP != 3 && SKIP != 5) {
            const uint row = (uint)lane / 4u;
            const uint col0 = ((uint)lane % 4u) * (BK / 4u);
            float local_max = -INFINITY;
            for (uint c = 0u; c < BK / 4u; c++) {
                float s = -INFINITY;
                if (sel[col0 + c] >= 0)
                    s = s_tile[(row_base + row) * BK + col0 + c] * scale;
                s_tile[(row_base + row) * BK + col0 + c] = s;
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    row_base + row == 0u) {
                    out[(ulong)999u * args.query_heads * args.head_dim +
                        col0 + c] = s;
                }
                local_max = max(local_max, s);
            }
            float row_max = local_max;
            for (uint j = 1u; j < 4u; j++)
                row_max = max(row_max,
                              simd_shuffle(local_max, row * 4u + j));
            const float old_max = stats[(row_base + row) * 2u];
            const float old_sum = stats[(row_base + row) * 2u + 1u];
            const float new_max = max(old_max, row_max);
            const bool grows = isfinite(old_max) && new_max > old_max;
            float local_sum = 0.0f;
            for (uint c = 0u; c < BK / 4u; c++) {
                const float s = s_tile[(row_base + row) * BK + col0 + c];
                const float p = isfinite(s) ? exp(s - new_max) : 0.0f;
                p_tile[(row_base + row) * BK + col0 + c] = half(p);
                local_sum += p;
            }
            float row_sum = local_sum;
            for (uint j = 1u; j < 4u; j++)
                row_sum += simd_shuffle(local_sum, row * 4u + j);
            if ((uint)lane % 4u == 0u) {
                const float factor = isfinite(old_max)
                    ? exp(old_max - new_max) : 1.0f;
                rfac[row_base + row] = factor;
                stats[(row_base + row) * 2u] = new_max;
                stats[(row_base + row) * 2u + 1u] =
                    old_sum * factor + row_sum;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (ORG == 1) {
                if ((uint)lane == 0u) vote[sg] = grows ? 1 : 0;
                PROTO_GROUP_BARRIER();
                if (vote[0] != 0 || vote[1] != 0) {
                    for (uint hidx = 0u; hidx < NSG; hidx++) {
                        if ((uint)sg == hidx) {
                            for (uint dd = 0u; dd < OD; dd++)
                                simdgroup_store(ofrag[dd], ores + dd * 8u,
                                                D, 0, false);
                        }
                        PROTO_GROUP_BARRIER();
                        if ((uint)sg == hidx) {
                            const uint r = (uint)lane / 4u;
                            const float factor = rfac[row_base + r];
                            const uint col = ((uint)lane % 4u) * 64u;
                            for (uint c = 0u; c < 64u; c++)
                                ores[r * D + col + c] *= factor;
                        }
                        PROTO_GROUP_BARRIER();
                        if ((uint)sg == hidx) {
                            for (uint dd = 0u; dd < OD; dd++)
                                simdgroup_load(ofrag[dd], ores + dd * 8u,
                                               D, 0, false);
                        }
                        PROTO_GROUP_BARRIER();
                    }
                }
                PROTO_GROUP_BARRIER();
            } else if (grows) {
                for (uint dd = 0u; dd < OD; dd++)
                    simdgroup_store(ofrag[dd], ores + dd * 8u, D, 0, false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                const uint r = (uint)lane / 4u;
                const float factor = rfac[row_base + r];
                const uint col = ((uint)lane % 4u) * 64u;
                for (uint c = 0u; c < 64u; c++)
                    ores[r * D + col + c] *= factor;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint dd = 0u; dd < OD; dd++)
                    simdgroup_load(ofrag[dd], ores + dd * 8u, D, 0, false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint chunk = 0u; chunk < DCHUNKS; chunk++) {
            for (uint i = tid; i < BK * (DC / 8u); i += TPT) {
                const uint k = i / (DC / 8u);
                const uint d8 = i % (DC / 8u);
                uint4 raw = 0u;
                if (SKIP != 1 && sel[k] >= 0) {
                    raw = *((device const uint4 *)(
                        value_cache +
                        (ulong)sel[k] * kv_token_stride + kv_base +
                        chunk * DC + d8 * 8u));
                }
                thread const ushort *elements =
                    (thread const ushort *)&raw;
                for (uint e = 0u; e < 8u; e++)
                    kvs[k * DC + d8 * 8u + e] = half(as_type<float>(
                        (uint)elements[e] << 16));
            }
            PROTO_GROUP_BARRIER();
            if (SKIP < 2) {
                for (uint ik = 0u; ik < TK; ik++) {
                    simdgroup_half8x8 pf;
                    simdgroup_load(pf, p_tile + row_base * BK + ik * 8u,
                                   BK, 0, false);
                    for (uint dd = 0u; dd < DC / 8u; dd++) {
                        simdgroup_half8x8 vf;
                        simdgroup_load(vf, kvs + ik * 8u * DC + dd * 8u,
                                       DC, 0, false);
                        simdgroup_multiply_accumulate(
                            ofrag[chunk * (DC / 8u) + dd], pf, vf,
                            ofrag[chunk * (DC / 8u) + dd]);
                    }
                }
            }
            PROTO_GROUP_BARRIER();
        }
    }

    /* Emit through the overlay tile with normalization and sigmoid gate. */
    if (ORG == 1) {
        for (uint hidx = 0u; hidx < NSG; hidx++) {
            if ((uint)sg == hidx) {
                for (uint dd = 0u; dd < OD; dd++)
                    simdgroup_store(ofrag[dd], ores + dd * 8u, D, 0, false);
            }
            PROTO_GROUP_BARRIER();
            if ((uint)sg == hidx) {
                for (uint i = (uint)lane; i < valid_rows * (D / 8u);
                     i += 32u) {
                    const uint r = i / (D / 8u);
                    const uint d8 = i % (D / 8u);
                    const uint head = kv_head * GQA + row_base + r;
                    const ulong base = ((ulong)query * args.query_heads +
                                        head) * args.head_dim;
                    const float m = stats[(row_base + r) * 2u];
                    const float s = stats[(row_base + r) * 2u + 1u];
                    const float inv =
                        isfinite(m) && s > 0.0f ? 1.0f / s : 0.0f;
                    for (uint e = 0u; e < 8u; e++) {
                        const float g = 1.0f /
                            (1.0f + exp(-raw_gate[base + d8 * 8u + e]));
                        out[base + d8 * 8u + e] =
                            ores[r * D + d8 * 8u + e] * inv * g;
                    }
                }
            }
            PROTO_GROUP_BARRIER();
        }
    } else {
        for (uint dd = 0u; dd < OD; dd++)
            simdgroup_store(ofrag[dd], ores + dd * 8u, D, 0, false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < valid_rows * (D / 8u); i += TPT) {
            const uint r = i / (D / 8u);
            const uint d8 = i % (D / 8u);
            const uint head = kv_head * GQA + half_idx * 8u + r;
            const ulong base = ((ulong)query * args.query_heads + head) *
                args.head_dim;
            const float m = stats[(row_base + r) * 2u];
            const float s = stats[(row_base + r) * 2u + 1u];
            const float inv = isfinite(m) && s > 0.0f ? 1.0f / s : 0.0f;
            for (uint e = 0u; e < 8u; e++) {
                const float g =
                    1.0f / (1.0f + exp(-raw_gate[base + d8 * 8u + e]));
                out[base + d8 * 8u + e] =
                    ores[r * D + d8 * 8u + e] * inv * g;
            }
        }
    }
}


/* Role-split organization: one 192-thread threadgroup (six simdgroups)
 * per (query, KV head).  sg 0/1 are SCORE simdgroups (eight head rows
 * each, full-width QK plus the online softmax, no O accumulators);
 * sg 2..5 are PV simdgroups owning eight rows x 128 dims each (16 O
 * fragments, half the accumulator registers of the lockstep design).
 * Halving per-thread accumulators raises resident simdgroups so the
 * matrix units can be fed at a far higher rate. */
template <int BK, int DC>
static inline void gqa_mma_rolesplit_body(
        constant proto_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        threadgroup uchar *tgm,
        uint2 group,
        ushort lane,
        ushort sg) {
    constexpr uint GQA2 = 12u;
    constexpr uint D2 = 256u;
    constexpr uint DCHUNKS2 = D2 / DC;
    constexpr uint TK2 = BK / 8u;
    constexpr uint OD2 = (D2 / 2u) / 8u;   /* per PV simdgroup */
    constexpr uint TPT2 = 192u;

    threadgroup half *qs = (threadgroup half *)tgm;            /* 16*D  */
    threadgroup half *kvs = qs + 16u * D2;                     /* BK*DC */
    threadgroup float *s_tile = (threadgroup float *)(kvs + BK * DC);
    threadgroup half *p_tile = (threadgroup half *)(s_tile + 16u * BK);
    threadgroup int *sel = (threadgroup int *)(p_tile + 16u * BK);
    threadgroup float *stats = (threadgroup float *)(sel + BK);
    threadgroup float *rfac = stats + 16u * 2u;
    threadgroup int *vote = (threadgroup int *)(rfac + 16u);
    /* One explicit 8x128 float O block shared in four phases. */
    threadgroup float *oblock = (threadgroup float *)(vote + 2u);

    const uint query = group.x;
    if (query >= args.queries) return;
    if (group.y >= args.kv_heads) return;
    const uint kv_head = group.y;
    const uint lane32 = (uint)lane;
    const uint tid = (uint)sg * 32u + lane32;
    const bool is_score = sg < 2u;
    const uint row_half = is_score ? (uint)sg : ((uint)sg - 2u) / 2u;
    const uint row_base = row_half * 8u;
    const uint dim_half = ((uint)sg - 2u) % 2u;  /* PV only */

    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count =
        min(min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;

    if (selected == 0u) {
        for (uint i = tid; i < GQA2 * D2; i += TPT2)
            out[((ulong)query * args.query_heads + kv_head * GQA2 + i / D2) *
                args.head_dim + (i % D2)] = 0.0f;
        return;
    }

    for (uint i = tid; i < 16u * (D2 / 8u); i += TPT2) {
        const uint row = i / (D2 / 8u);
        const uint d8 = i % (D2 / 8u);
        float4 f0 = 0.0f;
        float4 f1 = 0.0f;
        if (row < GQA2) {
            const device const float *qrow = q +
                ((ulong)query * args.query_heads + kv_head * GQA2 + row) *
                    args.head_dim +
                d8 * 8u;
            f0 = *((device const float4 *)qrow);
            f1 = *((device const float4 *)(qrow + 4u));
        }
        qs[row * D2 + d8 * 8u + 0u] = half(f0.x);
        qs[row * D2 + d8 * 8u + 1u] = half(f0.y);
        qs[row * D2 + d8 * 8u + 2u] = half(f0.z);
        qs[row * D2 + d8 * 8u + 3u] = half(f0.w);
        qs[row * D2 + d8 * 8u + 4u] = half(f1.x);
        qs[row * D2 + d8 * 8u + 5u] = half(f1.y);
        qs[row * D2 + d8 * 8u + 6u] = half(f1.z);
        qs[row * D2 + d8 * 8u + 7u] = half(f1.w);
    }
    for (uint r = tid; r < 16u; r += TPT2) {
        stats[r * 2u] = -INFINITY;
        stats[r * 2u + 1u] = 0.0f;
    }
    if (args.debug != 0u && query == 7u && kv_head == 0u && tid < 16u)
        out[(ulong)998u * args.query_heads * args.head_dim + tid] =
            (float)qs[tid];

    simdgroup_float8x8 ofrag[OD2];
    if (!is_score)
        for (uint dd = 0u; dd < OD2; dd++)
            ofrag[dd] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const float scale = rsqrt((float)args.head_dim);
    const uint n_tiles = (selected + BK - 1u) / BK;
    const ulong kv_base = (ulong)kv_head * args.head_dim;
    const ulong kv_token_stride = (ulong)args.kv_heads * args.head_dim;

    for (uint ktile = 0u; ktile < n_tiles; ktile++) {
        for (uint k = tid; k < BK; k += TPT2) {
            const uint rank = ktile * BK + k;
            uint token = UINT_MAX;
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else if (rank < selected) {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            sel[k] = (token == UINT_MAX || token >= visible)
                ? -1 : (int)token;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* QK: every simdgroup stages the shared dim-major K tile; only
         * the score simdgroups run the products. */
        {
            simdgroup_float8x8 sfrag[TK2];
            if (is_score)
                for (uint ik = 0u; ik < TK2; ik++)
                    sfrag[ik] =
                        make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint chunk = 0u; chunk < DCHUNKS2; chunk++) {
                for (uint i = tid; i < BK * (DC / 8u); i += TPT2) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            key_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[(d8 * 8u + e) * BK + k] = half(as_type<float>(
                            (uint)elements[e] << 16));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    chunk == 0u && tid < 8u) {
                    out[(ulong)998u * args.query_heads * args.head_dim +
                        16u + tid] = (float)kvs[tid * BK + 0u];
                    out[(ulong)998u * args.query_heads * args.head_dim +
                        24u + tid] = (float)kvs[0u * BK + tid];
                }
                for (uint dd8 = 0u; dd8 < DC / 8u && is_score; dd8++) {
                    simdgroup_half8x8 qf;
                    simdgroup_load(qf, qs + row_base * D2 + chunk * DC +
                                       dd8 * 8u, D2, 0, false);
                    for (uint ik = 0u; ik < TK2; ik++) {
                        simdgroup_half8x8 kt;
                        simdgroup_load(kt, kvs + (dd8 * 8u) * BK + ik * 8u,
                                       BK, 0, false);
                        simdgroup_multiply_accumulate(
                            sfrag[ik], qf, kt, sfrag[ik]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (is_score)
                for (uint ik = 0u; ik < TK2; ik++)
                    simdgroup_store(sfrag[ik],
                                    s_tile + row_base * BK + ik * 8u,
                                    BK, 0, false);
            if (args.debug != 0u && query == 7u && kv_head == 0u &&
                sg == 0u) {
                simdgroup_store(sfrag[0u], oblock, 8u, 0, false);
                simdgroup_store(sfrag[1u], oblock + 64u, 8u, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (args.debug != 0u && query == 7u && kv_head == 0u &&
            tid < 128u) {
            out[(ulong)997u * args.query_heads * args.head_dim + tid] =
                oblock[tid];
        }
        if (args.debug != 0u && query == 7u && kv_head == 0u &&
            tid < 1024u) {
            out[(ulong)995u * args.query_heads * args.head_dim + tid] =
                s_tile[tid];
        }
        if (is_score) {
            const uint row = lane32 / 4u;
            const uint col0 = (lane32 % 4u) * (BK / 4u);
            float local_max = -INFINITY;
            for (uint c = 0u; c < BK / 4u; c++) {
                float s = -INFINITY;
                if (sel[col0 + c] >= 0)
                    s = s_tile[(row_base + row) * BK + col0 + c] * scale;
                s_tile[(row_base + row) * BK + col0 + c] = s;
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    row_base + row == 0u) {
                    out[(ulong)999u * args.query_heads * args.head_dim +
                        col0 + c] = s;
                }
                local_max = max(local_max, s);
            }
            float row_max = local_max;
            for (uint j = 1u; j < 4u; j++)
                row_max = max(row_max,
                              simd_shuffle(local_max, row * 4u + j));
            const float old_max = stats[(row_base + row) * 2u];
            const float old_sum = stats[(row_base + row) * 2u + 1u];
            const float new_max = max(old_max, row_max);
            const bool grows = isfinite(old_max) && new_max > old_max;
            float local_sum = 0.0f;
            for (uint c = 0u; c < BK / 4u; c++) {
                const float s = s_tile[(row_base + row) * BK + col0 + c];
                const float pp = isfinite(s) ? exp(s - new_max) : 0.0f;
                p_tile[(row_base + row) * BK + col0 + c] = half(pp);
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    row_base + row == 0u) {
                    out[(ulong)999u * args.query_heads * args.head_dim +
                        256u + col0 + c] = pp;
                }
                local_sum += pp;
            }
            float row_sum = local_sum;
            for (uint j = 1u; j < 4u; j++)
                row_sum += simd_shuffle(local_sum, row * 4u + j);
            if (lane32 % 4u == 0u) {
                const float factor = isfinite(old_max)
                    ? exp(old_max - new_max) : 1.0f;
                rfac[row_base + row] = factor;
                stats[(row_base + row) * 2u] = new_max;
                stats[(row_base + row) * 2u + 1u] =
                    old_sum * factor + row_sum;
            }
            if (lane32 == 0u) vote[sg] = grows ? 1 : 0;
            if (args.debug != 0u && query == 7u && kv_head == 0u &&
                sg == 0u && lane32 % 4u == 0u) {
                out[(ulong)994u * args.query_heads * args.head_dim +
                    (row_base + row)] = row_max;
                out[(ulong)994u * args.query_heads * args.head_dim +
                    16u + (row_base + row)] =
                    stats[(row_base + row) * 2u + 1u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vote[0] != 0 || vote[1] != 0) {
            for (uint phase = 0u; phase < 4u; phase++) {
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD2; dd++)
                        simdgroup_store(ofrag[dd], oblock + dd * 8u,
                                        128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    const uint r = lane32 / 4u;
                    const float factor = rfac[row_base + r];
                    const uint col = (lane32 % 4u) * 32u;
                    for (uint c = 0u; c < 32u; c++)
                        oblock[r * 128u + col + c] *= factor;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD2; dd++)
                        simdgroup_load(ofrag[dd], oblock + dd * 8u,
                                       128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* PV: every simdgroup stages the shared V tile; each PV
         * simdgroup runs its dim half's products. */
        {
            for (uint chunk = 0u; chunk < DCHUNKS2; chunk++) {
                const bool pv_chunk =
                    !is_score &&
                    chunk >= dim_half * (DCHUNKS2 / 2u) &&
                    chunk < (dim_half + 1u) * (DCHUNKS2 / 2u);
                for (uint i = tid; i < BK * (DC / 8u); i += TPT2) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            value_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[k * DC + d8 * 8u + e] = half(as_type<float>(
                            (uint)elements[e] << 16));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (pv_chunk) {
                    for (uint ik = 0u; ik < TK2; ik++) {
                        simdgroup_half8x8 pf;
                        simdgroup_load(pf,
                                       p_tile + row_base * BK + ik * 8u,
                                       BK, 0, false);
                        for (uint dd = 0u; dd < DC / 8u; dd++) {
                            simdgroup_half8x8 vf;
                            simdgroup_load(vf, kvs + ik * 8u * DC + dd * 8u,
                                           DC, 0, false);
                            simdgroup_multiply_accumulate(
                                ofrag[(chunk % (DCHUNKS2 / 2u)) *
                                          (DC / 8u) +
                                      dd],
                                pf, vf,
                                ofrag[(chunk % (DCHUNKS2 / 2u)) *
                                          (DC / 8u) +
                                      dd]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    /* Emit: each PV simdgroup normalizes and gates its 8x128 block;
     * row half 1 writes only its GQA2-8 valid heads. */
    for (uint phase = 0u; phase < 4u; phase++) {
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA2 - 8u;
            for (uint dd = 0u; dd < OD2; dd++)
                simdgroup_store(ofrag[dd], oblock + dd * 8u, 128u, 0,
                                false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA2 - 8u;
            for (uint i = lane32; i < emit_rows * (128u / 8u); i += 32u) {
                const uint r = i / (128u / 8u);
                const uint d8 = i % (128u / 8u);
                const uint head = kv_head * GQA2 + row_base + r;
                const ulong base =
                    ((ulong)query * args.query_heads + head) *
                    args.head_dim;
                const float m = stats[(row_base + r) * 2u];
                const float s = stats[(row_base + r) * 2u + 1u];
                const float inv =
                    isfinite(m) && s > 0.0f ? 1.0f / s : 0.0f;
                for (uint e = 0u; e < 8u; e++) {
                    const uint d = dim_half * 128u + d8 * 8u + e;
                    const float g =
                        1.0f / (1.0f + exp(-raw_gate[base + d]));
                    out[base + d] =
                        oblock[r * 128u + d8 * 8u + e] * inv * g;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* ===== Metal-4 TensorOps (matmul2d) gathered-attention variants =====
 *
 * One 128-thread threadgroup (4 simdgroups) owns one (query, KV head)
 * pair: all 12 query heads (padded to M=16) share every gathered K/V
 * tile, and QK^T and PV run as threadgroup-staged cooperative-tensor
 * matmuls.  The historical F16-MMA failure was the F16 P-tile rounding
 * (the Q-split probe showed Q's own F16 rounding is NOT binding), so
 * the SPLIT variant keeps the P/V operands fp32 (exact products, exact
 * fp32 accumulate per the f32stage finding) and stages only the Q/K
 * QK operands as half (K is value-exact BF16->F16; Q rounds once per
 * group, measured non-binding).  The EXACT variant stages every
 * operand fp32.  Layouts follow the t2d probe: LEFT [m][k] k-contig,
 * RIGHT [n][k] k-contig, dest stored through a (N, M) extents tensor
 * with strides {1, N} = plain [m][n] row-major; register epilogues
 * use get_multidimensional_index with ids[0] = n, ids[1] = m.
 *
 * Geometry: BK=32 tokens per tile (top_k 512 x ratio 4 -> 64 tiles at
 * the full 2048-token budget), PV split into two N=128 halves so the
 * fp32 V chunk (16 KB) fits the 32 KB threadgroup limit; the S and P
 * tiles share one buffer element-for-element (softmax reads S and
 * overwrites P in place), and in SPLIT the V chunks overlay the dead
 * K tile. */
constant constexpr int T2D_D = 256;
constant constexpr int T2D_NK = 32;
constant constexpr int T2D_NH = 128;
constant constexpr int T2D_T = 128;

/* TILEK: the QK contraction chunk (32 = the probe-proven small step,
 * 256 = one run over the resident tile letting the op tile internally).
 * ABLATE: 0 full, 1 skip QK runs, 2 skip PV runs, 3 skip device gathers
 * (staging writes constants, barriers kept). */
template <bool EXACT, int TILEK, int ABLATE>
static inline void gqa_t2d_body(
        constant proto_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        threadgroup uchar *tgm,
        uint2 group,
        uint tid) {
    constexpr uint GQA = 12u;
    constexpr uint M = 16u;
    constexpr uint BK = 32u;
    constexpr uint NK = (uint)T2D_NK;
    constexpr uint NH = (uint)T2D_NH;
    constexpr uint TPT = (uint)T2D_T;

    /* Threadgroup layout. */
    threadgroup uchar *cursor = tgm;
    threadgroup half *qt = nullptr;    /* M*D half, SPLIT only        */
    threadgroup half *kt = nullptr;    /* BK*D half, SPLIT only       */
    threadgroup float *qc[2] = {nullptr, nullptr}; /* M*NK f32, EXACT */
    threadgroup float *kc[2] = {nullptr, nullptr}; /* BK*NK f32, EXACT */
    threadgroup float *vb = nullptr;   /* NH*NK f32 V chunk           */
    threadgroup float *pt = nullptr;   /* M*BK f32, S then P in place */
    threadgroup int *sel = nullptr;    /* BK                          */
    threadgroup float *stats = nullptr;/* M*2                         */
    threadgroup float *rfac = nullptr; /* M                           */
    threadgroup int *vote = nullptr;   /* M                           */
    if constexpr (EXACT) {
        qc[0] = (threadgroup float *)cursor;
        qc[1] = qc[0] + M * (uint)TILEK;
        kc[0] = qc[1] + M * (uint)TILEK;
        kc[1] = kc[0] + BK * (uint)TILEK;
        vb = kc[1] + BK * (uint)TILEK;
        cursor = (threadgroup uchar *)(vb + NH * NK);
    } else {
        qt = (threadgroup half *)cursor;
        kt = qt + M * T2D_D;
        /* The fp32 V chunks (16 KB) overlay the K tile's region: K is
         * dead once QK finishes, and the sizes match exactly. */
        vb = (threadgroup float *)kt;
        cursor = (threadgroup uchar *)(kt + BK * T2D_D);
    }
    pt = (threadgroup float *)cursor;
    sel = (threadgroup int *)(pt + M * BK);
    stats = (threadgroup float *)(sel + BK);
    rfac = stats + M * 2u;
    vote = (threadgroup int *)(rfac + M);

    const uint query = group.x;
    if (query >= args.queries) return;
    if (group.y >= args.kv_heads) return;
    const uint kv_head = group.y;

    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count =
        min(min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;

    if (selected == 0u) {
        for (uint i = tid; i < GQA * T2D_D; i += TPT)
            out[((ulong)query * args.query_heads + kv_head * GQA + i / T2D_D)
                * args.head_dim + (i % T2D_D)] = 0.0f;
        return;
    }

    const ulong kv_base = (ulong)kv_head * args.head_dim;
    const ulong kv_token_stride = (ulong)args.kv_heads * args.head_dim;
    const float scale = rsqrt((float)args.head_dim);

    /* Stage Q once per group (SPLIT: resident half tile; EXACT: chunks
     * are restaged inside the d-chunk loop). */
    if (!EXACT) {
        for (uint i = tid; i < M * (uint)T2D_D; i += TPT) {
            const uint m = i / (uint)T2D_D, d = i % (uint)T2D_D;
            float v = 0.0f;
            if (m < GQA) {
                v = q[((ulong)query * args.query_heads + kv_head * GQA + m) *
                          args.head_dim +
                      d];
            }
            qt[i] = half(v);
        }
    }
    for (uint r = tid; r < M; r += TPT) {
        stats[r * 2u] = -INFINITY;
        stats[r * 2u + 1u] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* QK matmul: S(M,BK) = Q(M,D) @ K(BK,D)^T, d-chunked by NK. */
    matmul2d<
        matmul2d_descriptor(
            M, BK, TILEK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_qk;
    /* PV matmuls: O_half(M,NH) = P(M,BK) @ V(BK,NH)^T, one chunk step. */
    matmul2d<
        matmul2d_descriptor(
            M, NH, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm_pv;

    threadgroup float *stage_qk_chunk = nullptr; /* EXACT chunk tensors */
    auto tQ_sp = tensor(qt, dextents<int32_t, 2>(T2D_D, M));
    auto tK_sp = tensor(kt, dextents<int32_t, 2>(T2D_D, BK));
    auto tP = tensor(pt, dextents<int32_t, 2>(BK, M));
    auto tV = tensor(vb, dextents<int32_t, 2>(NK, NH));
    (void)stage_qk_chunk;

    auto stage_chunk = [&](const uint d0, const uint buf) {
        for (uint i = tid; i < M * (uint)TILEK; i += TPT) {
            const uint m = i / (uint)TILEK, k = i % (uint)TILEK;
            qc[buf][i] = m < GQA
                ? q[((ulong)query * args.query_heads + kv_head * GQA + m) *
                        args.head_dim +
                    d0 + k]
                : 0.0f;
        }
        for (uint i = tid; i < BK * (uint)TILEK; i += TPT) {
            const uint t = i / (uint)TILEK, k = i % (uint)TILEK;
            float v = 0.0f;
            if (sel[t] >= 0 && ABLATE != 3)
                v = qwen4_bf16_to_f32(
                    key_cache[(ulong)sel[t] * kv_token_stride + kv_base +
                              d0 + k]);
            kc[buf][i] = v;
        }
    };

    /* Two persistent register destinations for the O halves. */
    auto cO0 = mm_pv.template get_destination_cooperative_tensor<
        decltype(tV), decltype(tP), float>();
    auto cO1 = mm_pv.template get_destination_cooperative_tensor<
        decltype(tV), decltype(tP), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cO0.get_capacity(); ++i) {
        if (cO0.is_valid_element(i)) cO0[i] = 0.0f;
        if (cO1.is_valid_element(i)) cO1[i] = 0.0f;
    }

    const uint n_tiles = (selected + BK - 1u) / BK;
    for (uint ktile = 0u; ktile < n_tiles; ktile++) {
        /* Resolve this tile's ranks to tokens (-1 invalid). */
        for (uint k = tid; k < BK; k += TPT) {
            const uint rank = ktile * BK + k;
            uint token = UINT_MAX;
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else if (rank < selected) {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            sel[k] = (token == UINT_MAX || token >= visible)
                ? -1 : (int)token;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* QK^T. */
        constexpr uint QKC = (uint)T2D_D / (uint)TILEK;
        if constexpr (EXACT) {
            auto tQ0 = tensor(qc[0], dextents<int32_t, 2>(TILEK, M));
            auto tQ1 = tensor(qc[1], dextents<int32_t, 2>(TILEK, M));
            auto tK0 = tensor(kc[0], dextents<int32_t, 2>(TILEK, BK));
            auto tK1 = tensor(kc[1], dextents<int32_t, 2>(TILEK, BK));
            auto cS = mm_qk.template get_destination_cooperative_tensor<
                decltype(tK0), decltype(tQ0), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cS.get_capacity(); ++i)
                if (cS.is_valid_element(i)) cS[i] = 0.0f;
            stage_chunk(0u, 0u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0u; c < QKC; ++c) {
                if (c + 1u < QKC)
                    stage_chunk((c + 1u) * (uint)TILEK, (c + 1u) & 1u);
                if (c & 1u) {
                    auto sQ = tQ1.slice(0, 0);
                    auto sK = tK1.slice(0, 0);
                    if (ABLATE != 1) mm_qk.run(sQ, sK, cS);
                } else {
                    auto sQ = tQ0.slice(0, 0);
                    auto sK = tK0.slice(0, 0);
                    if (ABLATE != 1) mm_qk.run(sQ, sK, cS);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            auto tS = tensor(pt, dextents<int32_t, 2>(BK, M),
                             array<int, 2>({1, BK}));
            cS.store(tS);
        } else {
            /* Gather the whole K tile once (half, dim-contiguous rows:
             * the resident (D, BK) tensor reads [t*D + d]).  128-bit
             * vector loads (8 ushorts) keep the gather
             * instruction-count at an eighth of the scalar form. */
            for (uint i = tid; i < BK * ((uint)T2D_D / 8u); i += TPT) {
                const uint t = i / ((uint)T2D_D / 8u);
                const uint d8 = i % ((uint)T2D_D / 8u);
                uint4 raw = 0u;
                if (sel[t] >= 0 && ABLATE != 3)
                    raw = *((device const uint4 *)(
                        key_cache + (ulong)sel[t] * kv_token_stride +
                        kv_base + d8 * 8u));
                thread const ushort *e =
                    (thread const ushort *)&raw;
                uint4 packed;
                thread ushort *pk = (thread ushort *)&packed;
                for (uint e8 = 0u; e8 < 8u; e8++) {
                    const half h = half(qwen4_bf16_to_f32(e[e8]));
                    pk[e8] = *((thread const ushort *)&h);
                }
                *((threadgroup uint4 *)(kt + t * (uint)T2D_D + d8 * 8u)) =
                    packed;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto cS = mm_qk.template get_destination_cooperative_tensor<
                decltype(tK_sp), decltype(tQ_sp), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cS.get_capacity(); ++i)
                if (cS.is_valid_element(i)) cS[i] = 0.0f;
            for (uint c = 0u; c < QKC; ++c) {
                auto sQ = tQ_sp.slice(c * (uint)TILEK, 0);
                auto sK = tK_sp.slice(c * (uint)TILEK, 0);
                if (ABLATE != 1) mm_qk.run(sQ, sK, cS);
            }
            auto tS = tensor(pt, dextents<int32_t, 2>(BK, M),
                             array<int, 2>({1, BK}));
            cS.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Online softmax, 8 threads per row of 32 columns.  Rows live
         * inside one simdgroup as 8-lane groups, so the row reductions
         * are 7 simd_shuffles.  S is scaled (invalid -> -inf) and then
         * overwritten by P in place. */
        {
            const uint row = tid >> 3;
            const uint sub = tid & 7u;
            const uint col0 = sub * 4u;
            const uint lane_base = (row & 3u) * 8u;
            float lmax = -INFINITY;
            for (uint c = 0u; c < 4u; c++) {
                const uint col = col0 + c;
                float s = -INFINITY;
                if (sel[col] >= 0) s = pt[row * BK + col] * scale;
                pt[row * BK + col] = s;
                lmax = max(lmax, s);
            }
            float rmax = lmax;
            for (uint j = 1u; j < 8u; j++)
                rmax = max(rmax, simd_shuffle(lmax, lane_base + j));
            const float old_max = stats[row * 2u];
            const float old_sum = stats[row * 2u + 1u];
            const float new_max = max(old_max, rmax);
            float lsum = 0.0f;
            for (uint c = 0u; c < 4u; c++) {
                const float s = pt[row * BK + col0 + c];
                const float p = isfinite(s) ? exp(s - new_max) : 0.0f;
                pt[row * BK + col0 + c] = p;
                lsum += p;
            }
            float rsum = lsum;
            for (uint j = 1u; j < 8u; j++)
                rsum += simd_shuffle(lsum, lane_base + j);
            if (sub == 0u) {
                const float factor = isfinite(old_max)
                    ? exp(old_max - new_max) : 1.0f;
                rfac[row] = factor;
                vote[row] =
                    (isfinite(old_max) && new_max > old_max) ? 1 : 0;
                stats[row * 2u] = new_max;
                stats[row * 2u + 1u] = old_sum * factor + rsum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Lazy register rescale of the running O when a row max grew. */
        bool any_vote = false;
        for (uint r = 0u; r < M; r++) any_vote = any_vote || vote[r] != 0;
        if (any_vote) {
            #pragma unroll
            for (uint16_t i = 0; i < cO0.get_capacity(); ++i) {
                if (cO0.is_valid_element(i)) {
                    const auto ids = cO0.get_multidimensional_index(i);
                    cO0[i] *= rfac[ids[1]];
                }
                if (cO1.is_valid_element(i)) {
                    const auto ids = cO1.get_multidimensional_index(i);
                    cO1[i] *= rfac[ids[1]];
                }
            }
        }

        /* PV: P is resident; stage each half's V chunk (fp32, [d][t]
         * for the NT operand) with 128-bit vector loads: consecutive
         * threads read consecutive 8-dim blocks of one token. */
        auto sP = tP.slice(0, 0);
        for (uint hf = 0u; hf < 2u; hf++) {
            for (uint i = tid; i < BK * (NH / 8u); i += TPT) {
                const uint t = i / (NH / 8u);
                const uint d8 = i % (NH / 8u);
                const int tok = sel[t];
                uint4 raw = 0u;
                if (tok >= 0 && ABLATE != 3)
                    raw = *((device const uint4 *)(
                        value_cache + (ulong)tok * kv_token_stride +
                        kv_base + hf * NH + d8 * 8u));
                thread const ushort *e =
                    (thread const ushort *)&raw;
                for (uint e8 = 0u; e8 < 8u; e8++)
                    vb[(d8 * 8u + e8) * NK + t] =
                        qwen4_bf16_to_f32(e[e8]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sV = tV.slice(0, 0);
            if (ABLATE != 2) {
                if (hf == 0u) mm_pv.run(sP, sV, cO0);
                else mm_pv.run(sP, sV, cO1);
            }
        }
    }

    /* Emit from registers: normalize, gate, write the 12 head rows. */
    #pragma unroll
    for (uint16_t i = 0; i < cO0.get_capacity(); ++i) {
        if (!cO0.is_valid_element(i)) continue;
        const auto ids = cO0.get_multidimensional_index(i);
        const uint m = (uint)ids[1], d = (uint)ids[0];
        if (m >= GQA) continue;
        const ulong base =
            ((ulong)query * args.query_heads + kv_head * GQA + m) *
                args.head_dim +
            d;
        const float mx = stats[m * 2u], sm = stats[m * 2u + 1u];
        const float inv = isfinite(mx) && sm > 0.0f ? 1.0f / sm : 0.0f;
        out[base] = cO0[i] * inv /
            (1.0f + exp(-raw_gate[base]));
    }
    #pragma unroll
    for (uint16_t i = 0; i < cO1.get_capacity(); ++i) {
        if (!cO1.is_valid_element(i)) continue;
        const auto ids = cO1.get_multidimensional_index(i);
        const uint m = (uint)ids[1], d = (uint)ids[0];
        if (m >= GQA) continue;
        const ulong base =
            ((ulong)query * args.query_heads + kv_head * GQA + m) *
                args.head_dim +
            NH + d;
        const float mx = stats[m * 2u], sm = stats[m * 2u + 1u];
        const float inv = isfinite(mx) && sm > 0.0f ? 1.0f / sm : 0.0f;
        out[base] = cO1[i] * inv /
            (1.0f + exp(-raw_gate[base]));
    }
}

#define INSTANTIATE_T2D(name, exact, tilek, ablate) \
    kernel void name( \
            constant proto_args &args [[buffer(0)]], \
            device const float *q [[buffer(1)]], \
            device const float *raw_gate [[buffer(2)]], \
            device const ushort *key_cache [[buffer(3)]], \
            device const ushort *value_cache [[buffer(4)]], \
            device const uint *selected_blocks [[buffer(5)]], \
            device const uint *selected_counts [[buffer(6)]], \
            device const uint *visible_tokens [[buffer(7)]], \
            device float *out [[buffer(8)]], \
            threadgroup uchar *tgm [[threadgroup(0)]], \
            uint2 group [[threadgroup_position_in_grid]], \
            uint tid [[thread_index_in_threadgroup]]) { \
        gqa_t2d_body<exact, tilek, ablate>( \
            args, q, raw_gate, key_cache, value_cache, selected_blocks, \
            selected_counts, visible_tokens, out, tgm, group, tid); \
    }

INSTANTIATE_T2D(gqa_t2d_split, false, 32, 0)
INSTANTIATE_T2D(gqa_t2d_exact, true, 32, 0)
INSTANTIATE_T2D(gqa_t2d_split_tk256, false, 256, 0)
INSTANTIATE_T2D(gqa_t2d_exact_tk256, true, 256, 0)
INSTANTIATE_T2D(gqa_t2d_split_noqk, false, 32, 1)
INSTANTIATE_T2D(gqa_t2d_split_nopv, false, 32, 2)
INSTANTIATE_T2D(gqa_t2d_split_nogather, false, 32, 3)
INSTANTIATE_T2D(gqa_t2d_split_tk256_nogather, false, 256, 3)

#define INSTANTIATE_PROTO(name, org, bk, dc, skip) \
    kernel void name( \
            constant proto_args &args [[buffer(0)]], \
            device const float *q [[buffer(1)]], \
            device const float *raw_gate [[buffer(2)]], \
            device const ushort *key_cache [[buffer(3)]], \
            device const ushort *value_cache [[buffer(4)]], \
            device const uint *selected_blocks [[buffer(5)]], \
            device const uint *selected_counts [[buffer(6)]], \
            device const uint *visible_tokens [[buffer(7)]], \
            device float *out [[buffer(8)]], \
            threadgroup uchar *tgm [[threadgroup(0)]], \
            uint2 group [[threadgroup_position_in_grid]], \
            ushort lane [[thread_index_in_simdgroup]], \
            ushort sg [[simdgroup_index_in_threadgroup]]) { \
        gqa_mma_proto_body<org, bk, dc, skip>( \
            args, q, raw_gate, key_cache, value_cache, selected_blocks, \
            selected_counts, visible_tokens, out, tgm, group, lane, sg); \
    }

/* Variants: ORG/BK/DC/Skip.  skip: 0 full, 1 no staging loads,
 * 2 no matrix products, 3 no softmax roundtrip. */
INSTANTIATE_PROTO(gqa_ps_b64_d64_full, 0, 64, 64, 0)
INSTANTIATE_PROTO(gqa_ls_b64_d64_full, 1, 64, 64, 0)
INSTANTIATE_PROTO(gqa_ls_b64_d64_nostage, 1, 64, 64, 1)
INSTANTIATE_PROTO(gqa_ls_b64_d64_nomad, 1, 64, 64, 2)
INSTANTIATE_PROTO(gqa_ls_b64_d64_nosoftmax, 1, 64, 64, 3)
INSTANTIATE_PROTO(gqa_ls_b128_d32_full, 1, 128, 32, 0)
INSTANTIATE_PROTO(gqa_ps_b128_d32_full, 0, 128, 32, 0)
INSTANTIATE_PROTO(gqa_ls_b128_d64_full, 1, 128, 64, 0)
INSTANTIATE_PROTO(gqa_ls_b64_d64_constp_tile, 1, 64, 64, 5)

#define INSTANTIATE_ROLESPLIT(name, bk, dc) \
    kernel void name( \
            constant proto_args &args [[buffer(0)]], \
            device const float *q [[buffer(1)]], \
            device const float *raw_gate [[buffer(2)]], \
            device const ushort *key_cache [[buffer(3)]], \
            device const ushort *value_cache [[buffer(4)]], \
            device const uint *selected_blocks [[buffer(5)]], \
            device const uint *selected_counts [[buffer(6)]], \
            device const uint *visible_tokens [[buffer(7)]], \
            device float *out [[buffer(8)]], \
            threadgroup uchar *tgm [[threadgroup(0)]], \
            uint2 group [[threadgroup_position_in_grid]], \
            ushort lane [[thread_index_in_simdgroup]], \
            ushort sg [[simdgroup_index_in_threadgroup]]) { \
        gqa_mma_rolesplit_body<bk, dc>( \
            args, q, raw_gate, key_cache, value_cache, selected_blocks, \
            selected_counts, visible_tokens, out, tgm, group, lane, sg); \
    }

INSTANTIATE_ROLESPLIT(gqa_rs_b64_d64, 64, 64)
INSTANTIATE_ROLESPLIT(gqa_rs_b128_d32, 128, 32)
INSTANTIATE_ROLESPLIT(gqa_rs_b64_d32, 64, 32)
INSTANTIATE_ROLESPLIT(gqa_rs_b128_d16, 128, 16)
INSTANTIATE_ROLESPLIT(gqa_rs_b64_d16, 64, 16)
INSTANTIATE_ROLESPLIT(gqa_rs_b128_d8, 128, 8)
INSTANTIATE_ROLESPLIT(gqa_rs_b64_d8, 64, 8)

/* Q-split (F16 hi/lo) study variant: the rolesplit body with a second
 * 16xD F16 tile holding half(q - half(q)), so the score simdgroups run
 * TWO QK products per K chunk and Q keeps ~22 mantissa bits instead of
 * F16's 11.  K stays value-exact (BF16 -> F16), so this restores the QK
 * products to scalar-kernel operand precision at the cost of one extra
 * staged tile (+8 KB threadgroup) and a doubled score-group MMA count. */
template <int BK, int DC>
static inline void gqa_mma_rolesplit_qsplit_body(
        constant proto_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        threadgroup uchar *tgm,
        uint2 group,
        ushort lane,
        ushort sg) {
    constexpr uint GQA2 = 12u;
    constexpr uint D2 = 256u;
    constexpr uint DCHUNKS2 = D2 / DC;
    constexpr uint TK2 = BK / 8u;
    constexpr uint OD2 = (D2 / 2u) / 8u;
    constexpr uint TPT2 = 192u;

    threadgroup half *qs = (threadgroup half *)tgm;            /* 16*D  */
    threadgroup half *qs_lo = qs + 16u * D2;                   /* 16*D  */
    threadgroup half *kvs = qs_lo + 16u * D2;                  /* BK*DC */
    threadgroup float *s_tile = (threadgroup float *)(kvs + BK * DC);
    threadgroup half *p_tile = (threadgroup half *)(s_tile + 16u * BK);
    threadgroup int *sel = (threadgroup int *)(p_tile + 16u * BK);
    threadgroup float *stats = (threadgroup float *)(sel + BK);
    threadgroup float *rfac = stats + 16u * 2u;
    threadgroup int *vote = (threadgroup int *)(rfac + 16u);
    threadgroup float *oblock = (threadgroup float *)(vote + 2u);

    const uint query = group.x;
    if (query >= args.queries) return;
    if (group.y >= args.kv_heads) return;
    const uint kv_head = group.y;
    const uint lane32 = (uint)lane;
    const uint tid = (uint)sg * 32u + lane32;
    const bool is_score = sg < 2u;
    const uint row_half = is_score ? (uint)sg : ((uint)sg - 2u) / 2u;
    const uint row_base = row_half * 8u;
    const uint dim_half = ((uint)sg - 2u) % 2u;

    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count =
        min(min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;

    if (selected == 0u) {
        for (uint i = tid; i < GQA2 * D2; i += TPT2)
            out[((ulong)query * args.query_heads + kv_head * GQA2 + i / D2) *
                args.head_dim + (i % D2)] = 0.0f;
        return;
    }

    for (uint i = tid; i < 16u * (D2 / 8u); i += TPT2) {
        const uint row = i / (D2 / 8u);
        const uint d8 = i % (D2 / 8u);
        float4 f0 = 0.0f;
        float4 f1 = 0.0f;
        if (row < GQA2) {
            const device const float *qrow = q +
                ((ulong)query * args.query_heads + kv_head * GQA2 + row) *
                    args.head_dim +
                d8 * 8u;
            f0 = *((device const float4 *)qrow);
            f1 = *((device const float4 *)(qrow + 4u));
        }
        qs[row * D2 + d8 * 8u + 0u] = half(f0.x);
        qs[row * D2 + d8 * 8u + 1u] = half(f0.y);
        qs[row * D2 + d8 * 8u + 2u] = half(f0.z);
        qs[row * D2 + d8 * 8u + 3u] = half(f0.w);
        qs[row * D2 + d8 * 8u + 4u] = half(f1.x);
        qs[row * D2 + d8 * 8u + 5u] = half(f1.y);
        qs[row * D2 + d8 * 8u + 6u] = half(f1.z);
        qs[row * D2 + d8 * 8u + 7u] = half(f1.w);
        qs_lo[row * D2 + d8 * 8u + 0u] = half(f0.x - (float)half(f0.x));
        qs_lo[row * D2 + d8 * 8u + 1u] = half(f0.y - (float)half(f0.y));
        qs_lo[row * D2 + d8 * 8u + 2u] = half(f0.z - (float)half(f0.z));
        qs_lo[row * D2 + d8 * 8u + 3u] = half(f0.w - (float)half(f0.w));
        qs_lo[row * D2 + d8 * 8u + 4u] = half(f1.x - (float)half(f1.x));
        qs_lo[row * D2 + d8 * 8u + 5u] = half(f1.y - (float)half(f1.y));
        qs_lo[row * D2 + d8 * 8u + 6u] = half(f1.z - (float)half(f1.z));
        qs_lo[row * D2 + d8 * 8u + 7u] = half(f1.w - (float)half(f1.w));
    }
    for (uint r = tid; r < 16u; r += TPT2) {
        stats[r * 2u] = -INFINITY;
        stats[r * 2u + 1u] = 0.0f;
    }

    simdgroup_float8x8 ofrag[OD2];
    if (!is_score)
        for (uint dd = 0u; dd < OD2; dd++)
            ofrag[dd] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const float scale = rsqrt((float)args.head_dim);
    const uint n_tiles = (selected + BK - 1u) / BK;
    const ulong kv_base = (ulong)kv_head * args.head_dim;
    const ulong kv_token_stride = (ulong)args.kv_heads * args.head_dim;

    for (uint ktile = 0u; ktile < n_tiles; ktile++) {
        for (uint k = tid; k < BK; k += TPT2) {
            const uint rank = ktile * BK + k;
            uint token = UINT_MAX;
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else if (rank < selected) {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            sel[k] = (token == UINT_MAX || token >= visible)
                ? -1 : (int)token;
        }
        if (args.debug != 0u && query == 0u && kv_head == 0u &&
            ktile == 0u && tid < 32u)
            out[(ulong)997u * args.query_heads * args.head_dim + tid] =
                (float)sel[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            simdgroup_float8x8 sfrag[TK2];
            if (is_score)
                for (uint ik = 0u; ik < TK2; ik++)
                    sfrag[ik] =
                        make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint chunk = 0u; chunk < DCHUNKS2; chunk++) {
                for (uint i = tid; i < BK * (DC / 8u); i += TPT2) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            key_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[(d8 * 8u + e) * BK + k] = half(as_type<float>(
                            (uint)elements[e] << 16));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint dd8 = 0u; dd8 < DC / 8u && is_score; dd8++) {
                    simdgroup_half8x8 qf;
                    simdgroup_load(qf, qs + row_base * D2 + chunk * DC +
                                       dd8 * 8u, D2, 0, false);
                    simdgroup_half8x8 qfl;
                    simdgroup_load(qfl, qs_lo + row_base * D2 + chunk * DC +
                                        dd8 * 8u, D2, 0, false);
                    for (uint ik = 0u; ik < TK2; ik++) {
                        simdgroup_half8x8 kt;
                        simdgroup_load(kt, kvs + (dd8 * 8u) * BK + ik * 8u,
                                       BK, 0, false);
                        simdgroup_multiply_accumulate(
                            sfrag[ik], qf, kt, sfrag[ik]);
                        simdgroup_multiply_accumulate(
                            sfrag[ik], qfl, kt, sfrag[ik]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (is_score)
                for (uint ik = 0u; ik < TK2; ik++)
                    simdgroup_store(sfrag[ik],
                                    s_tile + row_base * BK + ik * 8u,
                                    BK, 0, false);
            if (args.debug != 0u && query == 7u && kv_head == 0u &&
                sg == 0u) {
                simdgroup_store(sfrag[0u], oblock, 8u, 0, false);
                simdgroup_store(sfrag[1u], oblock + 64u, 8u, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (args.debug != 0u && query == 7u && kv_head == 0u &&
            tid < 128u) {
            out[(ulong)997u * args.query_heads * args.head_dim + tid] =
                oblock[tid];
        }
        if (args.debug != 0u && query == 7u && kv_head == 0u &&
            tid < 1024u) {
            out[(ulong)995u * args.query_heads * args.head_dim + tid] =
                s_tile[tid];
        }
        if (is_score) {
            const uint row = lane32 / 4u;
            const uint col0 = (lane32 % 4u) * (BK / 4u);
            float local_max = -INFINITY;
            for (uint c = 0u; c < BK / 4u; c++) {
                float s = -INFINITY;
                if (sel[col0 + c] >= 0)
                    s = s_tile[(row_base + row) * BK + col0 + c] * scale;
                s_tile[(row_base + row) * BK + col0 + c] = s;
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    row_base + row == 0u) {
                    out[(ulong)999u * args.query_heads * args.head_dim +
                        col0 + c] = s;
                }
                local_max = max(local_max, s);
            }
            float row_max = local_max;
            for (uint j = 1u; j < 4u; j++)
                row_max = max(row_max,
                    simd_shuffle(local_max, row * 4u + j));
            const float old_max = stats[(row_base + row) * 2u];
            const float old_sum = stats[(row_base + row) * 2u + 1u];
            const float new_max = max(old_max, row_max);
            const bool grows = isfinite(old_max) && new_max > old_max;
            float local_sum = 0.0f;
            for (uint c = 0u; c < BK / 4u; c++) {
                const float s = s_tile[(row_base + row) * BK + col0 + c];
                const float pp = isfinite(s) ? exp(s - new_max) : 0.0f;
                p_tile[(row_base + row) * BK + col0 + c] = half(pp);
                if (args.debug != 0u && query == 7u && kv_head == 0u &&
                    row_base + row == 0u) {
                    out[(ulong)999u * args.query_heads * args.head_dim +
                        256u + col0 + c] = pp;
                }
                local_sum += pp;
            }
            float row_sum = local_sum;
            for (uint j = 1u; j < 4u; j++)
                row_sum += simd_shuffle(local_sum, row * 4u + j);
            if (lane32 % 4u == 0u) {
                const float factor = isfinite(old_max)
                    ? exp(old_max - new_max) : 1.0f;
                rfac[row_base + row] = factor;
                stats[(row_base + row) * 2u] = new_max;
                stats[(row_base + row) * 2u + 1u] =
                    old_sum * factor + row_sum;
            }
            if (lane32 == 0u) vote[sg] = grows ? 1 : 0;
            if (args.debug != 0u && query == 7u && kv_head == 0u &&
                sg == 0u && lane32 % 4u == 0u) {
                out[(ulong)994u * args.query_heads * args.head_dim +
                    (row_base + row)] = row_max;
                out[(ulong)994u * args.query_heads * args.head_dim +
                    16u + (row_base + row)] =
                    stats[(row_base + row) * 2u + 1u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vote[0] != 0 || vote[1] != 0) {
            for (uint phase = 0u; phase < 4u; phase++) {
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD2; dd++)
                        simdgroup_store(ofrag[dd], oblock + dd * 8u,
                                        128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    const uint r = lane32 / 4u;
                    const float factor = rfac[row_base + r];
                    const uint col = (lane32 % 4u) * 32u;
                    for (uint c = 0u; c < 32u; c++)
                        oblock[r * 128u + col + c] *= factor;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD2; dd++)
                        simdgroup_load(ofrag[dd], oblock + dd * 8u,
                                       128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            for (uint chunk = 0u; chunk < DCHUNKS2; chunk++) {
                const bool pv_chunk =
                    !is_score &&
                    chunk >= dim_half * (DCHUNKS2 / 2u) &&
                    chunk < (dim_half + 1u) * (DCHUNKS2 / 2u);
                for (uint i = tid; i < BK * (DC / 8u); i += TPT2) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            value_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[k * DC + d8 * 8u + e] = half(as_type<float>(
                            (uint)elements[e] << 16));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (pv_chunk) {
                    for (uint ik = 0u; ik < TK2; ik++) {
                        simdgroup_half8x8 pf;
                        simdgroup_load(pf,
                                       p_tile + row_base * BK + ik * 8u,
                                       BK, 0, false);
                        for (uint dd = 0u; dd < DC / 8u; dd++) {
                            simdgroup_half8x8 vf;
                            simdgroup_load(vf, kvs + ik * 8u * DC + dd * 8u,
                                           DC, 0, false);
                            simdgroup_multiply_accumulate(
                                ofrag[(chunk % (DCHUNKS2 / 2u)) *
                                          (DC / 8u) +
                                      dd],
                                pf, vf,
                                ofrag[(chunk % (DCHUNKS2 / 2u)) *
                                          (DC / 8u) +
                                      dd]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    for (uint phase = 0u; phase < 4u; phase++) {
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA2 - 8u;
            for (uint dd = 0u; dd < OD2; dd++)
                simdgroup_store(ofrag[dd], oblock + dd * 8u, 128u, 0,
                                false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA2 - 8u;
            for (uint i = lane32; i < emit_rows * (128u / 8u); i += 32u) {
                const uint r = i / (128u / 8u);
                const uint d8 = i % (128u / 8u);
                const uint head = kv_head * GQA2 + row_base + r;
                const ulong base =
                    ((ulong)query * args.query_heads + head) *
                    args.head_dim;
                const float m = stats[(row_base + r) * 2u];
                const float s = stats[(row_base + r) * 2u + 1u];
                const float inv =
                    isfinite(m) && s > 0.0f ? 1.0f / s : 0.0f;
                for (uint e = 0u; e < 8u; e++) {
                    const uint d = dim_half * 128u + d8 * 8u + e;
                    const float g =
                        1.0f / (1.0f + exp(-raw_gate[base + d]));
                    out[base + d] =
                        oblock[r * 128u + d8 * 8u + e] * inv * g;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define INSTANTIATE_ROLESPLIT_QSPLIT(name, bk, dc) \
    kernel void name( \
            constant proto_args &args [[buffer(0)]], \
            device const float *q [[buffer(1)]], \
            device const float *raw_gate [[buffer(2)]], \
            device const ushort *key_cache [[buffer(3)]], \
            device const ushort *value_cache [[buffer(4)]], \
            device const uint *selected_blocks [[buffer(5)]], \
            device const uint *selected_counts [[buffer(6)]], \
            device const uint *visible_tokens [[buffer(7)]], \
            device float *out [[buffer(8)]], \
            threadgroup uchar *tgm [[threadgroup(0)]], \
            uint2 group [[threadgroup_position_in_grid]], \
            ushort lane [[thread_index_in_simdgroup]], \
            ushort sg [[simdgroup_index_in_threadgroup]]) { \
        gqa_mma_rolesplit_qsplit_body<bk, dc>( \
            args, q, raw_gate, key_cache, value_cache, selected_blocks, \
            selected_counts, visible_tokens, out, tgm, group, lane, sg); \
    }

INSTANTIATE_ROLESPLIT_QSPLIT(gqa_rs_b64_d8_qsplit, 64, 8)
INSTANTIATE_ROLESPLIT_QSPLIT(gqa_rs_b128_d8_qsplit, 128, 8)
