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
using namespace metal;

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
