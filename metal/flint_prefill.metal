// Recursive Flint prefill kernels (fork-owned; docs/RECURSIVE_FLINT.md step 2).
//
// rec prefills with MLX's native graph (model.py) in chunks of 2048 tokens:
// quantized_matmul for every projection (weights dequantized to bf16 with one
// rounding, bf16 MMA with f32 accumulation, bf16 output), gather_qmm for the
// routed experts over tokens sorted by expert, mx.fast.scaled_dot_product_attention
// with sinks (f32 online softmax, probabilities in bf16 for the PV product),
// rms_norm / rope as in the decode kernel.  These kernels follow that
// arithmetic operation by operation; what cannot match bit for bit is the
// summation order inside the MMA tiles, so prefill is held to rec's own
// spread rather than exactness (the decode kernel in flint.metal is exact).
//
// Every activation is bf16 in device memory; every intermediate that MLX
// rounds to bf16 is rounded here at the same point.  Compiled by
// ds4_metal_flint.m with strict math, like flint.metal.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

typedef bfloat bfloat16_t;

constant constexpr uint FP_BM = 64;
constant constexpr uint FP_BN = 64;
constant constexpr uint FP_BK = 32;
constant constexpr uint FP_LDS = 40;        // tile row stride (elements), 16-byte padded
constant constexpr uint FP_THREADS = 128;   // 4 simdgroups: 2 x 2 of 32 x 32
constant constexpr float FP_LOG2E = 1.4426950408889634f;
constant constexpr float FP_NEG_INF = -3.402823466e+38f;

// Expert pack layout (gguf-tools/flint_int4_to_gguf.py): byte offsets inside one pack.
constant constexpr uint FP_PACK_GATE_W = 0u;
constant constexpr uint FP_PACK_GATE_S = 524288u;
constant constexpr uint FP_PACK_GATE_B = 557056u;
constant constexpr uint FP_PACK_UP_W = 589824u;
constant constexpr uint FP_PACK_UP_S = 1114112u;
constant constexpr uint FP_PACK_UP_B = 1146880u;
constant constexpr uint FP_PACK_DOWN_W = 1179648u;
constant constexpr uint FP_PACK_DOWN_S = 1703936u;
constant constexpr uint FP_PACK_DOWN_B = 1736704u;

// Element coordinates (col, row) this lane holds in an 8 x 8 simdgroup matrix,
// the mapping MLX's steel kernels rely on (mma.h, BaseMMAFrag::get_coord).
// Each lane holds (row, col) and (row, col + 1); the four lanes of one row
// differ in lane bits 0 and 3.  tests/flint/test_flint_kernels.m checks it.
inline short2 fp_frag_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2(fn, fm);
}

inline float fp_row_max(float v) {
    v = max(v, simd_shuffle_xor(v, 1));
    v = max(v, simd_shuffle_xor(v, 8));
    return v;
}
inline float fp_row_sum(float v) {
    v += simd_shuffle_xor(v, 1);
    v += simd_shuffle_xor(v, 8);
    return v;
}

inline float fp_bf16_round(float x) { return float(bfloat16_t(x)); }

// The two elements of an 8 x 8 simdgroup matrix this lane holds (MLX's mma.h
// reinterprets thread_elements() the same way).
template <typename T>
inline thread vec<T, 2>& fp_elems(thread simdgroup_matrix<T, 8, 8>& m) {
    return reinterpret_cast<thread vec<T, 2>&>(m.thread_elements());
}

// ---------------------------------------------------------------------------
// GEMM tiles: C[M][N] = A[M][K] . W[N][K]^T, 64 x 64 output tile, k steps of 32.
// ---------------------------------------------------------------------------

// A: 64 rows x 32 k from device rows (null row pointer -> zeros).
struct FpRowsPlain {
    const device bfloat16_t* a;
    uint lda, M;
    const device bfloat16_t* row(uint r) const { return r < M ? a + (size_t)r * lda : nullptr; }
};
struct FpRowsGather {
    const device bfloat16_t* a;        // token rows, lda elements each
    const device uint* rows;           // sorted entries: token * 8 + slot
    uint lda, begin, count;
    const device bfloat16_t* row(uint r) const {
        return r < count ? a + (size_t)(rows[begin + r] >> 3) * lda : nullptr;
    }
};
struct FpRowsSorted {
    const device bfloat16_t* a;        // rows in sorted order, lda elements each
    uint lda, begin, count;
    const device bfloat16_t* row(uint r) const { return r < count ? a + (size_t)(begin + r) * lda : nullptr; }
};

template <typename Src>
inline void fp_load_a(threadgroup bfloat16_t* As, Src src, uint k0, ushort tid) {
    const ushort r = tid >> 1, c = (tid & 1) * 16;
    const device bfloat16_t* p = src.row(r);
    threadgroup vec<bfloat16_t, 8>* dst = (threadgroup vec<bfloat16_t, 8>*)(As + r * FP_LDS + c);
    if (p) {
        const device vec<bfloat16_t, 8>* v = (const device vec<bfloat16_t, 8>*)(p + k0 + c);
        dst[0] = v[0];
        dst[1] = v[1];
    } else {
        dst[0] = vec<bfloat16_t, 8>(bfloat16_t(0.0f));
        dst[1] = vec<bfloat16_t, 8>(bfloat16_t(0.0f));
    }
}

// B: 64 n rows x 32 k of a 4-bit affine matrix [N][K/2] nibbles (low first),
// scales and biases bf16 [N][K/64]; w = bf16(scale * q + bias), one rounding
// as MLX's QuantizedBlockLoader.
inline void fp_load_b_q4(threadgroup bfloat16_t* Bs, const device uchar* W, const device bfloat16_t* S,
                         const device bfloat16_t* Bz, uint n0, uint k0, uint K, ushort tid) {
    const ushort r = tid >> 1, hf = tid & 1;
    const uint n = n0 + r;
    const uint gi = n * (K >> 6) + (k0 >> 6);
    const float s = float(S[gi]), b = float(Bz[gi]);
    const ulong bits = *(const device ulong*)(W + (size_t)n * (K >> 1) + (k0 >> 1) + hf * 8u);
    threadgroup bfloat16_t* dst = Bs + r * FP_LDS + hf * 16;
    for (uint i = 0; i < 16; i++) dst[i] = bfloat16_t(s * float((bits >> (4 * i)) & 15ul) + b);
}

// B: 64 n rows x 32 k of a bf16 matrix [N][K].
inline void fp_load_b_bf16(threadgroup bfloat16_t* Bs, const device bfloat16_t* W, uint n0, uint k0, uint K,
                           ushort tid) {
    const ushort r = tid >> 1, c = (tid & 1) * 16;
    const device vec<bfloat16_t, 8>* v = (const device vec<bfloat16_t, 8>*)(W + (size_t)(n0 + r) * K + k0 + c);
    threadgroup vec<bfloat16_t, 8>* dst = (threadgroup vec<bfloat16_t, 8>*)(Bs + r * FP_LDS + c);
    dst[0] = v[0];
    dst[1] = v[1];
}

inline void fp_mma(thread simdgroup_matrix<float, 8, 8> (&acc)[4][4], const threadgroup bfloat16_t* As,
                   const threadgroup bfloat16_t* Bs, ushort sm, ushort sn) {
    for (ushort kk = 0; kk < FP_BK; kk += 8) {
        simdgroup_matrix<bfloat16_t, 8, 8> a[4], b[4];
        for (ushort i = 0; i < 4; i++) simdgroup_load(a[i], As + (sm + 8 * i) * FP_LDS + kk, FP_LDS);
        for (ushort j = 0; j < 4; j++) simdgroup_load(b[j], Bs + (sn + 8 * j) * FP_LDS + kk, FP_LDS, ulong2(0, 0), true);
        for (ushort i = 0; i < 4; i++)
            for (ushort j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

inline void fp_zero(thread simdgroup_matrix<float, 8, 8> (&acc)[4][4]) {
    for (ushort i = 0; i < 4; i++)
        for (ushort j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

struct FpGemmArgs {
    uint M, N, K;
    uint lda, ldc, ldr;
};

// MODE 0: C = bf16(acc)
// MODE 1: C = bf16(R + bf16(acc))                           (residual add)
// MODE 2: C = bf16(silu(bf16(acc_gate)) * bf16(acc_up))     (dense / shared GLU; silu by table)
// MODE 3: C = bf16(R + bf16(bf16(sum_6 Y6) + bf16(acc)))    (shared-expert down + routed combine)
// BKIND 0: 4-bit affine weights; 1: bf16 weights.
template <int MODE, int BKIND>
kernel void flint_gemm(const device bfloat16_t* A [[buffer(0)]],
                       const device uchar* W [[buffer(1)]],
                       const device bfloat16_t* S [[buffer(2)]],
                       const device bfloat16_t* Bz [[buffer(3)]],
                       device bfloat16_t* C [[buffer(4)]],
                       constant FpGemmArgs& args [[buffer(5)]],
                       const device bfloat16_t* R [[buffer(6)]],
                       const device uchar* W2 [[buffer(7)]],
                       const device bfloat16_t* S2 [[buffer(8)]],
                       const device bfloat16_t* B2 [[buffer(9)]],
                       const device bfloat16_t* silu [[buffer(10)]],
                       const device bfloat16_t* Y6 [[buffer(11)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat16_t As[FP_BM * FP_LDS];
    threadgroup bfloat16_t Bs[FP_BN * FP_LDS];
    threadgroup bfloat16_t Bs2[FP_BN * FP_LDS];   // GLU up operand (unused tiles are dead code otherwise)
    const uint m0 = tg.y * FP_BM, n0 = tg.x * FP_BN;
    const ushort sm = (sg >> 1) * 32, sn = (sg & 1) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    simdgroup_matrix<float, 8, 8> acc2[4][4];
    fp_zero(acc);
    if (MODE == 2) fp_zero(acc2);
    FpRowsPlain src = {A + (size_t)m0 * args.lda, args.lda, args.M - m0};   // rows of this tile
    for (uint k0 = 0; k0 < args.K; k0 += FP_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_load_a(As, src, k0, tid);
        if (BKIND == 0) fp_load_b_q4(Bs, W, S, Bz, n0, k0, args.K, tid);
        else fp_load_b_bf16(Bs, (const device bfloat16_t*)W, n0, k0, args.K, tid);
        if (MODE == 2) fp_load_b_q4(Bs2, W2, S2, B2, n0, k0, args.K, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_mma(acc, As, Bs, sm, sn);
        if (MODE == 2) fp_mma(acc2, As, Bs2, sm, sn);
    }
    const short2 fc = fp_frag_coord(lane);
    for (ushort i = 0; i < 4; i++) {
        const uint row = m0 + sm + 8 * i + fc.y;
        if (row >= args.M) continue;
        for (ushort j = 0; j < 4; j++) {
            const uint col = n0 + sn + 8 * j + fc.x;
            const thread vec<float, 2>& e = fp_elems(acc[i][j]);
            vec<bfloat16_t, 2> out;
            for (ushort t = 0; t < 2; t++) {
                const float v = fp_bf16_round(e[t]);
                if (MODE == 0) {
                    out[t] = bfloat16_t(v);
                } else if (MODE == 1) {
                    out[t] = bfloat16_t(float(R[(size_t)row * args.ldr + col + t]) + v);
                } else if (MODE == 2) {
                    const thread vec<float, 2>& e2 = fp_elems(acc2[i][j]);
                    const bfloat16_t g = bfloat16_t(v);
                    out[t] = bfloat16_t(float(silu[as_type<uint>(float(g)) >> 16]) * fp_bf16_round(e2[t]));
                } else {
                    const device bfloat16_t* y = Y6 + ((size_t)row * 6) * args.N + col + t;
                    float sum = float(y[0]) + 0.0f;
                    for (uint r = 1; r < 6; r++) sum = fp_bf16_round((float(y[r * args.N]) + 0.0f) + sum);
                    const float mlp = fp_bf16_round(fp_bf16_round(sum) + v);
                    out[t] = bfloat16_t(float(R[(size_t)row * args.ldr + col + t]) + mlp);
                }
            }
            *(device vec<bfloat16_t, 2>*)(C + (size_t)row * args.ldc + col) = out;
        }
    }
}

template [[host_name("flint_gemm_q4_store")]] [[kernel]] decltype(flint_gemm<0, 0>) flint_gemm<0, 0>;
template [[host_name("flint_gemm_q4_residual")]] [[kernel]] decltype(flint_gemm<1, 0>) flint_gemm<1, 0>;
template [[host_name("flint_gemm_q4_glu")]] [[kernel]] decltype(flint_gemm<2, 0>) flint_gemm<2, 0>;
template [[host_name("flint_gemm_q4_combine")]] [[kernel]] decltype(flint_gemm<3, 0>) flint_gemm<3, 0>;
template [[host_name("flint_gemm_bf16_store")]] [[kernel]] decltype(flint_gemm<0, 1>) flint_gemm<0, 1>;

// ---------------------------------------------------------------------------
// Routed experts.  Routing writes ids/weights per token and a count per
// expert; offsets turns counts into sorted-row starts and a tile table (64
// rows per tile, one expert per tile) plus indirect dispatch arguments;
// scatter fills the sorted row list; then two grouped GEMMs.
// ---------------------------------------------------------------------------

struct FpTile { uint expert, begin, count, pad; };

// Router logits [M][256] bf16 (from flint_gemm_bf16_store) -> top-6 by
// sigmoid probability (own-rank, ties to the lower index, as the decode
// kernel), normalized weights bf16(p / (sum + 1e-20)), expert counts.
// One simdgroup per token; lane handles experts lane + 32 j.
kernel void flint_route_topk(const device bfloat16_t* logits [[buffer(0)]],
                             device uint* ids [[buffer(1)]],
                             device bfloat16_t* weights [[buffer(2)]],
                             device atomic_uint* counts [[buffer(3)]],
                             constant uint& M [[buffer(4)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint token = tgid * 4 + sg;
    if (token >= M) return;
    float p[8], b[8];
    for (uint j = 0; j < 8; j++) {
        const float a = float(logits[(size_t)token * 256 + lane + 32 * j]);
        const float tail = 1.0f / (1.0f + metal::precise::exp(metal::abs(a)));
        p[j] = a < 0.0f ? tail : 1.0f - tail;
        b[j] = p[j];
    }
    uint chosen[6];
    float prob[6];
    float sum = 0.0f;
    for (uint rank = 0; rank < 6; rank++) {
        float best = FP_NEG_INF;
        uint idx = 0;
        for (uint j = 0; j < 8; j++) {
            const uint i = lane + 32 * j;
            if (b[j] > best || (b[j] == best && i < idx)) { best = b[j]; idx = i; }
        }
        const float peak = simd_max(best);
        const uint selected = simd_min(best == peak ? idx : 0xffffffffu);
        const float value = simd_shuffle(p[selected / 32], selected % 32);
        chosen[rank] = selected;
        prob[rank] = value;
        sum += value;
        for (uint j = 0; j < 8; j++) if (lane + 32 * j == selected) b[j] = FP_NEG_INF;
    }
    sum += 1.0e-20f;
    if (lane < 6) {
        ids[token * 6 + lane] = chosen[lane];
        weights[token * 6 + lane] = bfloat16_t(prob[lane] / sum);
        atomic_fetch_add_explicit(counts + chosen[lane], 1u, memory_order_relaxed);
    }
}

// One threadgroup of 256 threads: exclusive prefix of the 256 counts ->
// starts[257]; tiles per expert -> tile table; indirect dispatch arguments
// {tiles, 8, 1} for gate/up and {tiles, 32, 1} for down; cursors zeroed.
kernel void flint_route_offsets(device uint* counts [[buffer(0)]],             // zeroed again for the next layer
                                device uint* starts [[buffer(1)]],
                                device FpTile* tiles [[buffer(2)]],
                                device uint* dispatch [[buffer(3)]],    // 2 x 3 words
                                device uint* cursors [[buffer(4)]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[8], tpart[8];
    const uint c = counts[tid];
    const uint nt = (c + 63u) >> 6;
    const uint ex = simd_prefix_exclusive_sum(c);
    const uint tex = simd_prefix_exclusive_sum(nt);
    if (lane == 31) { part[sg] = ex + c; tpart[sg] = tex + nt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0, tbase = 0;
    for (ushort s = 0; s < sg; s++) { base += part[s]; tbase += tpart[s]; }
    const uint start = base + ex, tstart = tbase + tex;
    starts[tid] = start;
    cursors[tid] = 0;
    counts[tid] = 0;
    for (uint t = 0; t < nt; t++) {
        FpTile tile = {tid, start + t * 64u, min(64u, c - t * 64u), 0u};
        tiles[tstart + t] = tile;
    }
    if (tid == 255) {
        starts[256] = start + c;
        const uint total = tstart + nt;
        dispatch[0] = total; dispatch[1] = 8;  dispatch[2] = 1;
        dispatch[3] = total; dispatch[4] = 32; dispatch[5] = 1;
    }
}

kernel void flint_route_scatter(const device uint* ids [[buffer(0)]],
                                const device uint* starts [[buffer(1)]],
                                device atomic_uint* cursors [[buffer(2)]],
                                device uint* rows [[buffer(3)]],
                                constant uint& M [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= M * 6) return;
    const uint e = ids[gid];
    const uint pos = starts[e] + atomic_fetch_add_explicit(cursors + e, 1u, memory_order_relaxed);
    rows[pos] = (gid / 6) * 8 + (gid % 6);
}

// Gate/up GLU over one 64-row tile of one expert: act_sorted[pos][512].
kernel void flint_moe_gateup(const device bfloat16_t* X [[buffer(0)]],        // normed tokens [M][2048]
                             const device uint* rows [[buffer(1)]],
                             const device FpTile* tiles [[buffer(2)]],
                             const device ulong* experts [[buffer(3)]],        // pack addresses of this layer
                             const device bfloat16_t* silu [[buffer(4)]],
                             device bfloat16_t* act [[buffer(5)]],             // [M*6][512] sorted order
                             uint2 tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat16_t As[FP_BM * FP_LDS];
    threadgroup bfloat16_t Bs[FP_BN * FP_LDS];
    threadgroup bfloat16_t Bs2[FP_BN * FP_LDS];
    const FpTile tile = tiles[tg.x];
    const device uchar* pack = (const device uchar*)experts[tile.expert];
    const uint n0 = tg.y * FP_BN;
    const ushort sm = (sg >> 1) * 32, sn = (sg & 1) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4], acc2[4][4];
    fp_zero(acc);
    fp_zero(acc2);
    FpRowsGather src = {X, rows, 2048u, tile.begin, tile.count};
    for (uint k0 = 0; k0 < 2048u; k0 += FP_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_load_a(As, src, k0, tid);
        fp_load_b_q4(Bs, pack + FP_PACK_GATE_W, (const device bfloat16_t*)(pack + FP_PACK_GATE_S),
                     (const device bfloat16_t*)(pack + FP_PACK_GATE_B), n0, k0, 2048u, tid);
        fp_load_b_q4(Bs2, pack + FP_PACK_UP_W, (const device bfloat16_t*)(pack + FP_PACK_UP_S),
                     (const device bfloat16_t*)(pack + FP_PACK_UP_B), n0, k0, 2048u, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_mma(acc, As, Bs, sm, sn);
        fp_mma(acc2, As, Bs2, sm, sn);
    }
    const short2 fc = fp_frag_coord(lane);
    for (ushort i = 0; i < 4; i++) {
        const uint r = sm + 8 * i + fc.y;
        if (r >= tile.count) continue;
        for (ushort j = 0; j < 4; j++) {
            const uint col = n0 + sn + 8 * j + fc.x;
            const thread vec<float, 2>& g = fp_elems(acc[i][j]);
            const thread vec<float, 2>& u = fp_elems(acc2[i][j]);
            vec<bfloat16_t, 2> out;
            for (ushort t = 0; t < 2; t++) {
                const bfloat16_t gb = bfloat16_t(g[t]);
                out[t] = bfloat16_t(float(silu[as_type<uint>(float(gb)) >> 16]) * fp_bf16_round(u[t]));
            }
            *(device vec<bfloat16_t, 2>*)(act + (size_t)(tile.begin + r) * 512u + col) = out;
        }
    }
}

// Down projection of one tile: y6[token][slot][2048] = bf16(bf16(acc) * weight).
kernel void flint_moe_down(const device bfloat16_t* act [[buffer(0)]],
                           const device uint* rows [[buffer(1)]],
                           const device FpTile* tiles [[buffer(2)]],
                           const device ulong* experts [[buffer(3)]],
                           const device bfloat16_t* weights [[buffer(4)]],     // [M][6]
                           device bfloat16_t* y6 [[buffer(5)]],                // [M][6][2048]
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat16_t As[FP_BM * FP_LDS];
    threadgroup bfloat16_t Bs[FP_BN * FP_LDS];
    const FpTile tile = tiles[tg.x];
    const device uchar* pack = (const device uchar*)experts[tile.expert];
    const uint n0 = tg.y * FP_BN;
    const ushort sm = (sg >> 1) * 32, sn = (sg & 1) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    fp_zero(acc);
    FpRowsSorted src = {act, 512u, tile.begin, tile.count};
    for (uint k0 = 0; k0 < 512u; k0 += FP_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_load_a(As, src, k0, tid);
        fp_load_b_q4(Bs, pack + FP_PACK_DOWN_W, (const device bfloat16_t*)(pack + FP_PACK_DOWN_S),
                     (const device bfloat16_t*)(pack + FP_PACK_DOWN_B), n0, k0, 512u, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fp_mma(acc, As, Bs, sm, sn);
    }
    const short2 fc = fp_frag_coord(lane);
    for (ushort i = 0; i < 4; i++) {
        const uint r = sm + 8 * i + fc.y;
        if (r >= tile.count) continue;
        const uint entry = rows[tile.begin + r];
        const uint token = entry >> 3, slot = entry & 7u;
        const float w = float(weights[token * 6 + slot]);
        device bfloat16_t* dst = y6 + ((size_t)token * 6 + slot) * 2048u;
        for (ushort j = 0; j < 4; j++) {
            const uint col = n0 + sn + 8 * j + fc.x;
            const thread vec<float, 2>& e = fp_elems(acc[i][j]);
            vec<bfloat16_t, 2> out;
            for (ushort t = 0; t < 2; t++) out[t] = bfloat16_t(fp_bf16_round(e[t]) * w);
            *(device vec<bfloat16_t, 2>*)(dst + col) = out;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention: 64 queries x 32 heads per threadgroup grid cell, keys in tiles
// of 32, f32 online softmax in base 2 with the scale folded in, sinks as the
// initial running maximum (sum 1), probabilities rounded to bf16 before the
// PV product, as mx.fast.scaled_dot_product_attention's steel kernel.
// Keys/values: [4 kv heads][n_keys][128] rows (stride kv_head_stride between
// heads), key j at absolute position key_pos0 + j.  Query i is at position
// q_pos0 + i.  window: 0 = causal, else keys with q - k >= window are masked.
// ---------------------------------------------------------------------------

struct FpAttnArgs {
    uint M;             // queries
    uint q_pos0;
    uint key_pos0;
    uint n_keys;
    uint kv_head_stride;   // elements between kv heads
    uint window;           // 0 or 513
    uint ldq, ldo;         // 4096
};

constant constexpr uint FP_ATT_BQ = 64;
constant constexpr uint FP_ATT_BK = 32;
constant constexpr uint FP_ATT_LDK = 136;   // 128 + 8 pad

kernel void flint_attention(const device bfloat16_t* Q [[buffer(0)]],        // [M][4096]
                            const device bfloat16_t* K [[buffer(1)]],
                            const device bfloat16_t* V [[buffer(2)]],
                            const device bfloat16_t* sinks [[buffer(3)]],    // [32] bf16
                            device bfloat16_t* O [[buffer(4)]],              // [M][4096]
                            constant FpAttnArgs& args [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat16_t Ks[FP_ATT_BK * FP_ATT_LDK];
    threadgroup bfloat16_t Vs[FP_ATT_BK * FP_ATT_LDK];
    const uint head = tg.y, kv_head = head >> 3;
    const uint qt0 = tg.x * FP_ATT_BQ;
    if (qt0 >= args.M) return;
    const uint qrow0 = qt0 + sg * 16;             // this simdgroup's 16 query rows
    const device bfloat16_t* Kh = K + (size_t)kv_head * args.kv_head_stride;
    const device bfloat16_t* Vh = V + (size_t)kv_head * args.kv_head_stride;
    const float scale = (1.0f / 11.313708498984761f) * FP_LOG2E;   // 1/sqrt(128) * log2(e)

    // Q fragments: 2 row frags x 16 d frags.  Rows past M are read anyway (the
    // host pads the Q buffer to a multiple of 64 rows); their outputs are dropped
    // and rows do not mix inside the products.
    simdgroup_matrix<bfloat16_t, 8, 8> qf[2][16];
    for (ushort i = 0; i < 2; i++) {
        const device bfloat16_t* qp = Q + (size_t)(qrow0 + 8 * i) * args.ldq + head * 128;
        for (ushort d = 0; d < 16; d++) simdgroup_load(qf[i][d], qp + d * 8, args.ldq);
    }
    simdgroup_matrix<float, 8, 8> of[2][16];
    for (ushort i = 0; i < 2; i++)
        for (ushort d = 0; d < 16; d++) of[i][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const short2 fc = fp_frag_coord(lane);
    float m_run[2], l_run[2];
    const float sink = float(sinks[head]) * FP_LOG2E;
    for (ushort i = 0; i < 2; i++) { m_run[i] = sink; l_run[i] = 1.0f; }

    // Key range for this query tile.
    const uint q_last = min(qt0 + FP_ATT_BQ, args.M) - 1 + args.q_pos0;
    uint k_lo = 0;
    if (args.window) {
        const uint q_first = qt0 + args.q_pos0;
        const uint lowest = q_first + 1 >= args.window ? q_first + 1 - args.window : 0;
        k_lo = lowest > args.key_pos0 ? lowest - args.key_pos0 : 0;
    }
    k_lo &= ~(FP_ATT_BK - 1);
    const uint k_hi = min(args.n_keys, q_last + 1 - args.key_pos0);

    for (uint kt0 = k_lo; kt0 < k_hi; kt0 += FP_ATT_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 32 keys x 128 dims = 4096 elements per tile: each thread copies 4 x 8 of K and of V.
        for (ushort c = 0; c < 4; c++) {
            const ushort idx = tid + c * FP_THREADS;      // 0..511 vectors of 8
            const ushort r = idx >> 4, col = (idx & 15) * 8;
            const uint j = kt0 + r;
            vec<bfloat16_t, 8> kv8 = vec<bfloat16_t, 8>(bfloat16_t(0.0f)), vv8 = kv8;
            if (j < k_hi) {
                kv8 = *(const device vec<bfloat16_t, 8>*)(Kh + (size_t)j * 128 + col);
                vv8 = *(const device vec<bfloat16_t, 8>*)(Vh + (size_t)j * 128 + col);
            }
            *(threadgroup vec<bfloat16_t, 8>*)(Ks + r * FP_ATT_LDK + col) = kv8;
            *(threadgroup vec<bfloat16_t, 8>*)(Vs + r * FP_ATT_LDK + col) = vv8;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for 16 x 32.
        simdgroup_matrix<float, 8, 8> sf[2][4];
        for (ushort i = 0; i < 2; i++)
            for (ushort kj = 0; kj < 4; kj++) sf[i][kj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (ushort d = 0; d < 16; d++) {
            simdgroup_matrix<bfloat16_t, 8, 8> kf[4];
            for (ushort kj = 0; kj < 4; kj++)
                simdgroup_load(kf[kj], Ks + kj * 8 * FP_ATT_LDK + d * 8, FP_ATT_LDK, ulong2(0, 0), true);
            for (ushort i = 0; i < 2; i++)
                for (ushort kj = 0; kj < 4; kj++) simdgroup_multiply_accumulate(sf[i][kj], qf[i][d], kf[kj], sf[i][kj]);
        }
        // Scale, mask, online softmax per row.
        for (ushort i = 0; i < 2; i++) {
            const uint qpos = qrow0 + 8 * i + fc.y + args.q_pos0;
            float rmax = FP_NEG_INF;
            for (ushort kj = 0; kj < 4; kj++) {
                thread vec<float, 2>& e = fp_elems(sf[i][kj]);
                for (ushort t = 0; t < 2; t++) {
                    const uint j = kt0 + kj * 8 + fc.x + t;
                    const uint kpos = args.key_pos0 + j;
                    bool ok = j < k_hi && kpos <= qpos;
                    if (args.window && ok) ok = qpos - kpos < args.window;
                    e[t] = ok ? e[t] * scale : FP_NEG_INF;
                    rmax = max(rmax, e[t]);
                }
            }
            rmax = fp_row_max(rmax);
            const float m_new = max(m_run[i], rmax);
            const float alpha = metal::fast::exp2(m_run[i] - m_new);
            float rsum = 0.0f;
            for (ushort kj = 0; kj < 4; kj++) {
                thread vec<float, 2>& e = fp_elems(sf[i][kj]);
                for (ushort t = 0; t < 2; t++) {
                    const float p = e[t] == FP_NEG_INF ? 0.0f : metal::fast::exp2(e[t] - m_new);
                    e[t] = p;
                    rsum += p;
                }
            }
            rsum = fp_row_sum(rsum);
            l_run[i] = l_run[i] * alpha + rsum;
            m_run[i] = m_new;
            for (ushort d = 0; d < 16; d++) {
                thread vec<float, 2>& o = fp_elems(of[i][d]);
                o[0] *= alpha;
                o[1] *= alpha;
            }
        }
        // O += P V with P in bf16.
        simdgroup_matrix<bfloat16_t, 8, 8> pf[2][4];
        for (ushort i = 0; i < 2; i++)
            for (ushort kj = 0; kj < 4; kj++) {
                const thread vec<float, 2>& e = fp_elems(sf[i][kj]);
                thread vec<bfloat16_t, 2>& p = fp_elems(pf[i][kj]);
                p[0] = bfloat16_t(e[0]);
                p[1] = bfloat16_t(e[1]);
            }
        for (ushort kj = 0; kj < 4; kj++) {
            simdgroup_matrix<bfloat16_t, 8, 8> vf[16];
            for (ushort d = 0; d < 16; d++) simdgroup_load(vf[d], Vs + kj * 8 * FP_ATT_LDK + d * 8, FP_ATT_LDK);
            for (ushort i = 0; i < 2; i++)
                for (ushort d = 0; d < 16; d++) simdgroup_multiply_accumulate(of[i][d], pf[i][kj], vf[d], of[i][d]);
        }
    }
    for (ushort i = 0; i < 2; i++) {
        const uint row = qrow0 + 8 * i + fc.y;
        if (row >= args.M) continue;
        const float inv = 1.0f / l_run[i];
        device bfloat16_t* dst = O + (size_t)row * args.ldo + head * 128;
        for (ushort d = 0; d < 16; d++) {
            const thread vec<float, 2>& o = fp_elems(of[i][d]);
            vec<bfloat16_t, 2> out;
            out[0] = bfloat16_t(o[0] * inv);
            out[1] = bfloat16_t(o[1] * inv);
            *(device vec<bfloat16_t, 2>*)(dst + d * 8 + fc.x) = out;
        }
    }
}

// ---------------------------------------------------------------------------
// Small kernels: embedding, RMS norm, head norm + RoPE + KV placement, KV
// copies, the last-row head.
// ---------------------------------------------------------------------------

// out[t][i] = bf16(s * bf16(code) + b) in bf16 arithmetic (mx.dequantize).
kernel void flint_embed(const device int* tokens [[buffer(0)]],
                        const device uchar* W [[buffer(1)]],
                        const device bfloat16_t* S [[buffer(2)]],
                        const device bfloat16_t* Bz [[buffer(3)]],
                        device bfloat16_t* out [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, t = gid.y;
    const size_t row = (size_t)uint(tokens[t]);
    const uint gi = uint(row) * 32u + (i >> 6);
    const bfloat16_t code = bfloat16_t(float(W[row * 2048u + i]));
    out[(size_t)t * 2048u + i] = S[gi] * code + Bz[gi];
}

// RMS norm of 2048-wide rows: out = w * bf16(x * rsqrt(mean(x^2) + eps)), one
// threadgroup of 256 threads per row, 8 elements per thread (mx.fast.rms_norm).
kernel void flint_rmsnorm(const device bfloat16_t* x [[buffer(0)]],
                          const device bfloat16_t* w [[buffer(1)]],
                          device bfloat16_t* out [[buffer(2)]],
                          constant uint& row_offset [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float temp[8];
    const size_t row = (size_t)(row_offset + tg);
    const device bfloat16_t* p = x + row * 2048u;
    float v[8];
    float squared = 0.0f;
    for (uint j = 0; j < 8; j++) {
        v[j] = float(p[tid * 8 + j]);
        squared += v[j] * v[j];
    }
    squared = simd_sum(squared);
    if (lane == 0) temp[sg] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    const float inverse = metal::precise::rsqrt(total / 2048.0f + 1e-06f);
    device bfloat16_t* o = out + row * 2048u;
    for (uint j = 0; j < 8; j++) {
        const uint i = tid * 8 + j;
        o[i] = w[i] * bfloat16_t(v[j] * inverse);
    }
}

// Per (token, head): RMS norm over 128 with the head's weight, RoPE when
// rotary, exactly as the decode kernel's head_norm.  Grid: one simdgroup per
// (token, head) over heads 0..31 (q, in place) and 32..35 (k, written to
// kdst rows) ; v rows are copied to vdst by heads 36..39.
struct FpHeadArgs {
    uint M;
    uint pos0;          // position of token 0
    uint rotary;        // 0 for global layers
    uint dst_row0;      // first destination row index for k/v
    uint dst_head_stride;   // elements between kv heads in the destination
};

kernel void flint_head_norm_rope(device bfloat16_t* q [[buffer(0)]],          // [M][4096]
                                 const device bfloat16_t* k [[buffer(1)]],    // [M][512]
                                 const device bfloat16_t* v [[buffer(2)]],    // [M][512]
                                 const device bfloat16_t* qw [[buffer(3)]],
                                 const device bfloat16_t* kw [[buffer(4)]],
                                 device bfloat16_t* kdst [[buffer(5)]],
                                 device bfloat16_t* vdst [[buffer(6)]],
                                 constant FpHeadArgs& args [[buffer(7)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    const uint job = tg * 8 + sg;            // 8 simdgroups per threadgroup
    const uint token = job / 40, head = job % 40;
    if (token >= args.M) return;
    if (head >= 36) {
        const uint h = head - 36;
        const device bfloat16_t* src = v + (size_t)token * 512u + h * 128u;
        device bfloat16_t* dst = vdst + (size_t)h * args.dst_head_stride + (size_t)(args.dst_row0 + token) * 128u;
        for (uint j = 0; j < 4; j++) dst[lane * 4 + j] = src[lane * 4 + j];
        return;
    }
    const bool is_q = head < 32;
    const device bfloat16_t* src = is_q ? q + (size_t)token * 4096u + head * 128u
                                        : k + (size_t)token * 512u + (head - 32) * 128u;
    device bfloat16_t* dst = is_q ? q + (size_t)token * 4096u + head * 128u
                                  : kdst + (size_t)(head - 32) * args.dst_head_stride + (size_t)(args.dst_row0 + token) * 128u;
    const device bfloat16_t* w = is_q ? qw : kw;
    float xv[4];
    float squared = 0.0f;
    for (uint j = 0; j < 4; j++) { xv[j] = float(src[lane * 4 + j]); squared += xv[j] * xv[j]; }
    const float inverse = metal::precise::rsqrt(simd_sum(squared) / 128.0f + 1e-06f);
    if (args.rotary) {
        // Lanes own pairs (lane*2 + j, +64); the pair's values come from lanes lane/2 (first half) and 16 + lane/2.
        const uint position = args.pos0 + token;
        for (uint j = 0; j < 2; j++) {
            const uint pair = lane * 2 + j;
            const float xa = float(src[pair]), xb = float(src[pair + 64]);   // own pairs only
            const bfloat16_t a = w[pair] * bfloat16_t(xa * inverse);
            const bfloat16_t b = w[pair + 64] * bfloat16_t(xb * inverse);
            const float theta = float(position) * metal::exp2(-float(pair) / 64.0f * 13.287712379549449f);
            const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
            dst[pair] = bfloat16_t(float(a) * c - float(b) * s);
            dst[pair + 64] = bfloat16_t(float(a) * s + float(b) * c);
        }
    } else {
        for (uint j = 0; j < 4; j++) {
            const uint d = lane * 4 + j;
            dst[d] = w[d] * bfloat16_t(xv[j] * inverse);
        }
    }
}

// Copies n_rows rows of 4 heads x 128 between two [head][rows][128] layouts
// whose row indices are given per row by a small mapping: dst_row = f(src_row).
// Used for the sliding windows: gather ring slots into the window before a
// chunk and scatter the window's last rows back into ring slots after it.
// slot(p) = ring_initial >= 513 ? (p - ring_initial) mod 513 : p mod 513.
struct FpRingArgs {
    uint n_rows;
    uint src_head_stride, dst_head_stride;
    uint first_pos;        // absolute position of row 0 of the copy
    uint ring_initial;     // mapping for the ring side
    uint ring_side;        // 0: ring is the source (gather), 1: ring is the destination (scatter)
    uint linear_row0;      // row index in the linear (window) side of the first copied position
};

inline uint fp_ring_slot(uint p, uint initial) {
    if (initial >= 513u) {
        const uint d = p + 513u * 65536u - initial;   // keep the subtraction non-negative
        return d % 513u;
    }
    return p % 513u;
}

kernel void flint_ring_copy(const device bfloat16_t* K [[buffer(0)]],
                            const device bfloat16_t* V [[buffer(1)]],
                            device bfloat16_t* Kd [[buffer(2)]],
                            device bfloat16_t* Vd [[buffer(3)]],
                            constant FpRingArgs& args [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    // one thread per 8 elements: 4 heads x n_rows x 16 vectors
    const uint vecs_per_row = 16u;
    const uint per_head = args.n_rows * vecs_per_row;
    if (gid >= 4u * per_head) return;
    const uint head = gid / per_head, rem = gid % per_head;
    const uint r = rem / vecs_per_row, c = (rem % vecs_per_row) * 8u;
    const uint p = args.first_pos + r;
    const uint slot = fp_ring_slot(p, args.ring_initial);
    const uint lin = args.linear_row0 + r;
    const uint src_row = args.ring_side == 0 ? slot : lin;
    const uint dst_row = args.ring_side == 0 ? lin : slot;
    const size_t si = (size_t)head * args.src_head_stride + (size_t)src_row * 128u + c;
    const size_t di = (size_t)head * args.dst_head_stride + (size_t)dst_row * 128u + c;
    *(device vec<bfloat16_t, 8>*)(Kd + di) = *(const device vec<bfloat16_t, 8>*)(K + si);
    *(device vec<bfloat16_t, 8>*)(Vd + di) = *(const device vec<bfloat16_t, 8>*)(V + si);
}

// Last-row logits: x[2048] . W[150144][2048]^T with the decode kernel's
// vector arithmetic (qmv form: s * sum(x q) + b * sum(x)).  One simdgroup per
// four output rows; threadgroups of 8 simdgroups.
inline float fp_dot16(const device uchar* w, thread float* x, float s, float b, float sum) {
    const device ushort* p = (const device ushort*)w;
    float a = 0.0f;
    for (uint j = 0; j < 4; ++j)
        a += x[4 * j] * (p[j] & 15u) + x[4 * j + 1] * (p[j] & 240u) + x[4 * j + 2] * (p[j] & 3840u) +
             x[4 * j + 3] * (p[j] & 61440u);
    return s * a + sum * b;
}

kernel void flint_head_row(const device bfloat16_t* x [[buffer(0)]],
                           const device uchar* W [[buffer(1)]],
                           const device bfloat16_t* S [[buffer(2)]],
                           const device bfloat16_t* Bz [[buffer(3)]],
                           device bfloat16_t* out [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint first = (tg * 8 + sg) * 4;
    if (first >= N) return;
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < 2048u; block += 512) {
        const uint scalar = block + lane * 16;
        float v[16];
        float sum = 0.0f;
        for (uint i = 0; i < 16; i += 4) {
            const float x0 = float(x[scalar + i]), x1 = float(x[scalar + i + 1]), x2 = float(x[scalar + i + 2]),
                        x3 = float(x[scalar + i + 3]);
            sum += x0 + x1 + x2 + x3;
            v[i] = x0; v[i + 1] = x1 / 16.0f; v[i + 2] = x2 / 256.0f; v[i + 3] = x3 / 4096.0f;
        }
        for (uint r = 0; r < 4; ++r) {
            const uint row = first + r, gi = row * 32 + scalar / 64;
            a[r] += fp_dot16(W + (size_t)row * 1024 + scalar / 2, v, float(S[gi]), float(Bz[gi]), sum);
        }
    }
    for (uint r = 0; r < 4; ++r) {
        const float v = simd_sum(a[r]);
        if (lane == 0) out[first + r] = bfloat16_t(v);
    }
}

// bf16 -> f32 widening of n elements (logits readback helper).
kernel void flint_widen(const device bfloat16_t* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = float(in[gid]);
}
