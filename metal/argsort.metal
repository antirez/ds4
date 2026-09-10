struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
    uint32_t causal_start;
    uint32_t causal_ratio;
    uint32_t block_width;
    uint32_t block_top_k;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_f32_i32_desc_causal")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order, bool causal = false, bool prefix = false>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);
    const uint width = causal ? min(uint(args.ne00),
        (args.causal_start + uint(i01) + 1u) / args.causal_ratio) : 0u;
    const int work_width = causal ? int(((width - 1u) / args.block_width) * args.block_top_k +
        min((width - 1u) % args.block_width + 1u, args.block_top_k)) : args.ne0;

    // A merged run can contribute at most 512 entries to the final result.
    // Keep the original run offsets so the comparison and tie order agree.
    const int read_limit = prefix ? min(args.len, 512) : args.len;
    const int len0 = MIN(read_limit, MAX(0, work_width - start));
    const int len1 = MIN(read_limit, MAX(0, work_width - (start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int limit = prefix ? min(args.top_k, 512) : args.top_k;
    const int k1 = MIN(MIN(k0 + chunk, total), limit);

    if (k0 >= limit) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal_prefix")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

// Adapted from IngeniousIdiocy/ds4 95eb218 (MIT).
// Bounded exact DSA selector.
// LEVER topk-select-fast: bounded radix fast path for the GLM-5.3 serial-decode
// DSA pool selection.
//
// WHAT THE PRODUCTION CHAIN EMITS, AND WHY THIS CAN REPRODUCE IT
// The block sort above is a comparison-exchange network whose predicate is
// `lo.val < hi.val` with out-of-range indices forced last; the fused merge
// takes the left run on `p0.val >= p1.val`.  On an input whose valid scores
// are all finite and whose 512 largest are pairwise distinct AND strictly
// greater than every other valid score, that whole chain is a TOTAL ORDER BY
// VALUE ALONE: each block emits its own top-512 in descending order, no global
// winner is lost to the per-block truncation (a block cannot hold more than
// 512 of the global top 512), and merging descending runs of distinct values
// yields the descending union.  Index order, block partition, chunking and the
// NaN-dependent merge-path partition never enter.  That is the accepted class,
// and it is exactly this path's acceptance predicate; ANY other input takes the
// unchanged production chain via an indirect dispatch whose threadgroup count
// this path writes.
//
// KEY.  bits ^ (bits>>31 ? 0xFFFFFFFF : 0x80000000) is order-isomorphic to the
// float order over the full signed range.  It would separate -0 from +0, which
// the production comparisons treat as EQUAL, so -0 is normalised to +0 first;
// a -0/+0 pair then shows up as an exact key tie and is rejected like any other
// tie.  Subnormals need no special handling: they are ordinary finite values
// with a monotone bit pattern and the comparator compares them normally.
// Non-finite values are classified by exponent bits over the COMPLETE valid
// input in pass 1 (nothing is skipped by the narrowing) and force fallback.
// SUBNORMALS.  The shader library is built with MTLCompileOptions defaults,
// i.e. fastMathEnabled = YES (ds4_metal.m ~7232; only DS4_METAL_MATH_SAFE, off
// by default, pins MTLMathModeSafe).  Under fast math the shipped comparator's
// `<` / `>=` may flush denormals to zero, which this bit-level key does not.
// Rather than guess which way the shipped comparator resolves a denormal, any
// subnormal anywhere in the valid input is treated as an unsupported class and
// takes the production chain.  Zero subnormals occurred in 88 M captured real
// scores.  The classification is a pure integer exponent/mantissa test, so
// fast math cannot alter it.
// ===========================================================================

struct ds4_metal_args_glm53_topk_fast {
    uint32_t n_comp;          // valid scores in the row
    uint32_t top_k;           // winners to emit (512 for the DSA pool select)
    uint32_t hist_bits;       // radix digit width for the single narrowing pass
    uint32_t cand_cap;        // candidate buffer capacity (entries)
    uint32_t fb_count;        // fallback dispatches whose grid this path gates
    uint32_t pad0;
    uint32_t fb_grid[16];     // (x,y) per fallback dispatch, in encode order
};

#define DS4_TOPK_FAST_FLAG_NONFINITE 1u   // Inf, NaN or subnormal
#define DS4_TOPK_FAST_FLAG_OVERFLOW  2u
#define DS4_TOPK_FAST_FLAG_NOBIN     4u
#define DS4_TOPK_FAST_FLAG_TIE       8u
#define DS4_TOPK_FAST_FLAG_SHORT     16u
#define DS4_TOPK_FAST_FLAG_BADIDX    32u

// ctrl[0] fallback flags, ctrl[1] candidate counter, ctrl[2] accepted (1/0),
// ctrl[3] candidate count observed by the finisher, ctrl[4] calls, ctrl[5]
// accepted calls.  ctrl[4]/ctrl[5] are cumulative telemetry, never cleared here.
//
// Cumulative rejection telemetry (added 2026-09-05;
// none of it is cleared per call, and none of it feeds any decision):
//   ctrl[6]  reject_flags_or  - OR of the reject word over EVERY rejected call
//                               in the process.  It is NOT "the last rejection";
//                               the old label `last_reject_flags` was wrong.
//   ctrl[7]  current consecutive-rejection run length
//   ctrl[8]  rejections whose reject word had NONFINITE (bit 0)
//   ctrl[9]  ...                                OVERFLOW  (bit 1)
//   ctrl[10] ...                                NOBIN     (bit 2)
//   ctrl[11] ...                                TIE       (bit 3)
//   ctrl[12] ...                                SHORT     (bit 4)
//   ctrl[13] ...                                BADIDX    (bit 5)
//   ctrl[14] longest consecutive-rejection run observed
// A call can set more than one cause, so the six per-cause counters sum to at
// least the rejected-call count; they are per-cause frequencies, not a
// partition.  The finisher is a single threadgroup and the selector calls are
// serialised by dispatch order, so lane 0 owns this state.

static inline uint ds4_topk_fast_key(float v) {
    uint b = as_type<uint>(v);
    if (b == 0x80000000u) b = 0u;                     // -0 == +0 for the comparator
    return (b >> 31) ? ~b : (b ^ 0x80000000u);
}

kernel void kernel_glm53_topk_fast_hist(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float  * scores,
        device atomic_uint  * ctrl,
        device atomic_uint  * hist,
        threadgroup atomic_uint * lhist [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tgpg  [[threadgroups_per_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth   = ntg.x;
    const uint tid   = tpitg.x;
    const uint nbins = 1u << args.hist_bits;
    const uint shift = 32u - args.hist_bits;

    for (uint b = tid; b < nbins; b += nth)
        atomic_store_explicit(&lhist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nonfinite = 0u;
    const uint stride = nth * tgpg.x;
    for (uint i = tgpig.x * nth + tid; i < args.n_comp; i += stride) {
        const float v = scores[i];
        const uint  b = as_type<uint>(v);
        const uint e = b & 0x7f800000u;
        const uint m = b & 0x007fffffu;
        if (e == 0x7f800000u) nonfinite = 1u;                   // Inf or NaN
        if (e == 0u && m != 0u) nonfinite = 1u;                 // subnormal (see above)
        atomic_fetch_add_explicit(&lhist[ds4_topk_fast_key(v) >> shift],
                                  1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint b = tid; b < nbins; b += nth) {
        const uint c = atomic_load_explicit(&lhist[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&hist[b], c, memory_order_relaxed);
    }
    if (nonfinite)
        atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_NONFINITE,
                                 memory_order_relaxed);
    if (tgpig.x == 0u && tid == 0u) {
        atomic_fetch_add_explicit(&ctrl[4], 1u, memory_order_relaxed);
        if (args.n_comp < args.top_k)
            atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_SHORT,
                                     memory_order_relaxed);
    }
}

/* Every threadgroup redundantly reduces the same integer histogram to the same
 * boundary bin, which is bit-identical by construction (§3.3: redundant
 * recomputation in identical structure).  That removes a dependent 1-TG scan
 * dispatch between the histogram and the gather. */
kernel void kernel_glm53_topk_fast_gather(
        constant ds4_metal_args_glm53_topk_fast & args,
        device const float  * scores,
        device atomic_uint  * ctrl,
        device const uint   * hist,
        device uint2        * cand,
        threadgroup uint    * sscan [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        uint3   tgpg  [[threadgroups_per_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth   = ntg.x;
    const uint tid   = tpitg.x;
    const uint nbins = 1u << args.hist_bits;
    const uint shift = 32u - args.hist_bits;
    const uint per   = (nbins + nth - 1u) / nth;      // bins owned per thread

    // sscan[0..nth) : per-thread sum of its contiguous bin block
    // sscan[nth]    : boundary bin, sscan[nth+1] : count strictly above it
    uint mine = 0u;
    const uint b0 = tid * per;
    for (uint b = b0; b < b0 + per && b < nbins; b++) mine += hist[b];
    sscan[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inclusive SUFFIX sum over sscan[0..nth) (Hillis-Steele, reversed)
    for (uint off = 1u; off < nth; off <<= 1) {
        const uint add = (tid + off < nth) ? sscan[tid + off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sscan[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { sscan[nth] = 0xffffffffu; sscan[nth + 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // count strictly above this thread's block = suffix sum of the next block
    const uint above_block = (tid + 1u < nth) ? sscan[tid + 1u] : 0u;
    if (above_block < args.top_k && b0 < nbins) {
        uint acc = above_block;
        const uint hi = min(b0 + per, nbins);
        for (uint b = hi; b-- > b0; ) {
            const uint c = hist[b];
            if (acc < args.top_k && acc + c >= args.top_k) {
                sscan[nth]      = b;
                sscan[nth + 1u] = acc;
                break;
            }
            acc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin_lo = sscan[nth];
    if (bin_lo == 0xffffffffu) {
        if (tid == 0u)
            atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_NOBIN,
                                     memory_order_relaxed);
        return;
    }

    const uint stride = nth * tgpg.x;
    for (uint i = tgpig.x * nth + tid; i < args.n_comp; i += stride) {
        const uint key = ds4_topk_fast_key(scores[i]);
        if ((key >> shift) >= bin_lo) {
            const uint slot = atomic_fetch_add_explicit(&ctrl[1], 1u,
                                                        memory_order_relaxed);
            if (slot < args.cand_cap) cand[slot] = uint2(key, i);
            else atomic_fetch_or_explicit(&ctrl[0], DS4_TOPK_FAST_FLAG_OVERFLOW,
                                          memory_order_relaxed);
        }
    }
}

/* One threadgroup.  Sorts the narrowed candidates descending by (key, index) -
 * a total order, so the result is deterministic regardless of the gather's
 * atomic arrival order - checks the acceptance predicate, and on acceptance
 * writes the ordered pool list. It also writes the fallback dispatches'
 * threadgroup counts (zero when accepted) and clears the scratch for the next
 * call. */
kernel void kernel_glm53_topk_fast_finish(
        constant ds4_metal_args_glm53_topk_fast & args,
        device atomic_uint * ctrl,
        device uint        * hist,
        device const uint2 * cand,
        device int32_t     * out_idx,
        device uint32_t    * indirect,
        threadgroup uint2  * shmem [[threadgroup(0)]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const uint nth = ntg.x;             // == args.cand_cap, one candidate per lane
    const uint tid = tpitg.x;

    const uint flags = atomic_load_explicit(&ctrl[0], memory_order_relaxed);
    const uint count = atomic_load_explicit(&ctrl[1], memory_order_relaxed);
    const uint have  = min(count, args.cand_cap);

    /* Register-resident bitonic sort, descending by (key, index).  Exactly the
     * structure of kernel_argsort_f32_i32_desc_pair above: every stage with
     * j < 32 is a simd_shuffle_xor and the rest use double-buffered threadgroup
     * staging, so 55 stages cost 15 barriers instead of 55.  (key, index) is a
     * total order, so the result does not depend on the order in which the
     * gather's atomic counter handed out candidate slots. */
    uint2 cur = (tid < have) ? cand[tid] : uint2(0u, 0xffffffffu);
    threadgroup uint2 *buf0 = shmem;
    threadgroup uint2 *buf1 = shmem + nth;
    bool use_buf1 = false;
    for (uint k = 2u; k <= nth; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            const uint partner = tid ^ j;
            uint2 other;
            if (j >= 32u) {
                threadgroup uint2 *buf = use_buf1 ? buf1 : buf0;
                buf[tid] = cur;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                other = buf[partner];
                use_buf1 = !use_buf1;
            } else {
                other.x = simd_shuffle_xor(cur.x, (ushort)j);
                other.y = simd_shuffle_xor(cur.y, (ushort)j);
            }
            const bool cur_greater = (cur.x != other.x) ? (cur.x > other.x)
                                                        : (cur.y < other.y);
            const bool want_greater = ((tid & k) == 0u) == (tid < partner);
            if (cur_greater != want_greater) cur = other;
        }
    }
    buf0[tid] = cur;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint2 *s = buf0;

    /* Acceptance predicate over the ordered candidates.
     *
     * REPAIRED 2026-09-05.  The previous version had
     * every lane write an ordinary threadgroup `uint bad` (1 for a tie, 2 for
     * an invalid index).  Those conflicting non-atomic stores are a data race
     * on a value that decides acceptance, the output-writing branch AND the
     * indirect fallback grids; a barrier around them does not serialise them
     * with each other, and the two causes overwrote one another so only one
     * could be reported.  It is now a threadgroup atomic OR of DISTINCT cause
     * bits, reduced before the barrier that publishes it, so every lane reads
     * ONE uniform value and both causes survive.  The bits are the same
     * DS4_TOPK_FAST_FLAG_* values used device-side, so the OR composes with
     * `flags` directly.  No floating-point operation changes. */
    threadgroup atomic_uint bad_bits;
    if (tid == 0u) atomic_store_explicit(&bad_bits, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ncheck = min(count, args.top_k + 1u);   // covers the cut tie at 511/512
    uint mine = 0u;
    for (uint i = tid; i + 1u < ncheck && i + 1u < nth; i += nth)
        if (s[i].x == s[i + 1u].x) mine |= DS4_TOPK_FAST_FLAG_TIE;
    for (uint i = tid; i < args.top_k && i < nth; i += nth)
        if (s[i].y >= args.n_comp) mine |= DS4_TOPK_FAST_FLAG_BADIDX;
    if (mine) atomic_fetch_or_explicit(&bad_bits, mine, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* one uniform acceptance value, published after the reduction */
    uint reject = flags | atomic_load_explicit(&bad_bits, memory_order_relaxed);
    if (count < args.top_k || count > args.cand_cap || nth != args.cand_cap)
        reject |= DS4_TOPK_FAST_FLAG_SHORT;
    const bool accept = (reject == 0u);

    if (accept) {
        for (uint i = tid; i < args.top_k; i += nth)
            out_idx[i] = (int32_t)s[i].y;
    }

    // fallback dispatch grids: zero when accepted, the real grid otherwise
    for (uint d = tid; d < args.fb_count && d < 8u; d += nth) {
        indirect[d * 3u + 0u] = accept ? 0u : args.fb_grid[d * 2u + 0u];
        indirect[d * 3u + 1u] = accept ? 0u : args.fb_grid[d * 2u + 1u];
        indirect[d * 3u + 2u] = accept ? 0u : 1u;
    }

    // clear the scratch for the next call
    const uint nbins = 1u << args.hist_bits;
    for (uint b = tid; b < nbins; b += nth) hist[b] = 0u;
    if (tid == 0u) {
        atomic_store_explicit(&ctrl[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[1], 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[2], accept ? 1u : 0u, memory_order_relaxed);
        atomic_store_explicit(&ctrl[3], count, memory_order_relaxed);
        if (accept) {
            atomic_fetch_add_explicit(&ctrl[5], 1u, memory_order_relaxed);
            atomic_store_explicit(&ctrl[7], 0u, memory_order_relaxed);   // run ends
        } else {
            atomic_fetch_or_explicit(&ctrl[6], reject, memory_order_relaxed);
            for (uint b = 0u; b < 6u; b++)
                if (reject & (1u << b))
                    atomic_fetch_add_explicit(&ctrl[8u + b], 1u, memory_order_relaxed);
            const uint run = atomic_load_explicit(&ctrl[7], memory_order_relaxed) + 1u;
            atomic_store_explicit(&ctrl[7], run, memory_order_relaxed);
            if (run > atomic_load_explicit(&ctrl[14], memory_order_relaxed))
                atomic_store_explicit(&ctrl[14], run, memory_order_relaxed);
        }
    }
}
