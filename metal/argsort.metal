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
template<ds4_sort_order order, bool canon = false>
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

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < args.ne00) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                /* canon: equal scores tie-break on the index (ascending), so
                 * the permutation is a total order over (score, idx) — the
                 * prerequisite for comparing against, or replacing with, any
                 * streaming/partial top-k whose comparator is totally
                 * ordered.  The merge kernel is already canonical (left run
                 * first on ties = index-ascending across runs). */
                const int32_t ia = shmem_i32[col];
                const int32_t ib_ = shmem_i32[ixj];
                const float va = ia < args.ne00 ? shmem_f32[ia - i00] : 0.0f;
                const float vb = ib_ < args.ne00 ? shmem_f32[ib_ - i00] : 0.0f;
                if ((col & k) == 0) {
                    if (ia >= args.ne00 ||
                       (ib_ <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            (va > vb || (canon && va == vb && ia > ib_)) :
                            (va < vb || (canon && va == vb && ia > ib_))))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (ib_ >= args.ne00 ||
                       (ia <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            (va < vb || (canon && va == vb && ia < ib_)) :
                            (va > vb || (canon && va == vb && ia < ib_))))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
// Canonical total order over (score desc, idx asc); tie order among equal
// scores is the only difference from the kernel above.
template [[host_name("kernel_argsort_f32_i32_desc_canon")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;

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
template<ds4_sort_order order>
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

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

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
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
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

// Exact streaming top-512 for wide prefill rows, the Metal port of the CUDA
// stream selector (45f4ef9): the threshold is the 512th-best key seen so
// far, so it can only discard candidates that cannot belong to the final top
// set.  Keys pack (score, idx) into one ulong with the same total order as
// the canonical argsort — score descending, index ascending — so the output
// list is bit-identical to the canon comparator path.  Buffer compaction
// uses ballot + prefix rank + one threadgroup atomic per simdgroup; the
// atomic makes buffer ORDER nondeterministic, but every emitted result
// passes through a full sort of totally-ordered keys, so the output is
// deterministic.  CUB's radix sort becomes an in-place threadgroup bitonic
// over the 2048-slot buffer; it runs only when the buffer nears capacity
// (a handful of times per row once the threshold warms) and once at the end.
static inline ulong ds4_topk_pack_key(float v, uint32_t idx) {
    const uint32_t u = as_type<uint32_t>(v);
    const uint32_t ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    return ((ulong)ordered << 32) | (ulong)(0xffffffffu - idx);
}

static inline void ds4_topk_bitonic_desc_2048(
        threadgroup ulong *buf,
        ushort tid) {
    for (uint k = 2; k <= 2048u; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < 2048u; i += 512u) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool descending = (i & k) == 0u;
                    const ulong a = buf[i];
                    const ulong b = buf[ixj];
                    if (descending ? (a < b) : (a > b)) {
                        buf[i] = b;
                        buf[ixj] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

[[max_total_threads_per_threadgroup(512)]]
kernel void kernel_dsv4_indexer_topk_stream512(
        constant ds4_metal_args_argsort & args,
        device const char * src0,
        device      int32_t * dst,
        threadgroup ulong * buf [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tid  [[thread_index_in_threadgroup]],
        ushort  lane [[thread_index_in_simdgroup]],
        ushort  sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr uint CAP = 2048u;
    constexpr uint TILE = 512u;
    const uint n_comp = (uint)args.ne00;
    const uint top_k = (uint)args.top_k;
    const uint t = tgpig.x;

    threadgroup uint *cnt = (threadgroup uint *)(buf + CAP);
    threadgroup ulong *thr_tg = (threadgroup ulong *)(buf + CAP + 1);
    threadgroup uint *sg_counts = (threadgroup uint *)(buf + CAP + 2); // [16]

    device const float *row =
        (device const float *)(src0 + args.nb01 * t);

    if (tid == 0) {
        cnt[0] = 0u;
        thr_tg[0] = 0ul;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start = (uint)(((ulong)(t + 1u) * 0x9E3779B9ul) % n_comp);
    for (uint base = 0; base < n_comp; base += TILE) {
        const ulong thr = thr_tg[0];
        const uint i = base + tid;
        ulong key = 0ul;
        bool take = false;
        if (i < n_comp) {
            uint c = start + i;
            if (c >= n_comp) c -= n_comp;
            key = ds4_topk_pack_key(row[c], c);
            take = key > thr;
        }
        /* Deterministic, atomic-free compaction: per-simdgroup accept
         * counts land in a threadgroup array, one thread prefix-sums the
         * sixteen entries, and every simdgroup writes at its reserved
         * offset.  All simd ops run in uniform control flow. */
        const uint ballot = (uint)(ulong)simd_ballot(take);
        const uint sg_count = popcount(ballot);
        const uint sg = (uint)sgid;
        if (lane == 0) sg_counts[sg] = sg_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = cnt[0];
            for (uint g = 0; g < 16u; g++) {
                const uint c = sg_counts[g];
                sg_counts[g] = acc;
                acc += c;
            }
            cnt[0] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (take) {
            const uint rank = popcount(ballot & ((1u << lane) - 1u));
            buf[sg_counts[sg] + rank] = key;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (cnt[0] > CAP - TILE) {
            const uint have = cnt[0];
            for (uint j = tid; j < CAP; j += 512u) {
                if (j >= have) buf[j] = 0ul;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ds4_topk_bitonic_desc_2048(buf, tid);
            if (tid == 0) {
                thr_tg[0] = buf[top_k - 1u];
                cnt[0] = top_k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    {
        const uint have = cnt[0];
        for (uint j = tid; j < CAP; j += 512u) {
            if (j >= have) buf[j] = 0ul;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ds4_topk_bitonic_desc_2048(buf, tid);
    }

    device int32_t *out = dst + (ulong)t * args.top_k;
    for (uint j = tid; j < top_k; j += 512u) {
        out[j] = (int32_t)(0xffffffffu - (uint32_t)buf[j]);
    }
}
