// Alternative instantiations of the Flint prefill attention template for
// tests/flint/bench_flint_attention (appended to metal/flint_prefill.metal at
// run time with --variants).  A name says the threadgroup's query count when
// it is not 64; the thread count is 32 x simdgroups.  flint_attention_old is
// the first kernel (64 queries, 4 simdgroups of 16 rows, Q in registers): its
// 16-row output tile spills registers and runs at 0.7 TFLOPS on M4 Max.

FP_ATTENTION_KERNEL(flint_attention_sg4_qsmem16, 4, 16, false, 32, true, 64, 1)
FP_ATTENTION_KERNEL(flint_attention_sg4_qreg8_bq32, 4, 8, true, 32, false, 1, 32)
FP_ATTENTION_KERNEL(flint_attention_sg4_qsmem8_bq32, 4, 8, false, 32, false, 32, 32)
FP_ATTENTION_KERNEL(flint_attention_sg4_qsmem8_bk64_bq32, 4, 8, false, 64, true, 32, 1)
FP_ATTENTION_KERNEL(flint_attention_sg4_qreg8_bk64_bq32, 4, 8, true, 64, true, 1, 1)
FP_ATTENTION_KERNEL(flint_attention_sg8_qreg8, 8, 8, true, 32, false, 1, 32)
FP_ATTENTION_KERNEL(flint_attention_sg8_qreg8_bk64, 8, 8, true, 64, true, 1, 1)
FP_ATTENTION_KERNEL(flint_attention_sg8_qsmem8_bk64, 8, 8, false, 64, true, 64, 1)

constant constexpr uint FP_ATT_BQ = 64;
constant constexpr uint FP_ATT_BK = 32;

[[max_total_threads_per_threadgroup(128)]]
kernel void flint_attention_old(const device bfloat16_t* Q [[buffer(0)]],        // [M][4096]
                            const device bfloat16_t* K0 [[buffer(1)]],
                            const device bfloat16_t* K1 [[buffer(2)]],
                            const device bfloat16_t* K2 [[buffer(3)]],
                            const device bfloat16_t* K3 [[buffer(4)]],
                            const device bfloat16_t* V0 [[buffer(5)]],
                            const device bfloat16_t* V1 [[buffer(6)]],
                            const device bfloat16_t* V2 [[buffer(7)]],
                            const device bfloat16_t* V3 [[buffer(8)]],
                            const device bfloat16_t* sinks [[buffer(9)]],    // [32] bf16
                            device bfloat16_t* O [[buffer(10)]],             // [M][4096]
                            constant FpAttnArgs& args [[buffer(11)]],
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
        const uint blk = kt0 >> 16;
        const device bfloat16_t* Kh = (blk == 0 ? K0 : blk == 1 ? K1 : blk == 2 ? K2 : K3) + (size_t)kv_head * args.kv_head_stride;
        const device bfloat16_t* Vh = (blk == 0 ? V0 : blk == 1 ? V1 : blk == 2 ? V2 : V3) + (size_t)kv_head * args.kv_head_stride;
        // 32 keys x 128 dims = 4096 elements per tile: each thread copies 4 x 8 of K and of V.
        for (ushort c = 0; c < 4; c++) {
            const ushort idx = tid + c * FP_THREADS;      // 0..511 vectors of 8
            const ushort r = idx >> 4, col = (idx & 15) * 8;
            const uint j = kt0 + r;
            vec<bfloat16_t, 8> kv8 = vec<bfloat16_t, 8>(bfloat16_t(0.0f)), vv8 = kv8;
            if (j < k_hi) {
                kv8 = *(const device vec<bfloat16_t, 8>*)(Kh + (size_t)(j & 65535u) * 128 + col);
                vv8 = *(const device vec<bfloat16_t, 8>*)(Vh + (size_t)(j & 65535u) * 128 + col);
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

