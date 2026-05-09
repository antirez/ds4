struct ds4_metal_args_bin {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
};

constant short FC_bin_op [[function_constant(FC_BIN + 0)]];
constant short FC_bin_f  [[function_constant(FC_BIN + 1)]];
constant bool  FC_bin_rb [[function_constant(FC_BIN + 2)]];
constant bool  FC_bin_cb [[function_constant(FC_BIN + 3)]];

// Generic binary elementwise op with compile-time operation and broadcast
// modes. DS4 currently instantiates this as add, multiply, scalar multiply, and
// row division in the static graph.
template <typename T0, typename T1, typename T>
kernel void kernel_bin_fuse_impl(
        constant ds4_metal_args_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP FC_bin_op
#define FC_F  FC_bin_f
#define FC_RB FC_bin_rb
#define FC_CB FC_bin_cb

    if (FC_RB) {
        const uint i0 = tgpig.y*args.ne00 + tgpig.x;
        const uint i1 = FC_CB ? tgpig.x%args.ne10 : tgpig.x;

        device const T0 * src0_row = (device const T0 *) (src0);
        device       T  * dst_row  = (device       T  *) (dst);

        if (FC_F == 1) {
            device const T1 * src1_row = (device const T1 *) (src1 + args.o1[0]);

            if (FC_OP == 0) {
                dst_row[i0] = src0_row[i0] + src1_row[i1];
            }

            if (FC_OP == 1) {
                dst_row[i0] = src0_row[i0] - src1_row[i1];
            }

            if (FC_OP == 2) {
                dst_row[i0] = src0_row[i0] * src1_row[i1];
            }

            if (FC_OP == 3) {
                dst_row[i0] = src0_row[i0] / src1_row[i1];
            }
        } else {
            T0 res = src0_row[i0];

            if (FC_OP == 0) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res += ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 1) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res -= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 2) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res *= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 3) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res /= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            dst_row[i0] = res;
        }
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int i01 = tgpig.x;

        if (i01 >= args.ne01) {
            return;
        }

        const int i13 = i03%args.ne13;
        const int i12 = i02%args.ne12;
        const int i11 = i01%args.ne11;

        device const T0 * src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);
        device       T  * dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);

        if (FC_F == 1) {
            device const T1 * src1_ptr = (device const T1 *) (src1 + args.o1[0] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                if (FC_OP == 0) {
                    dst_ptr[i0] = src0_ptr[i0] + src1_ptr[i10];
                }

                if (FC_OP == 1) {
                    dst_ptr[i0] = src0_ptr[i0] - src1_ptr[i10];
                }

                if (FC_OP == 2) {
                    dst_ptr[i0] = src0_ptr[i0] * src1_ptr[i10];
                }

                if (FC_OP == 3) {
                    dst_ptr[i0] = src0_ptr[i0] / src1_ptr[i10];
                }
            }
        } else {
            device const T1 * src1_ptr[8];
            FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                src1_ptr[j] = (device const T1 *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);
            }

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                T res = src0_ptr[i0];

                if (FC_OP == 0) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res += src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 1) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res -= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 2) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res *= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 3) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res /= src1_ptr[j][i10];
                    }
                }

                dst_ptr[i0] = res;
            }
        }
    }

#undef FC_OP
#undef FC_F
#undef FC_RB
#undef FC_CB
}

typedef decltype(kernel_bin_fuse_impl<float, float, float>) kernel_bin_fuse_t;
// Host-visible F32 binary op; function constants specialize it per use site.
template [[host_name("kernel_bin_fuse_f32_f32_f32")]] kernel kernel_bin_fuse_t kernel_bin_fuse_impl<float, float, float>;
