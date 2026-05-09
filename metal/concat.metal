struct ds4_metal_args_concat {
    int32_t  ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;

    int32_t  ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12, nb13;

    int32_t  ne0, ne1, ne2, ne3;
    uint64_t nb0, nb1, nb2, nb3;

    int32_t dim;
};

inline int get_dim_offset(int dim, const ds4_metal_args_concat & a) {
    if (dim == 0) return a.ne00;
    if (dim == 1) return a.ne01;
    if (dim == 2) return a.ne02;
    return a.ne03;
}

kernel void kernel_concat(
        constant ds4_metal_args_concat & args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        uint3 gid[[threadgroup_position_in_grid]],
        ushort3 tid[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {

    const int i3 = gid.z;
    const int i2 = gid.y;
    const int i1 = gid.x;

    const int dim_offset = get_dim_offset(args.dim, args);

    for (int i0 = tid.x; i0 < args.ne0; i0 += ntg.x) {

        const bool in_src0 =
            (i0 < args.ne00 &&
             i1 < args.ne01 &&
             i2 < args.ne02 &&
             i3 < args.ne03);

        int s0 = i0;
        int s1 = i1;
        int s2 = i2;
        int s3 = i3;

        device const char * src = src1;

        if (in_src0) {
            src = src0;
        } else {
            if (args.dim == 0) s0 = i0 - dim_offset;
            if (args.dim == 1) s1 = i1 - dim_offset;
            if (args.dim == 2) s2 = i2 - dim_offset;
            if (args.dim == 3) s3 = i3 - dim_offset;
        }

        device const float * x =
            (device const float *)(src +
                s3 * (src == src0 ? args.nb03 : args.nb13) +
                s2 * (src == src0 ? args.nb02 : args.nb12) +
                s1 * (src == src0 ? args.nb01 : args.nb11) +
                s0 * (src == src0 ? args.nb00 : args.nb10));

        device float * y = (device float *)(dst +
            i3 * args.nb3 +
            i2 * args.nb2 +
            i1 * args.nb1 +
            i0 * args.nb0);

        *y = *x;
    }
}
