// DS4 Metal set-rows kernel used for KV writes.

struct ds4_metal_args_set_rows {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Scatters rows into the KV cache by token position. DS4 uses this after Q/K/V
// preparation so decode and later prefill chunks can attend to previous tokens.
template<typename T, typename TI>
kernel void kernel_set_rows_f(
        constant ds4_metal_args_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device T     * dst_row = (      device T     *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (T) src_row[ind];
    }
}

typedef decltype(kernel_set_rows_f<float, int64_t>) set_rows_f_t;

// Host-visible F32/I32 scatter variant used by KV-cache writes.
template [[host_name("kernel_set_rows_f32_i32")]] kernel set_rows_f_t kernel_set_rows_f<float, int32_t>;
