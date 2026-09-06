struct ds4_metal_args_glu {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};

kernel void kernel_swiglu_flat_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint i [[thread_position_in_grid]]) {
    if (i >= (uint)args.ne0) return;

    device const float * src0_f32 = (device const float *) src0 + args.i00;
    device const float * src1_f32 = (device const float *) src1 + args.i10;
    device       float * dst_f32  = (device       float *) dst;

    float x0 = src0_f32[i];
    float x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = min(x0, args.limit);
        x1 = clamp(x1, -args.limit, args.limit);
    }

    const float silu = ds4_silu(x0);
    dst_f32[i] = silu*x1*args.alpha;
}
