struct ds4_metal_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};

// RMSNorm over one activation row, optionally fusing the learned weight
// multiply. DS4 calls this before attention, before the FFN, and for plain
// diagnostics that need normalized but unweighted rows.
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ds4_metal_args_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;

// Host-visible RMSNorm variants: plain norm and norm multiplied by weight.
template [[host_name("kernel_rms_norm_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_mul_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;

struct ds4_metal_args_qkv_rms_norm {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
};

// Normalizes DS4's q-lora row and KV row in one dispatch.  The two reductions
// deliberately mirror kernel_rms_norm_mul_f32_4: Q uses the full 256-thread
// row shape for 1024 floats, while KV only has work in the first 128 lanes for
// its 512 floats.  This keeps the q/kv normalization math aligned with the
// standalone kernels while removing one tiny launch from the attention setup.
kernel void kernel_dsv4_qkv_rms_norm_f32_4(
        constant ds4_metal_args_qkv_rms_norm & args,
        device const float4 * q_src,
        device const float4 * q_weight,
        device       float4 * q_dst,
        device const float4 * kv_src,
        device const float4 * kv_weight,
        device       float4 * kv_dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const uint row = tgpig.x;
    const bool kv_task = tgpig.y != 0;
    const int n = kv_task ? args.kv_n : args.q_n;
    const int n4 = kv_task ? args.kv_n4 : args.q_n4;
    const uint64_t row_stride4 = (kv_task ? args.kv_row_stride : args.q_row_stride) / sizeof(float4);

    device const float4 * x = kv_task ? kv_src + row * row_stride4 : q_src + row * row_stride4;
    device const float4 * w = kv_task ? kv_weight : q_weight;
    device       float4 * y = kv_task ? kv_dst + row * row_stride4 : q_dst + row * row_stride4;

    float sumf = 0.0f;
    for (int i = tpitg.x; i < n4; i += ntg.x) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

#ifdef DS4_METAL_NORM_RSQRT_DISABLE
    // Match the formula used by kernel_rms_norm_fuse_impl above so both RMSNorm
    // entry points produce bit-identical scales. Hardware rsqrt() and 1.0f/sqrt()
    // can differ by ~1 ULP and that difference compounds across 43 layers.
    const float scale = 1.0f / sqrt(sumf / float(n) + args.eps);
#else
    const float scale = rsqrt(sumf / float(n) + args.eps);
#endif

    for (int i = tpitg.x; i < n4; i += ntg.x) {
        y[i] = (x[i] * scale) * w[i];
    }
}
