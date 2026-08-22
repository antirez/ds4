// DS4 Metal repeat kernel used for HC embedding expansion.

struct ds4_metal_args_repeat {
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
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Repeats a source row into the HC channel dimension. DS4 uses this when the
// token embedding has to become an HC activation block before layer 0.
template<typename T>
kernel void kernel_repeat(
        constant ds4_metal_args_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

// Host-visible F32 repeat used for HC expansion of embeddings.
template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;

// Qwen GQA expand: 4 kv heads (1024) -> 24 heads (6144), head_dim 256, factor 6.
struct ds4_gqa_args {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_gqa_expand_f32(
        device const float * src [[buffer(0)]],
        device float * dst [[buffer(1)]],
        constant ds4_gqa_args & args [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_head * args.head_dim;
    if (gid >= total) return;
    const uint h = gid / args.head_dim;
    const uint off = gid % args.head_dim;
    const uint group = args.n_head / args.n_head_kv;
    const uint kv_h = h / group;
    dst[gid] = src[kv_h * args.head_dim + off];
}

// Fill with zero (or arbitrary value) for attn_out zero path
kernel void kernel_fill_f32(
        device float * dst [[buffer(0)]],
        constant float & value [[buffer(1)]],
        constant uint & n [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = value;
}

// Interleave contiguous source rows into one slot of a packed row-major
// [row][slot][width] destination. Qwen target-layer capture invokes this once
// for each requested layer, avoiding one blit encoder per token row.
struct ds4_interleave_rows_args {
    uint32_t width;
    uint32_t rows;
    uint32_t slots;
    uint32_t slot;
};

kernel void kernel_interleave_rows_f32(
        device const float * src [[buffer(0)]],
        device       float * dst [[buffer(1)]],
        constant ds4_interleave_rows_args & args [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)args.rows * args.width;
    if ((ulong)gid >= total) return;
    const uint row = gid / args.width;
    const uint col = gid - row * args.width;
    dst[((ulong)row * args.slots + args.slot) * args.width + col] = src[gid];
}

// Qwen GQA causal attention decode with online FlashAttention softmax
struct ds4_qwen_attn_args {
    uint32_t pos;
    uint32_t max_ctx;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_qwen_gqa_attn_decode(
        device const float * q         [[buffer(0)]],
        device const float * k_new     [[buffer(1)]],
        device const float * v_new     [[buffer(2)]],
        device       float * k_cache   [[buffer(3)]],
        device       float * v_cache   [[buffer(4)]],
        device       float * heads_out [[buffer(5)]],
        constant ds4_qwen_attn_args & args [[buffer(6)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig;
    if (h >= args.n_head) return;
    const uint kv_dim = args.n_head_kv * args.head_dim;
    const uint group = args.n_head / args.n_head_kv;
    const uint kv_h = h / group;

    // One query head per KV group persists that group's new cache row.
    if (h % group == 0) {
        const uint kv_base = kv_h * args.head_dim;
        for (uint i = tiisg; i < args.head_dim; i += 32) {
            k_cache[(uint64_t)args.pos * kv_dim + kv_base + i] =
                k_new[kv_base + i];
            v_cache[(uint64_t)args.pos * kv_dim + kv_base + i] =
                v_new[kv_base + i];
        }
    }

    device const float * qh = q + (uint64_t)h * args.head_dim;

    // Each thread in simdgroup holds 8 floats (32 threads * 8 = 256 head_dim)
    float q_local[8];
    for (int j = 0; j < 8; j++) {
        q_local[j] = qh[tiisg * 8 + j];
    }

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[8] = {0.0f};
    const float kq_scale = 1.0f / sqrt((float)args.head_dim);

    for (uint t = 0; t <= args.pos; t++) {
        const bool current = t == args.pos;
        device const float * kt =
            current
                ? k_new + (uint64_t)kv_h * args.head_dim
                : k_cache + (uint64_t)t * kv_dim +
                      (uint64_t)kv_h * args.head_dim;
        float dot = 0.0f;
        for (int j = 0; j < 8; j++) {
            dot += q_local[j] * kt[tiisg * 8 + j];
        }
        dot = simd_sum(dot) * kq_scale;

        float m_curr = max(m_prev, dot);
        float alpha = exp(m_prev - m_curr);
        float beta = exp(dot - m_curr);
        l_prev = l_prev * alpha + beta;

        device const float * vt =
            current
                ? v_new + (uint64_t)kv_h * args.head_dim
                : v_cache + (uint64_t)t * kv_dim +
                      (uint64_t)kv_h * args.head_dim;
        for (int j = 0; j < 8; j++) {
            acc[j] = acc[j] * alpha + beta * vt[tiisg * 8 + j];
        }
        m_prev = m_curr;
    }

    float inv_l = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    device float * oh = heads_out + (uint64_t)h * args.head_dim;
    for (int j = 0; j < 8; j++) {
        oh[tiisg * 8 + j] = acc[j] * inv_l;
    }
}
