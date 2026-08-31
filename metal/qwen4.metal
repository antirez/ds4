// Qwen3.8-Flash-Next native kernels. Quantized matrices use the ordinary
// GGML block payloads emitted by GGUF: Q8_0 for dense projections and either
// Q4_K or mixed IQ2_XXS gate/up plus Q2_K down for routed experts. Inputs,
// outputs remain FP32.  The ordinary-decode path may keep recurrent state in
// BF16 (matching the reference MLX implementation); the verifier path retains
// the original FP32 state for byte-exact partial commits.

static inline float qwen4_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

static inline ushort qwen4_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return (ushort)(bits >> 16u);
}

struct qwen4_q8_0_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

struct qwen4_block_q4_0 {
    half d;
    uchar qs[16];
};

static inline float4 qwen4_q4_0_values4(
        device const qwen4_block_q4_0 *block,
        uint within) {
    const uint packed = within & 15u;
    const bool high = within >= 16u;
    const uchar4 codes = uchar4(
        block->qs[packed + 0u], block->qs[packed + 1u],
        block->qs[packed + 2u], block->qs[packed + 3u]);
    return float4(high ? codes >> 4u : codes & uchar4(15u)) - 8.0f;
}

/* Q4_0 decode traversal: each lane owns four contiguous values.  The 32
 * lanes therefore issue one coalesced 128-value sweep per loop iteration. */
static inline float qwen4_q4_0_dot_f32(
        device const qwen4_block_q4_0 *row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 4u; base < in_dim; base += 128u) {
        device const qwen4_block_q4_0 *block = row + base / 32u;
        const float4 q = qwen4_q4_0_values4(block, base & 31u);
        const float4 xv = *((device const float4 *)(x + base));
        sum = fma((float)block->d, dot(q, xv), sum);
    }
    return simd_sum(sum);
}

static inline float2 qwen4_q4_0_pair_dot_f32(
        device const qwen4_block_q4_0 *a,
        device const qwen4_block_q4_0 *b,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float2 sum = 0.0f;
    for (uint base = (uint)lane * 4u; base < in_dim; base += 128u) {
        const uint block_index = base / 32u;
        const uint within = base & 31u;
        const float4 xv = *((device const float4 *)(x + base));
        device const qwen4_block_q4_0 *ab = a + block_index;
        device const qwen4_block_q4_0 *bb = b + block_index;
        sum.x = fma((float)ab->d,
                    dot(qwen4_q4_0_values4(ab, within), xv), sum.x);
        sum.y = fma((float)bb->d,
                    dot(qwen4_q4_0_values4(bb, within), xv), sum.y);
    }
    return float2(simd_sum(sum.x), simd_sum(sum.y));
}

static inline float qwen4_q8_0_value(
        device const block_q8_0 *row,
        uint k) {
    device const block_q8_0 *block = row + k / QK8_0;
    return (float)block->d * (float)block->qs[k & (QK8_0 - 1u)];
}

/* One SIMDgroup owns one output row. Lanes walk 8-value slices so the block
 * delta is loaded once for each slice, matching the decode-friendly GGML
 * Q8_0 traversal without materializing dequantized weights. */
static inline float qwen4_q8_0_dot_f32(
        device const block_q8_0 *row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 8u; base < in_dim;
         base += 32u * 8u) {
        float dotq = 0.0f;
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = row + block_index;
        for (uint i = 0u; i < 8u; i++)
            dotq = fma((float)block->qs[in_block + i], x[base + i], dotq);
        sum = fma((float)block->d, dotq, sum);
    }
    return simd_sum(sum);
}

static inline float qwen4_q8_0_dot_bf16(
        device const block_q8_0 *row,
        device const ushort *x,
        uint in_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 8u; base < in_dim;
         base += 32u * 8u) {
        float dotq = 0.0f;
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = row + block_index;
        for (uint i = 0u; i < 8u; i++) {
            dotq = fma((float)block->qs[in_block + i],
                       qwen4_bf16_to_f32(x[base + i]), dotq);
        }
        sum = fma((float)block->d, dotq, sum);
    }
    return simd_sum(sum);
}

kernel void kernel_qwen4_q8_0_f32(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    device const float *xr = x + (ulong)token * args.in_dim;
    const float sum = qwen4_q8_0_dot_f32(wr, xr, args.in_dim, lane);
    if (lane == 0u)
        out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_qwen4_q8_0_f32_m1(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = sum;
}

/* Expert-independent wide-prefill specialization. Each SIMDgroup owns one
 * output row and advances eight token rows together, decoding each Q8_0
 * block slice once and retaining an independent FP32 reduction per token. */
kernel void kernel_qwen4_q8_0_f32_rows8(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token0 = tgpig.y * 8u;
    if (out_row >= args.out_dim || token0 >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    float4 sum0 = 0.0f;
    float4 sum1 = 0.0f;
    for (uint base = (uint)lane * 8u; base < args.in_dim;
         base += 32u * 8u) {
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = wr + block_index;
        const float delta = (float)block->d;
        for (uint i = 0u; i < 8u; i++) {
            const uint k = base + i;
            const float q = (float)block->qs[in_block + i] * delta;
            float4 xv0 = 0.0f;
            float4 xv1 = 0.0f;
            if (token0 + 0u < args.n_rows) xv0.x = x[((ulong)token0 + 0u) * args.in_dim + k];
            if (token0 + 1u < args.n_rows) xv0.y = x[((ulong)token0 + 1u) * args.in_dim + k];
            if (token0 + 2u < args.n_rows) xv0.z = x[((ulong)token0 + 2u) * args.in_dim + k];
            if (token0 + 3u < args.n_rows) xv0.w = x[((ulong)token0 + 3u) * args.in_dim + k];
            if (token0 + 4u < args.n_rows) xv1.x = x[((ulong)token0 + 4u) * args.in_dim + k];
            if (token0 + 5u < args.n_rows) xv1.y = x[((ulong)token0 + 5u) * args.in_dim + k];
            if (token0 + 6u < args.n_rows) xv1.z = x[((ulong)token0 + 6u) * args.in_dim + k];
            if (token0 + 7u < args.n_rows) xv1.w = x[((ulong)token0 + 7u) * args.in_dim + k];
            sum0 = fma(float4(q), xv0, sum0);
            sum1 = fma(float4(q), xv1, sum1);
        }
    }
    sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
    sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
    sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
    sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
    if (lane != 0u) return;
    if (token0 + 0u < args.n_rows) out[((ulong)token0 + 0u) * args.out_dim + out_row] = sum0.x;
    if (token0 + 1u < args.n_rows) out[((ulong)token0 + 1u) * args.out_dim + out_row] = sum0.y;
    if (token0 + 2u < args.n_rows) out[((ulong)token0 + 2u) * args.out_dim + out_row] = sum0.z;
    if (token0 + 3u < args.n_rows) out[((ulong)token0 + 3u) * args.out_dim + out_row] = sum0.w;
    if (token0 + 4u < args.n_rows) out[((ulong)token0 + 4u) * args.out_dim + out_row] = sum1.x;
    if (token0 + 5u < args.n_rows) out[((ulong)token0 + 5u) * args.out_dim + out_row] = sum1.y;
    if (token0 + 6u < args.n_rows) out[((ulong)token0 + 6u) * args.out_dim + out_row] = sum1.z;
    if (token0 + 7u < args.n_rows) out[((ulong)token0 + 7u) * args.out_dim + out_row] = sum1.w;
}

kernel void kernel_qwen4_q8_0_bf16(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const ushort      *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_bf16(
        weights + (ulong)out_row * row_blocks,
        x + (ulong)token * args.in_dim,
        args.in_dim, lane);
    if (lane == 0u)
        out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_qwen4_q8_0_f32_m1_silu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float value = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = value / (1.0f + exp(-value));
}

/* Wide split-K M=1 variant for narrow-output projections (the hyper-
 * connection down projection is N=320, K=10240): one 512-thread threadgroup
 * owns each output row and partitions K across sixteen simdgroups, giving
 * the GPU sixteen times the thread-level parallelism of the four-group
 * layout.  The cross-simdgroup reduction goes through threadgroup memory;
 * the accumulation order therefore differs from the four-group kernel by
 * construction, matching the batched-kernel parity policy. */
kernel void kernel_qwen4_q8_0_f32_m1_silu_wide(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    float sum = 0.0f;
    for (uint base = ((uint)sg * 32u + (uint)lane) * 8u;
         base < args.in_dim;
         base += (uint)nsg * 32u * 8u) {
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = wr + block_index;
        const float delta = (float)block->d;
        float dotq = 0.0f;
        for (uint i = 0u; i < 8u; i++)
            dotq = fma((float)block->qs[in_block + i], x[base + i], dotq);
        sum = fma(delta, dotq, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < (uint)nsg; group++)
            total += partials[group];
        out[out_row] = total / (1.0f + exp(-total));
    }
}

/* Wide split-K verifier-batch variant for narrow-output projections.
 * The generic per-row grid launches only (N/4)*rows groups — 320 for the
 * hyper-connection down projection at four rows — so each output row
 * instead gets a full 512-thread group that partitions K across sixteen
 * simdgroups and keeps the (up to eight) verifier rows in registers.
 * Unlike the wide-N rows-inner experiment, this multiplies the thread
 * count rather than dividing it. */
kernel void kernel_qwen4_q8_0_f32_silu_wide_rows(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const block_q8_0 *wr = weights + (ulong)out_row * row_blocks;
    float4 sum0 = 0.0f;
    float4 sum1 = 0.0f;
    for (uint base = ((uint)sg * 32u + (uint)lane) * 8u;
         base < args.in_dim;
         base += (uint)nsg * 32u * 8u) {
        const uint block_index = base / QK8_0;
        const uint in_block = base & (QK8_0 - 1u);
        device const block_q8_0 *block = wr + block_index;
        const float delta = (float)block->d;
        for (uint i = 0u; i < 8u; i++) {
            const uint k = base + i;
            const float q = (float)block->qs[in_block + i] * delta;
            float4 xv0 = 0.0f;
            float4 xv1 = 0.0f;
            if (args.n_rows > 0u) xv0.x = x[(ulong)0u * args.in_dim + k];
            if (args.n_rows > 1u) xv0.y = x[(ulong)1u * args.in_dim + k];
            if (args.n_rows > 2u) xv0.z = x[(ulong)2u * args.in_dim + k];
            if (args.n_rows > 3u) xv0.w = x[(ulong)3u * args.in_dim + k];
            if (args.n_rows > 4u) xv1.x = x[(ulong)4u * args.in_dim + k];
            if (args.n_rows > 5u) xv1.y = x[(ulong)5u * args.in_dim + k];
            if (args.n_rows > 6u) xv1.z = x[(ulong)6u * args.in_dim + k];
            if (args.n_rows > 7u) xv1.w = x[(ulong)7u * args.in_dim + k];
            sum0 = fma(float4(q), xv0, sum0);
            sum1 = fma(float4(q), xv1, sum1);
        }
    }
    sum0.x = simd_sum(sum0.x); sum0.y = simd_sum(sum0.y);
    sum0.z = simd_sum(sum0.z); sum0.w = simd_sum(sum0.w);
    sum1.x = simd_sum(sum1.x); sum1.y = simd_sum(sum1.y);
    sum1.z = simd_sum(sum1.z); sum1.w = simd_sum(sum1.w);
    /* partials layout: [simdgroup][token]. */
    if (lane == 0u) {
        partials[(uint)sg * 8u + 0u] = sum0.x;
        partials[(uint)sg * 8u + 1u] = sum0.y;
        partials[(uint)sg * 8u + 2u] = sum0.z;
        partials[(uint)sg * 8u + 3u] = sum0.w;
        partials[(uint)sg * 8u + 4u] = sum1.x;
        partials[(uint)sg * 8u + 5u] = sum1.y;
        partials[(uint)sg * 8u + 6u] = sum1.z;
        partials[(uint)sg * 8u + 7u] = sum1.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u || lane != 0u) return;
    for (uint token = 0u; token < args.n_rows && token < 8u; token++) {
        float total = 0.0f;
        for (uint group = 0u; group < (uint)nsg; group++)
            total += partials[group * 8u + token];
        out[(ulong)token * args.out_dim + out_row] =
            total / (1.0f + exp(-total));
    }
}

/* Tiny verifier-batch counterpart to the M=1 fusion above.  Keeping token
 * rows in the grid avoids materializing the down projection solely for a
 * following elementwise SiLU pass. */
kernel void kernel_qwen4_q8_0_f32_silu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0  *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float value = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks,
        x + (ulong)token * args.in_dim, args.in_dim, lane);
    if (lane == 0u) {
        out[(ulong)token * args.out_dim + out_row] =
            value / (1.0f + exp(-value));
    }
}

kernel void kernel_qwen4_q8_0_f32_m1_swiglu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0 *gate_weight [[buffer(1)]],
        device const block_q8_0 *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float gate = qwen4_q8_0_dot_f32(
        gate_weight + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    const float up = qwen4_q8_0_dot_f32(
        up_weight + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) out[out_row] = (gate / (1.0f + exp(-gate))) * up;
}

kernel void kernel_qwen4_q8_0_f32_swiglu(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0 *gate_weight [[buffer(1)]],
        device const block_q8_0 *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const uint row_blocks = args.in_dim / QK8_0;
    device const float *xr = x + (ulong)token * args.in_dim;
    const float gate = qwen4_q8_0_dot_f32(
        gate_weight + (ulong)out_row * row_blocks, xr, args.in_dim, lane);
    const float up = qwen4_q8_0_dot_f32(
        up_weight + (ulong)out_row * row_blocks, xr, args.in_dim, lane);
    if (lane == 0u) {
        out[(ulong)token * args.out_dim + out_row] =
            (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void kernel_qwen4_q8_0_f32_m1_pair(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const block_q8_0 *weight_a [[buffer(1)]],
        device const block_q8_0 *weight_b [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out_a [[buffer(4)]],
        device float *out_b [[buffer(5)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgroup * (uint)nsg + (uint)sg;
    if (out_row >= args.out_dim) return;
    const uint row_blocks = args.in_dim / QK8_0;
    const float a = qwen4_q8_0_dot_f32(
        weight_a + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    const float b = qwen4_q8_0_dot_f32(
        weight_b + (ulong)out_row * row_blocks, x, args.in_dim, lane);
    if (lane == 0u) { out_a[out_row] = a; out_b[out_row] = b; }
}

kernel void kernel_qwen4_q8_0_f32_m1_concat(
        constant qwen4_q8_0_args &args_a [[buffer(0)]],
        constant qwen4_q8_0_args &args_b [[buffer(1)]],
        device const block_q8_0 *weight_a [[buffer(2)]],
        device const block_q8_0 *weight_b [[buffer(3)]],
        device const float *x [[buffer(4)]],
        device float *out_a [[buffer(5)]],
        device float *out_b [[buffer(6)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint global_row = tgroup * (uint)nsg + (uint)sg;
    if (global_row >= args_a.out_dim + args_b.out_dim) return;
    const bool second = global_row >= args_a.out_dim;
    const uint out_row = second ? global_row - args_a.out_dim : global_row;
    const uint row_blocks = args_a.in_dim / QK8_0;
    device const block_q8_0 *weights = second ? weight_b : weight_a;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks, x, args_a.in_dim, lane);
    if (lane == 0u) {
        if (second) out_b[out_row] = sum;
        else out_a[out_row] = sum;
    }
}

kernel void kernel_qwen4_bf16_matmul_f32(
        constant qwen4_q8_0_args &args [[buffer(0)]],
        device const ushort      *weights [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *out [[buffer(3)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    device const ushort *wr = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    for (uint k = lane; k < args.in_dim; k += 32u)
        sum = fma(qwen4_bf16_to_f32(wr[k]), xr[k], sum);
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

struct qwen4_embedding_args {
    uint hidden_dim;
    uint vocab_size;
    uint n_tokens;
};

kernel void kernel_qwen4_q8_0_embedding_f32(
        constant qwen4_embedding_args &args [[buffer(0)]],
        device const block_q8_0       *weights [[buffer(1)]],
        device const int              *token_ids [[buffer(2)]],
        device float                  *out [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token_row = gid.y;
    if (dim >= args.hidden_dim || token_row >= args.n_tokens) return;
    const int token = token_ids[token_row];
    if (token < 0 || (uint)token >= args.vocab_size) {
        out[(ulong)token_row * args.hidden_dim + dim] = 0.0f;
        return;
    }
    const uint row_blocks = args.hidden_dim / QK8_0;
    device const block_q8_0 *row = weights + (ulong)(uint)token * row_blocks;
    out[(ulong)token_row * args.hidden_dim + dim] = qwen4_q8_0_value(row, dim);
}

struct qwen4_rows_args {
    uint width;
    uint rows;
    uint weight_rows;
    float eps;
};

kernel void kernel_qwen4_rms_norm_bf16_f32(
        constant qwen4_rows_args &args [[buffer(0)]],
        device const float       *x [[buffer(1)]],
        device const ushort      *weight [[buffer(2)]],
        device float             *out [[buffer(3)]],
        threadgroup float        *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    float sum = 0.0f;
    for (uint col = tid; col < args.width; col += (uint)nsg * 32u) {
        const float value = x[base + col];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u) partial[0] = rsqrt(sum / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const ulong weight_base =
        (ulong)(row % args.weight_rows) * args.width;
    for (uint col = tid; col < args.width; col += (uint)nsg * 32u) {
        out[base + col] = x[base + col] * scale *
            qwen4_bf16_to_f32(weight[weight_base + col]);
    }
}

struct qwen4_hc_norm_inject_args {
    uint n_tokens;
    uint hidden_dim;
    uint stream_count;
    float eps;
};

kernel void kernel_qwen4_hc_norm_inject_f32(
        constant qwen4_hc_norm_inject_args &args [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device const ushort *inject_weight [[buffer(3)]],
        device float *normalized [[buffer(4)]],
        device float *inject_partials [[buffer(5)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint total_rows = args.n_tokens * args.stream_count;
    if (row >= total_rows) return;
    const uint token = row / args.stream_count;
    const uint stream = row % args.stream_count;
    const ulong base = (ulong)row * args.hidden_dim;
    float sum = 0.0f;
    for (uint col = tid; col < args.hidden_dim; col += (uint)nsg * 32u) {
        const float value = streams[base + col];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.hidden_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const ulong norm_weight_base = (ulong)stream * args.hidden_dim;
    for (uint col = tid; col < args.hidden_dim; col += (uint)nsg * 32u) {
        normalized[base + col] = streams[base + col] * scale *
            qwen4_bf16_to_f32(norm_weight[norm_weight_base + col]);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (sg >= args.stream_count) return;
    const uint output = sg;
    const uint hc_dim = args.stream_count * args.hidden_dim;
    device const ushort *wr = inject_weight +
        (ulong)output * hc_dim + (ulong)stream * args.hidden_dim;
    device const float *xr = normalized + base;
    float dot = 0.0f;
    for (uint k = lane; k < args.hidden_dim; k += 32u)
        dot = fma(qwen4_bf16_to_f32(wr[k]), xr[k], dot);
    dot = simd_sum(dot);
    if (lane == 0u)
        inject_partials[((ulong)token * args.stream_count + stream) *
                        args.stream_count + output] = dot;
}

kernel void kernel_qwen4_hc_inject_reduce_f32(
        constant qwen4_hc_norm_inject_args &args [[buffer(0)]],
        device const float *inject_partials [[buffer(1)]],
        device float *inject [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint output = gid.x;
    const uint token = gid.y;
    if (output >= args.stream_count || token >= args.n_tokens) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < args.stream_count; stream++)
        sum += inject_partials[((ulong)token * args.stream_count + stream) *
                               args.stream_count + output];
    inject[(ulong)token * args.stream_count + output] =
        2.0f / (1.0f + exp(-sum));
}

kernel void kernel_qwen4_silu_f32(
        device const float *x [[buffer(0)]],
        device float *out [[buffer(1)]],
        constant ulong &elements [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if ((ulong)gid >= elements) return;
    const float value = x[gid];
    out[gid] = value / (1.0f + exp(-value));
}

kernel void kernel_qwen4_swiglu_f32(
        device const float *gate [[buffer(0)]],
        device const float *up [[buffer(1)]],
        device float *out [[buffer(2)]],
        constant ulong &elements [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if ((ulong)gid >= elements) return;
    const float value = gate[gid];
    out[gid] = (value / (1.0f + exp(-value))) * up[gid];
}

struct qwen4_hc_args {
    uint n_tokens;
    uint hidden_dim;
    uint stream_count;
};

struct qwen4_hc_up_mix_args {
    uint low_dim;
    uint hidden_dim;
    uint stream_count;
    uint n_tokens;
};

kernel void kernel_qwen4_repeat_streams_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *hidden [[buffer(1)]],
        device float *streams [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    const float value = hidden[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        streams[((ulong)token * args.stream_count + stream) *
                args.hidden_dim + dim] = value;
    }
}

kernel void kernel_qwen4_hc_mix_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *normalized [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device float *mixed [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[index]));
        sum = fma(normalized[index], gate, sum);
    }
    mixed[(ulong)token * args.hidden_dim + dim] =
        sum / (float)args.stream_count;
}

kernel void kernel_qwen4_hc_up_mix_bf16_m1_f32(
        constant qwen4_hc_up_mix_args &args [[buffer(0)]],
        device const ushort *weights [[buffer(1)]],
        device const float *low [[buffer(2)]],
        device const float *normalized [[buffer(3)]],
        device float *mixed [[buffer(4)]],
        threadgroup float *raw_gate [[threadgroup(0)]],
        uint dim [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (dim >= args.hidden_dim || sg >= args.stream_count) return;
    const uint output = (uint)sg * args.hidden_dim + dim;
    device const ushort *wr = weights + (ulong)output * args.low_dim;
    float sum = 0.0f;
    for (uint k = lane; k < args.low_dim; k += 32u) {
        sum = fma(qwen4_bf16_to_f32(wr[k]), low[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) raw_gate[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float value = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++) {
            const float gate = 1.0f / (1.0f + exp(-raw_gate[stream]));
            value = fma(normalized[(ulong)stream * args.hidden_dim + dim],
                        gate, value);
        }
        mixed[dim] = value / (float)args.stream_count;
    }
}

kernel void kernel_qwen4_hc_up_mix_q8_0_m1_f32(
        constant qwen4_hc_up_mix_args &args [[buffer(0)]],
        device const block_q8_0 *weights [[buffer(1)]],
        device const float *low [[buffer(2)]],
        device const float *normalized [[buffer(3)]],
        device float *mixed [[buffer(4)]],
        threadgroup float *raw_gate [[threadgroup(0)]],
        uint dim [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (dim >= args.hidden_dim || sg >= args.stream_count) return;
    const uint output = (uint)sg * args.hidden_dim + dim;
    const uint row_blocks = args.low_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)output * row_blocks, low, args.low_dim, lane);
    if (lane == 0u) raw_gate[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float value = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++) {
            const float gate = 1.0f / (1.0f + exp(-raw_gate[stream]));
            value = fma(normalized[(ulong)stream * args.hidden_dim + dim],
                        gate, value);
        }
        mixed[dim] = value / (float)args.stream_count;
    }
}

kernel void kernel_qwen4_hc_up_mix_q8_0_f32(
        constant qwen4_hc_up_mix_args &args [[buffer(0)]],
        device const block_q8_0 *weights [[buffer(1)]],
        device const float *low [[buffer(2)]],
        device const float *normalized [[buffer(3)]],
        device float *mixed [[buffer(4)]],
        threadgroup float *raw_gate [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint dim = tgpig.x;
    const uint token = tgpig.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens ||
        sg >= args.stream_count) return;
    const uint output = (uint)sg * args.hidden_dim + dim;
    const uint row_blocks = args.low_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)output * row_blocks,
        low + (ulong)token * args.low_dim, args.low_dim, lane);
    if (lane == 0u) raw_gate[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float value = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++) {
            const float gate = 1.0f / (1.0f + exp(-raw_gate[stream]));
            const ulong index =
                ((ulong)token * args.stream_count + stream) *
                args.hidden_dim + dim;
            value = fma(normalized[index], gate, value);
        }
        mixed[(ulong)token * args.hidden_dim + dim] =
            value / (float)args.stream_count;
    }
}

kernel void kernel_qwen4_hc_inject_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *raw_inject [[buffer(1)]],
        device float *inject [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong elements = (ulong)args.n_tokens * args.stream_count;
    if ((ulong)gid >= elements) return;
    inject[gid] = 2.0f / (1.0f + exp(-raw_inject[gid]));
}

kernel void kernel_qwen4_hc_write_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *block_output [[buffer(1)]],
        device const float *inject [[buffer(2)]],
        device float *streams [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.hidden_dim || token >= args.n_tokens) return;
    const float value = block_output[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        streams[index] = fma(value,
                             inject[(ulong)token * args.stream_count + stream],
                             streams[index]);
    }
}

kernel void kernel_qwen4_hc_write_partials_f32(
        constant qwen4_hc_args &args [[buffer(0)]],
        device const float *block_output [[buffer(1)]],
        device const float *inject_partials [[buffer(2)]],
        device float *streams [[buffer(3)]],
        threadgroup float *inject [[threadgroup(0)]],
        uint2 tgroup [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint token = tgroup.y;
    if (token >= args.n_tokens) return;
    if (tid < args.stream_count) {
        float sum = 0.0f;
        for (uint stream = 0; stream < args.stream_count; stream++)
            sum += inject_partials[
                ((ulong)token * args.stream_count + stream) *
                args.stream_count + tid];
        inject[tid] = 2.0f / (1.0f + exp(-sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint dim = tgroup.x * 256u + tid;
    if (dim >= args.hidden_dim) return;
    const float value = block_output[(ulong)token * args.hidden_dim + dim];
    for (uint stream = 0; stream < args.stream_count; stream++) {
        const ulong index = ((ulong)token * args.stream_count + stream) *
                            args.hidden_dim + dim;
        streams[index] = fma(value, inject[stream], streams[index]);
    }
}

/* Decode Gated DeltaNet output projection plus its hyper-connection write./* Decode Gated DeltaNet output projection plus its hyper-connection write.
 * The Q8_0 reduction and injection-partial reduction keep the same orders
 * as their standalone kernels; only the intermediate block write/read and a
 * second dispatch are removed. */
kernel void kernel_qwen4_q8_0_f32_m1_hc_write(
        constant qwen4_q8_0_args &projection [[buffer(0)]],
        constant qwen4_hc_args &hc [[buffer(1)]],
        device const block_q8_0 *weights [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const float *inject_partials [[buffer(4)]],
        device float *streams [[buffer(5)]],
        threadgroup float *inject [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (tid < hc.stream_count) {
        float partial = 0.0f;
        for (uint stream = 0; stream < hc.stream_count; stream++)
            partial += inject_partials[
                (ulong)stream * hc.stream_count + tid];
        inject[tid] = 2.0f / (1.0f + exp(-partial));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tgroup * (uint)nsg + sg;
    if (out_row >= projection.out_dim) return;
    const uint row_blocks = projection.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks,
        x, projection.in_dim, lane);
    if (lane == 0u) {
        for (uint stream = 0; stream < hc.stream_count; stream++) {
            const ulong at = (ulong)stream * hc.hidden_dim + out_row;
            streams[at] = fma(sum, inject[stream], streams[at]);
        }
    }
}

kernel void kernel_qwen4_q8_0_f32_hc_write(
        constant qwen4_q8_0_args &projection [[buffer(0)]],
        constant qwen4_hc_args &hc [[buffer(1)]],
        device const block_q8_0 *weights [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const float *inject_partials [[buffer(4)]],
        device float *streams [[buffer(5)]],
        threadgroup float *inject [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = tgpig.y;
    if (token >= projection.n_rows || token >= hc.n_tokens) return;
    if (tid < hc.stream_count) {
        float partial = 0.0f;
        for (uint stream = 0; stream < hc.stream_count; stream++) {
            partial += inject_partials[
                ((ulong)token * hc.stream_count + stream) *
                hc.stream_count + tid];
        }
        inject[tid] = 2.0f / (1.0f + exp(-partial));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint out_row = tgpig.x * (uint)nsg + sg;
    if (out_row >= projection.out_dim) return;
    const uint row_blocks = projection.in_dim / QK8_0;
    const float sum = qwen4_q8_0_dot_f32(
        weights + (ulong)out_row * row_blocks,
        x + (ulong)token * projection.in_dim, projection.in_dim, lane);
    if (lane == 0u) {
        for (uint stream = 0; stream < hc.stream_count; stream++) {
            const ulong at =
                ((ulong)token * hc.stream_count + stream) *
                hc.hidden_dim + out_row;
            streams[at] = fma(sum, inject[stream], streams[at]);
        }
    }
}

struct qwen4_ple_args {
    uint n_tokens;
    uint stream_count;
    uint hidden_dim;
    uint conv_width;
    uint dilation;
    uint state_len;
    uint has_mask;
    uint capture_slots;
    float eps;
};

kernel void kernel_qwen4_ple_gate_norm_f32(
        constant qwen4_ple_args &args [[buffer(0)]],
        device const float *hidden_streams [[buffer(1)]],
        device const float *key_raw [[buffer(2)]],
        device const float *value_raw [[buffer(3)]],
        device const ushort *key_weight [[buffer(4)]],
        device const ushort *query_weight [[buffer(5)]],
        device const ushort *conv_norm_weight [[buffer(6)]],
        device const uchar *mask [[buffer(7)]],
        device float *gated [[buffer(8)]],
        device float *gated_norm [[buffer(9)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint stream = group.y;
    if (token >= args.n_tokens || stream >= args.stream_count) return;
    const ulong stream_base =
        ((ulong)token * args.stream_count + stream) * args.hidden_dim;
    const ulong value_base = (ulong)token * args.hidden_dim;
    const ulong weight_base = (ulong)stream * args.hidden_dim;
    float key_sum = 0.0f;
    float query_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float key = key_raw[stream_base + dim];
        const float query = hidden_streams[stream_base + dim];
        const float value = value_raw[value_base + dim];
        key_sum = fma(key, key, key_sum);
        query_sum = fma(query, query, query_sum);
        value_sum = fma(value, value, value_sum);
    }
    key_sum = simd_sum(key_sum);
    query_sum = simd_sum(query_sum);
    value_sum = simd_sum(value_sum);
    if (lane == 0u) {
        partial[3u * sg] = key_sum;
        partial[3u * sg + 1u] = query_sum;
        partial[3u * sg + 2u] = value_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        key_sum = lane < nsg ? partial[3u * lane] : 0.0f;
        query_sum = lane < nsg ? partial[3u * lane + 1u] : 0.0f;
        value_sum = lane < nsg ? partial[3u * lane + 2u] : 0.0f;
        key_sum = simd_sum(key_sum);
        query_sum = simd_sum(query_sum);
        value_sum = simd_sum(value_sum);
        if (lane == 0u) {
            partial[3u * nsg] = rsqrt(
                key_sum / (float)args.hidden_dim + args.eps);
            partial[3u * nsg + 1u] = rsqrt(
                query_sum / (float)args.hidden_dim + args.eps);
            partial[3u * nsg + 2u] = value_sum / (float)args.hidden_dim;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float key_scale = partial[3u * nsg];
    const float query_scale = partial[3u * nsg + 1u];
    float dot = 0.0f;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float key = key_raw[stream_base + dim] * key_scale *
            qwen4_bf16_to_f32(key_weight[weight_base + dim]);
        const float query = hidden_streams[stream_base + dim] * query_scale *
            qwen4_bf16_to_f32(query_weight[weight_base + dim]);
        dot = fma(key, query, dot);
    }
    dot = simd_sum(dot);
    if (lane == 0u) partial[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        dot = lane < nsg ? partial[lane] : 0.0f;
        dot = simd_sum(dot) * rsqrt((float)args.hidden_dim);
        if (lane == 0u) {
            const float sign = dot > 0.0f ? 1.0f :
                               (dot < 0.0f ? -1.0f : 0.0f);
            const float transformed = sign * sqrt(max(abs(dot), 1.0e-6f));
            partial[0] = 1.0f / (1.0f + exp(-transformed));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = partial[0];
    const float norm_scale = rsqrt(
        gate * gate * partial[3u * nsg + 2u] + args.eps);
    const bool active = args.has_mask == 0u || mask[token] != 0u;
    for (uint dim = tid; dim < args.hidden_dim; dim += (uint)nsg * 32u) {
        const float value = active ? gate * value_raw[value_base + dim] : 0.0f;
        gated[stream_base + dim] = value;
        gated_norm[stream_base + dim] = active
            ? value * norm_scale *
              qwen4_bf16_to_f32(conv_norm_weight[weight_base + dim])
            : 0.0f;
    }
}

template <bool Capture>
static inline void qwen4_ple_dilated_conv(
        constant qwen4_ple_args &args,
        device const float *gated_norm,
        device const ushort *conv_weight,
        device float *conv_state,
        device float *out,
        device float *state_seq,
        uint channel) {
    const uint channels = args.stream_count * args.hidden_dim;
    if (channel >= channels || args.state_len != 9u ||
        args.conv_width != 4u || args.dilation != 3u) return;
    float history[9];
    for (uint i = 0; i < 9u; i++)
        history[i] = conv_state[(ulong)channel * 9u + i];
    for (uint token = 0; token < args.n_tokens; token++) {
        const ulong at = (ulong)token * channels + channel;
        const float current = gated_norm[at];
        float sum = current * qwen4_bf16_to_f32(
            conv_weight[(ulong)channel * 4u + 3u]);
        sum = fma(history[0], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u]), sum);
        sum = fma(history[3], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u + 1u]), sum);
        sum = fma(history[6], qwen4_bf16_to_f32(
                      conv_weight[(ulong)channel * 4u + 2u]), sum);
        for (uint i = 0; i < 8u; i++) history[i] = history[i + 1u];
        history[8] = current;
        if (Capture && token < args.capture_slots) {
            const ulong capture_base =
                ((ulong)token * channels + channel) * 9u;
            for (uint i = 0; i < 9u; i++)
                state_seq[capture_base + i] = history[i];
        }
        const float activated = sum / (1.0f + exp(-sum));
        out[at] += activated;
    }
    for (uint i = 0; i < 9u; i++)
        conv_state[(ulong)channel * 9u + i] = history[i];
}

kernel void kernel_qwen4_ple_dilated_conv_f32(
        constant qwen4_ple_args &args [[buffer(0)]],
        device const float *gated_norm [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device float *conv_state [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_ple_dilated_conv<false>(
        args, gated_norm, conv_weight, conv_state, out, conv_state, channel);
}

kernel void kernel_qwen4_ple_dilated_conv_capture_f32(
        constant qwen4_ple_args &args [[buffer(0)]],
        device const float *gated_norm [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device float *conv_state [[buffer(3)]],
        device float *out [[buffer(4)]],
        device float *state_seq [[buffer(5)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_ple_dilated_conv<true>(
        args, gated_norm, conv_weight, conv_state, out, state_seq, channel);
}

struct qwen4_moe_args {
    uint n_rows;
    uint n_experts;
    uint top_k;
    uint in_dim;
    uint expert_dim;
    uint out_dim;
    uint down_dim;
};

kernel void kernel_qwen4_moe_topk_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const float *router_logits [[buffer(1)]],
        device int *selected [[buffer(2)]],
        device float *weights [[buffer(3)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows || args.top_k > 16u ||
        args.n_experts > 512u || nsg > 8u) return;
    device const float *scores =
        router_logits + (ulong)row * args.n_experts;
    threadgroup float *partial_score = scratch;
    threadgroup int *partial_id = (threadgroup int *)(scratch + 8u);
    threadgroup float *chosen_score = scratch + 16u;
    threadgroup int *chosen_id = (threadgroup int *)(scratch + 32u);
    float local_score[2] = {-INFINITY, -INFINITY};
    int local_id[2] = {INT_MAX, INT_MAX};
    for (uint i = 0; i < 2u; i++) {
        const uint expert = tid + i * 256u;
        if (expert < args.n_experts) {
            local_score[i] = scores[expert];
            local_id[i] = (int)expert;
        }
    }

    for (uint slot = 0; slot < args.top_k; slot++) {
        float best = local_score[0];
        int id = local_id[0];
        if (local_score[1] > best ||
            (local_score[1] == best && local_id[1] < id)) {
            best = local_score[1];
            id = local_id[1];
        }
        for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
            const float other = simd_shuffle_down(best, offset);
            const int other_id = simd_shuffle_down(id, offset);
            if (lane + offset < 32u &&
                (other > best || (other == best && other_id < id))) {
                best = other;
                id = other_id;
            }
        }
        if (lane == 0u) {
            partial_score[sg] = best;
            partial_id[sg] = id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            best = lane < nsg ? partial_score[lane] : -INFINITY;
            id = lane < nsg ? partial_id[lane] : INT_MAX;
            for (ushort offset = 16u; offset != 0u; offset >>= 1u) {
                const float other = simd_shuffle_down(best, offset);
                const int other_id = simd_shuffle_down(id, offset);
                if (lane + offset < 32u &&
                    (other > best || (other == best && other_id < id))) {
                    best = other;
                    id = other_id;
                }
            }
            if (lane == 0u) {
                chosen_score[slot] = best;
                chosen_id[slot] = id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int remove = chosen_id[slot];
        if (local_id[0] == remove) {
            local_score[0] = -INFINITY;
            local_id[0] = INT_MAX;
        }
        if (local_id[1] == remove) {
            local_score[1] = -INFINITY;
            local_id[1] = INT_MAX;
        }
    }
    if (tid == 0u) {
        const float max_score = chosen_score[0];
        float selected_sum = 0.0f;
        for (uint slot = 0; slot < args.top_k; slot++) {
            chosen_score[slot] = exp(
                chosen_score[slot] - max_score);
            selected_sum += chosen_score[slot];
        }
        const float inv = 1.0f / selected_sum;
        for (uint slot = 0; slot < args.top_k; slot++) {
            const ulong at = (ulong)row * args.top_k + slot;
            selected[at] = chosen_id[slot];
            weights[at] = chosen_score[slot] * inv;
        }
    }
}

/* Deterministic token-major to expert-major map for the exact rows8 path.
 * Each expert scans the same staged route order, so hids remains stable across
 * runs and every work item names up to eight original token/slot assignments. */
kernel void kernel_qwen4_moe_map_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const int *selected [[buffer(1)]],
        device uint *counts [[buffer(2)]],
        device uint *hids [[buffer(3)]],
        device char *work [[buffer(4)]],
        threadgroup int *staged [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const uint expert = (uint)tid;
    uint count = 0u;
    device uint *expert_hids = hids + (ulong)expert * args.n_rows;

    for (uint row0 = 0u; row0 < args.n_rows; row0 += (uint)ntg) {
        const uint row = row0 + expert;
        if (row < args.n_rows) {
            for (uint slot = 0u; slot < args.top_k; slot++) {
                staged[(ulong)expert * args.top_k + slot] =
                    selected[(ulong)row * args.top_k + slot];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint rows = min((uint)ntg, args.n_rows - row0);
        for (uint local = 0u; local < rows; local++) {
            for (uint slot = 0u; slot < args.top_k; slot++) {
                if (staged[(ulong)local * args.top_k + slot] == (int)expert)
                    expert_hids[count++] = (row0 + local) * args.top_k + slot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    counts[expert] = count;
    threadgroup int *tile_counts = staged;
    tile_counts[expert] = (int)((count + 7u) / 8u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint work_base = 0u;
    for (uint i = 0u; i < expert; i++) work_base += tile_counts[i];
    device uint *work_count = (device uint *)work;
    device uint2 *work_items = (device uint2 *)(work + 8);
    for (uint tile = 0u; tile < (uint)tile_counts[expert]; tile++)
        work_items[work_base + tile] = uint2(expert, tile * 8u);
    if (expert + 1u == args.n_experts)
        work_count[0] = work_base + (uint)tile_counts[expert];
}

/* IQ2_XXS gate/up and Q2_K down rows use the ordinary GGML block payloads.
 * Keep the IQ2 reduction order aligned with the established generic kernel:
 * one lane owns a 32-value group, applies its block delta after the 32-value
 * dot, then the SIMD group performs the final reduction and 1/4 scale. */
static inline float2 qwen4_iq2_xxs_pair_dot(
        device const block_iq2_xxs *gate_row,
        device const block_iq2_xxs *up_row,
        device const float *x,
        uint logical_dim,
        ushort lane) {
    const uint groups32 = logical_dim / 32u;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint group32 = (uint)lane; group32 < groups32; group32 += 32u) {
        const uint block = group32 / 8u;
        const uint group = group32 & 7u;
        device const block_iq2_xxs *gb = gate_row + block;
        device const block_iq2_xxs *ub = up_row + block;
        device const ushort *gq = gb->qs + 4u * group;
        device const ushort *uq = ub->qs + 4u * group;
        device const uchar *gaux = (device const uchar *)gq;
        device const uchar *uaux = (device const uchar *)uq;
        const uint gs = (uint)gq[2] | ((uint)gq[3] << 16u);
        const uint us = (uint)uq[2] | ((uint)uq[3] << 16u);
        const float gd = (float)gb->d * (0.5f + (float)(gs >> 28u));
        const float ud = (float)ub->d * (0.5f + (float)(us >> 28u));
        float gate_group = 0.0f;
        float up_group = 0.0f;
        const uint xbase = group32 * 32u;
        for (uint sub = 0u; sub < 4u; sub++) {
            constant const uchar *ggrid =
                (constant const uchar *)(ds4_metal_iq2xxs_grid + gaux[sub]);
            constant const uchar *ugrid =
                (constant const uchar *)(ds4_metal_iq2xxs_grid + uaux[sub]);
            const uchar gsign = ds4_metal_ksigns_iq2xs[
                (gs >> (7u * sub)) & 127u];
            const uchar usign = ds4_metal_ksigns_iq2xs[
                (us >> (7u * sub)) & 127u];
            for (uint j = 0u; j < 8u; j++) {
                const float xv = x[xbase + sub * 8u + j];
                const float gv = (float)ggrid[j] *
                    ((gsign & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const float uv = (float)ugrid[j] *
                    ((usign & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                gate_group = fma(gv, xv, gate_group);
                up_group = fma(uv, xv, up_group);
            }
        }
        gate_sum = fma(gd, gate_group, gate_sum);
        up_sum = fma(ud, up_group, up_sum);
    }
    return float2(simd_sum(gate_sum) * 0.25f,
                  simd_sum(up_sum) * 0.25f);
}

static inline float qwen4_q2_k_dot(
        device const block_q2_K *row,
        device const float *x,
        uint logical_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint k = (uint)lane; k < logical_dim; k += 32u)
        sum = fma(ds4_glm_q2_K_value(row, k), x[k], sum);
    return simd_sum(sum);
}

kernel void kernel_qwen4_moe_iq2_xxs_gate_up_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_iq2_xxs *gate_weight [[buffer(1)]],
        device const block_iq2_xxs *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const uint *counts [[buffer(4)]],
        device const uint *hids [[buffer(5)]],
        device const char *work [[buffer(6)]],
        device float *mid [[buffer(7)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.expert_dim) return;

    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    uint tokens[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) {
        assignments[r] = expert_hids[local0 + r];
        tokens[r] = assignments[r] / args.top_k;
    }

    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)expert * args.expert_dim + output;
    device const block_iq2_xxs *gwr = gate_weight + weight_row * row_blocks;
    device const block_iq2_xxs *uwr = up_weight + weight_row * row_blocks;
    float4 gate0 = 0.0f, gate1 = 0.0f;
    float4 up0 = 0.0f, up1 = 0.0f;
    const uint groups32 = args.in_dim / 32u;
    for (uint group32 = (uint)lane; group32 < groups32; group32 += 32u) {
        const uint block = group32 / 8u;
        const uint group = group32 & 7u;
        device const block_iq2_xxs *gb = gwr + block;
        device const block_iq2_xxs *ub = uwr + block;
        device const ushort *gq = gb->qs + 4u * group;
        device const ushort *uq = ub->qs + 4u * group;
        device const uchar *gaux = (device const uchar *)gq;
        device const uchar *uaux = (device const uchar *)uq;
        const uint gs = (uint)gq[2] | ((uint)gq[3] << 16u);
        const uint us = (uint)uq[2] | ((uint)uq[3] << 16u);
        const float gd = (float)gb->d * (0.5f + (float)(gs >> 28u));
        const float ud = (float)ub->d * (0.5f + (float)(us >> 28u));
        float4 gate_group0 = 0.0f, gate_group1 = 0.0f;
        float4 up_group0 = 0.0f, up_group1 = 0.0f;
        const uint xbase = group32 * 32u;
        for (uint sub = 0u; sub < 4u; sub++) {
            constant const uchar *ggrid =
                (constant const uchar *)(ds4_metal_iq2xxs_grid + gaux[sub]);
            constant const uchar *ugrid =
                (constant const uchar *)(ds4_metal_iq2xxs_grid + uaux[sub]);
            const uchar gsign = ds4_metal_ksigns_iq2xs[
                (gs >> (7u * sub)) & 127u];
            const uchar usign = ds4_metal_ksigns_iq2xs[
                (us >> (7u * sub)) & 127u];
            for (uint j = 0u; j < 8u; j++) {
                const uint k = xbase + sub * 8u + j;
                const float gv = (float)ggrid[j] *
                    ((gsign & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const float uv = (float)ugrid[j] *
                    ((usign & ds4_metal_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                for (uint r = 0u; r < valid; r++) {
                    const float xv = x[(ulong)tokens[r] * args.in_dim + k];
                    if (r < 4u) {
                        gate_group0[r] = fma(gv, xv, gate_group0[r]);
                        up_group0[r] = fma(uv, xv, up_group0[r]);
                    } else {
                        gate_group1[r - 4u] =
                            fma(gv, xv, gate_group1[r - 4u]);
                        up_group1[r - 4u] =
                            fma(uv, xv, up_group1[r - 4u]);
                    }
                }
            }
        }
        gate0 = fma(float4(gd), gate_group0, gate0);
        gate1 = fma(float4(gd), gate_group1, gate1);
        up0 = fma(float4(ud), up_group0, up0);
        up1 = fma(float4(ud), up_group1, up1);
    }
    for (uint r = 0u; r < valid; r++) {
        const float gate =
            simd_sum(r < 4u ? gate0[r] : gate1[r - 4u]) * 0.25f;
        const float up =
            simd_sum(r < 4u ? up0[r] : up1[r - 4u]) * 0.25f;
        if (lane == 0u)
            mid[(ulong)assignments[r] * args.expert_dim + output] =
                (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void kernel_qwen4_moe_q2_k_down_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const uint *counts [[buffer(3)]],
        device const uint *hids [[buffer(4)]],
        device const char *work [[buffer(5)]],
        device float *contributions [[buffer(6)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.out_dim) return;
    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) assignments[r] = expert_hids[local0 + r];

    const uint row_blocks = args.down_dim / 256u;
    const ulong weight_row = (ulong)expert * args.out_dim + output;
    device const block_q2_K *wr = down_weight + weight_row * row_blocks;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    /* k=[640,768) is the physical Q2_K alignment tail and is exactly zero. */
    for (uint k = (uint)lane; k < args.expert_dim; k += 32u) {
        const float w = ds4_glm_q2_K_value(wr, k);
        for (uint r = 0u; r < valid; r++) {
            const float xv = mid[(ulong)assignments[r] * args.expert_dim + k];
            if (r < 4u) sum0[r] = fma(w, xv, sum0[r]);
            else sum1[r - 4u] = fma(w, xv, sum1[r - 4u]);
        }
    }
    for (uint r = 0u; r < valid; r++) {
        const float sum = simd_sum(r < 4u ? sum0[r] : sum1[r - 4u]);
        if (lane == 0u)
            contributions[(ulong)assignments[r] * args.out_dim + output] = sum;
    }
}

kernel void kernel_qwen4_moe_iq2_xxs_gate_up(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_iq2_xxs *gate_weight [[buffer(1)]],
        device const block_iq2_xxs *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const int *selected [[buffer(4)]],
        device float *mid [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    const uint slot = tgpig.z;
    if (output >= args.expert_dim || row >= args.n_rows || slot >= args.top_k) return;
    const ulong assignment = (ulong)row * args.top_k + slot;
    const int expert = selected[assignment];
    if (expert < 0 || (uint)expert >= args.n_experts) return;
    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)(uint)expert * args.expert_dim + output;
    const float2 pair = qwen4_iq2_xxs_pair_dot(
        gate_weight + weight_row * row_blocks,
        up_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    if (lane == 0u)
        mid[assignment * args.expert_dim + output] =
            (pair.x / (1.0f + exp(-pair.x))) * pair.y;
}

kernel void kernel_qwen4_moe_q2_k_down(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint row_blocks = args.down_dim / 256u;
    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
        const float sum = qwen4_q2_k_dot(
            down_weight + weight_row * row_blocks,
            mid + assignment * args.expert_dim,
            args.expert_dim, lane);
        total = fma(sum, selected_weights[assignment], total);
    }
    if (lane == 0u) out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q2_k_down_split(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x;
    const uint row = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)row * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float sum = 0.0f;
        if (output < args.out_dim && row < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 256u;
            const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
            sum = qwen4_q2_k_dot(
                down_weight + weight_row * row_blocks,
                mid + assignment * args.expert_dim,
                args.expert_dim, lane);
        }
        if (lane == 0u) partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u && output < args.out_dim && row < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)row * args.top_k + slot;
            total = fma(partials[slot], selected_weights[assignment], total);
        }
        out[(ulong)row * args.out_dim + output] = total;
    }
}

/* M=1 variant that retains one SIMDgroup per selected expert but evaluates
 * two adjacent output rows per group.  This preserves top-k parallelism while
 * halving threadgroup scheduling and reusing the expert activation. */
kernel void kernel_qwen4_moe_q2_k_down_split_rows2(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output0 = tgpig.x * 2u;
    const uint token = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)token * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float2 sums = 0.0f;
        if (output0 < args.out_dim && token < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 256u;
            const ulong expert_row =
                (ulong)(uint)expert * args.out_dim + output0;
            device const block_q2_K *weight0 =
                down_weight + expert_row * row_blocks;
            device const block_q2_K *weight1 = weight0 + row_blocks;
            device const float *activation =
                mid + assignment * args.expert_dim;
            float2 lane_sums = 0.0f;
            for (uint k = (uint)lane; k < args.expert_dim; k += 32u) {
                const float xv = activation[k];
                lane_sums[0] = fma(ds4_glm_q2_K_value(weight0, k),
                                   xv, lane_sums[0]);
                if (output0 + 1u < args.out_dim)
                    lane_sums[1] = fma(ds4_glm_q2_K_value(weight1, k),
                                       xv, lane_sums[1]);
            }
            sums = float2(simd_sum(lane_sums[0]),
                          simd_sum(lane_sums[1]));
        }
        if (lane == 0u) {
            partials[(uint)sg * 2u] = sums[0];
            partials[(uint)sg * 2u + 1u] = sums[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2u && output0 + tid < args.out_dim && token < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)token * args.top_k + slot;
            total = fma(partials[slot * 2u + tid],
                        selected_weights[assignment], total);
        }
        out[(ulong)token * args.out_dim + output0 + tid] = total;
    }
}

kernel void kernel_qwen4_moe_q2_k_down_split_rows4(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output0 = tgpig.x * 4u;
    const uint token = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)token * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float4 sums = 0.0f;
        if (output0 < args.out_dim && token < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 256u;
            const ulong expert_row =
                (ulong)(uint)expert * args.out_dim + output0;
            device const block_q2_K *weight0 =
                down_weight + expert_row * row_blocks;
            device const block_q2_K *weight1 = weight0 + row_blocks;
            device const block_q2_K *weight2 = weight1 + row_blocks;
            device const block_q2_K *weight3 = weight2 + row_blocks;
            device const float *activation =
                mid + assignment * args.expert_dim;
            float4 lane_sums = 0.0f;
            for (uint k = (uint)lane; k < args.expert_dim; k += 32u) {
                const float xv = activation[k];
                lane_sums[0] = fma(ds4_glm_q2_K_value(weight0, k),
                                   xv, lane_sums[0]);
                if (output0 + 1u < args.out_dim)
                    lane_sums[1] = fma(ds4_glm_q2_K_value(weight1, k),
                                       xv, lane_sums[1]);
                if (output0 + 2u < args.out_dim)
                    lane_sums[2] = fma(ds4_glm_q2_K_value(weight2, k),
                                       xv, lane_sums[2]);
                if (output0 + 3u < args.out_dim)
                    lane_sums[3] = fma(ds4_glm_q2_K_value(weight3, k),
                                       xv, lane_sums[3]);
            }
            sums = float4(simd_sum(lane_sums[0]),
                          simd_sum(lane_sums[1]),
                          simd_sum(lane_sums[2]),
                          simd_sum(lane_sums[3]));
        }
        if (lane == 0u) {
            for (uint r = 0u; r < 4u; r++)
                partials[(uint)sg * 4u + r] = sums[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u && output0 + tid < args.out_dim && token < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)token * args.top_k + slot;
            total = fma(partials[slot * 4u + tid],
                        selected_weights[assignment], total);
        }
        out[(ulong)token * args.out_dim + output0 + tid] = total;
    }
}

/* Decode-specialized Q2_K down projection.  Two SIMDgroups share each
 * threadgroup and each SIMDgroup evaluates four adjacent output rows.  This
 * keeps the GGML block decoder vectorized while reducing the M=1 launch from
 * one 320-thread threadgroup per output row to one 64-thread threadgroup per
 * eight rows.  The 640-value logical expert activation is handled explicitly;
 * the final half of its third physical 256-value block is zero padding. */
kernel void kernel_qwen4_moe_q2_k_down_rows8_m1(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q2_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr short rows_per_simd = 4;
    const uint row0 = (tgpig.x * 2u + (uint)sg) * (uint)rows_per_simd;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_rows) return;

    const short ix = lane / 8;
    const short it = lane % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short scale_column = (8 * ir) / 16;
    const uint row_blocks = args.down_dim / 256u;
    const ulong selected_base = (ulong)token * args.top_k;
    float total[rows_per_simd] = {0.0f};

    for (uint slot = 0u; slot < args.top_k; slot++) {
        const ulong assignment = selected_base + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        device const float *activation =
            mid + assignment * args.expert_dim;
        device const block_q2_K *expert_rows =
            down_weight +
            ((ulong)(uint)expert * args.out_dim + row0) * row_blocks;
        float partial[rows_per_simd] = {0.0f};

        for (uint block = (uint)ix; block < row_blocks; block += 4u) {
            float values[32];
            float4 sums = 0.0f;
            const uint block_base = block * 256u;
            const uint lane_base = 128u * (uint)iq + 8u * (uint)ir;
            for (short i = 0; i < 8; i++) {
                const uint p0 = lane_base + (uint)i;
                const uint p1 = p0 + 32u;
                const uint p2 = p0 + 64u;
                const uint p3 = p0 + 96u;
                values[i + 0] = block_base + p0 < args.expert_dim
                    ? activation[block_base + p0] : 0.0f;
                values[i + 8] = block_base + p1 < args.expert_dim
                    ? activation[block_base + p1] : 0.0f;
                values[i + 16] = block_base + p2 < args.expert_dim
                    ? activation[block_base + p2] : 0.0f;
                values[i + 24] = block_base + p3 < args.expert_dim
                    ? activation[block_base + p3] : 0.0f;
                sums[0] += values[i + 0];
                sums[1] += values[i + 8];
                sums[2] += values[i + 16];
                sums[3] += values[i + 24];
            }

            for (short r = 0; r < rows_per_simd &&
                            row0 + (uint)r < args.out_dim; r++) {
                device const block_q2_K *weight =
                    expert_rows + (ulong)r * row_blocks + block;
                device const uchar *scales =
                    (device const uchar *)weight->scales +
                    8 * iq + scale_column;
                device const ushort *quants =
                    (device const ushort *)weight->qs +
                    16 * iq + 4 * ir;
                float4 acc_low = 0.0f;
                float4 acc_high = 0.0f;
                for (short i = 0; i < 8; i += 2) {
                    const ushort packed = quants[i / 2];
                    acc_low[0] += values[i + 0] * (packed & 0x0003u);
                    acc_high[0] += values[i + 1] * (packed & 0x0300u);
                    acc_low[1] += values[i + 8] * (packed & 0x000cu);
                    acc_high[1] += values[i + 9] * (packed & 0x0c00u);
                    acc_low[2] += values[i + 16] * (packed & 0x0030u);
                    acc_high[2] += values[i + 17] * (packed & 0x3000u);
                    acc_low[3] += values[i + 24] * (packed & 0x00c0u);
                    acc_high[3] += values[i + 25] * (packed & 0xc000u);
                }
                const float d = (float)weight->d;
                const float m = (float)weight->dmin * (1.0f / 16.0f);
                partial[r] +=
                    d * ((acc_low[0] + acc_high[0] * (1.0f / 256.0f)) *
                             (float)(scales[0] & 0x0fu) +
                         (acc_low[1] + acc_high[1] * (1.0f / 256.0f)) *
                             (float)(scales[2] & 0x0fu) * (1.0f / 4.0f) +
                         (acc_low[2] + acc_high[2] * (1.0f / 256.0f)) *
                             (float)(scales[4] & 0x0fu) * (1.0f / 16.0f) +
                         (acc_low[3] + acc_high[3] * (1.0f / 256.0f)) *
                             (float)(scales[6] & 0x0fu) * (1.0f / 64.0f)) -
                    m * (sums[0] * (float)(scales[0] & 0xf0u) +
                         sums[1] * (float)(scales[2] & 0xf0u) +
                         sums[2] * (float)(scales[4] & 0xf0u) +
                         sums[3] * (float)(scales[6] & 0xf0u));
            }
        }

        const float route = selected_weights[assignment];
        for (short r = 0; r < rows_per_simd &&
                        row0 + (uint)r < args.out_dim; r++) {
            total[r] = fma(simd_sum(partial[r]), route, total[r]);
        }
    }

    if (lane == 0u) {
        for (short r = 0; r < rows_per_simd &&
                        row0 + (uint)r < args.out_dim; r++) {
            out[(ulong)token * args.out_dim + row0 + (uint)r] = total[r];
        }
    }
}

/* Q4_K expert rows use the same GGML decoder as the established GLM path.
 * Qwen differs in routing width (512/top-10), so the fused scheduling remains
 * local while the block interpretation is shared. */
static inline float qwen4_q4_k_dot(
        device const block_q4_K *row,
        device const float *x,
        uint logical_dim,
        ushort lane) {
    float sum = 0.0f;
    for (uint k = (uint)lane; k < logical_dim; k += 32u)
        sum = fma(ds4_glm_q4_K_value(row, k), x[k], sum);
    return simd_sum(sum);
}

kernel void kernel_qwen4_moe_q4_k_gate_up_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *gate_weight [[buffer(1)]],
        device const block_q4_K *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const uint *counts [[buffer(4)]],
        device const uint *hids [[buffer(5)]],
        device const char *work [[buffer(6)]],
        device float *mid [[buffer(7)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.expert_dim) return;

    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    uint tokens[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) {
        assignments[r] = expert_hids[local0 + r];
        tokens[r] = assignments[r] / args.top_k;
    }

    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)expert * args.expert_dim + output;
    device const block_q4_K *gwr = gate_weight + weight_row * row_blocks;
    device const block_q4_K *uwr = up_weight + weight_row * row_blocks;
    float4 gate0 = 0.0f, gate1 = 0.0f;
    float4 up0 = 0.0f, up1 = 0.0f;
    for (uint k = (uint)lane; k < args.in_dim; k += 32u) {
        const float gw = ds4_glm_q4_K_value(gwr, k);
        const float uw = ds4_glm_q4_K_value(uwr, k);
        for (uint r = 0u; r < valid; r++) {
            const float xv = x[(ulong)tokens[r] * args.in_dim + k];
            if (r < 4u) {
                gate0[r] = fma(gw, xv, gate0[r]);
                up0[r] = fma(uw, xv, up0[r]);
            } else {
                gate1[r - 4u] = fma(gw, xv, gate1[r - 4u]);
                up1[r - 4u] = fma(uw, xv, up1[r - 4u]);
            }
        }
    }
    for (uint r = 0u; r < valid; r++) {
        const float gate = simd_sum(r < 4u ? gate0[r] : gate1[r - 4u]);
        const float up = simd_sum(r < 4u ? up0[r] : up1[r - 4u]);
        if (lane == 0u)
            mid[(ulong)assignments[r] * args.expert_dim + output] =
                (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void kernel_qwen4_moe_q4_k_down_rows8(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const uint *counts [[buffer(3)]],
        device const uint *hids [[buffer(4)]],
        device const char *work [[buffer(5)]],
        device float *contributions [[buffer(6)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const uint *work_count = (device const uint *)work;
    if (tgpig.x >= work_count[0]) return;
    device const uint2 *work_items = (device const uint2 *)(work + 8);
    const uint2 item = work_items[tgpig.x];
    const uint expert = item.x;
    const uint local0 = item.y;
    const uint output = tgpig.y * (uint)nsg + (uint)sg;
    if (expert >= args.n_experts || output >= args.out_dim) return;
    const uint valid = min(8u, counts[expert] - local0);
    uint assignments[8];
    device const uint *expert_hids = hids + (ulong)expert * args.n_rows;
    for (uint r = 0u; r < valid; r++) assignments[r] = expert_hids[local0 + r];

    const uint row_blocks = args.down_dim / 256u;
    const ulong weight_row = (ulong)expert * args.out_dim + output;
    device const block_q4_K *wr = down_weight + weight_row * row_blocks;
    float4 sum0 = 0.0f, sum1 = 0.0f;
    /* k=[640,768) is the Q4_K alignment tail. Its activation is defined as
     * zero, so omitting those FMAs is exact and keeps mid physically 640-wide. */
    for (uint k = (uint)lane; k < args.expert_dim; k += 32u) {
        const float w = ds4_glm_q4_K_value(wr, k);
        for (uint r = 0u; r < valid; r++) {
            const float xv = mid[(ulong)assignments[r] * args.expert_dim + k];
            if (r < 4u) sum0[r] = fma(w, xv, sum0[r]);
            else sum1[r - 4u] = fma(w, xv, sum1[r - 4u]);
        }
    }
    for (uint r = 0u; r < valid; r++) {
        const float sum = simd_sum(r < 4u ? sum0[r] : sum1[r - 4u]);
        if (lane == 0u)
            contributions[(ulong)assignments[r] * args.out_dim + output] = sum;
    }
}

/* Keep the top-10 reduction token-major and strictly slot ordered.  The
 * rows8 kernels only change how expert contributions are produced; this
 * final accumulation must match decode and the scalar reference exactly. */
kernel void kernel_qwen4_moe_weighted_sum10_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const int *selected [[buffer(1)]],
        device const float *selected_weights [[buffer(2)]],
        device const float *contributions [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 tpig [[thread_position_in_grid]]) {
    const uint output = tpig.x;
    const uint row = tpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    float total = 0.0f;
    for (uint slot = 0u; slot < 10u; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        total = fma(contributions[assignment * args.out_dim + output],
                    selected_weights[assignment], total);
    }
    out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q4_k_gate_up(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *gate_weight [[buffer(1)]],
        device const block_q4_K *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const int *selected [[buffer(4)]],
        device float *mid [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    const uint slot = tgpig.z;
    if (output >= args.expert_dim || row >= args.n_rows || slot >= args.top_k) return;
    const ulong assignment = (ulong)row * args.top_k + slot;
    const int expert = selected[assignment];
    if (expert < 0 || (uint)expert >= args.n_experts) return;
    const uint row_blocks = args.in_dim / 256u;
    const ulong weight_row = (ulong)(uint)expert * args.expert_dim + output;
    const float gate = qwen4_q4_k_dot(
        gate_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    const float up = qwen4_q4_k_dot(
        up_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    if (lane == 0u)
        mid[assignment * args.expert_dim + output] =
            (gate / (1.0f + exp(-gate))) * up;
}

kernel void kernel_qwen4_moe_q4_k_down(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint row_blocks = args.down_dim / 256u;
    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
        const float sum = qwen4_q4_k_dot(
            down_weight + weight_row * row_blocks,
            mid + assignment * args.expert_dim,
            args.expert_dim, lane);
        total = fma(sum, selected_weights[assignment], total);
    }
    if (lane == 0u) out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q4_k_down_split(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const block_q4_K *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x;
    const uint row = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)row * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float sum = 0.0f;
        if (output < args.out_dim && row < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 256u;
            const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
            sum = qwen4_q4_k_dot(
                down_weight + weight_row * row_blocks,
                mid + assignment * args.expert_dim,
                args.expert_dim, lane);
        }
        if (lane == 0u) partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u && output < args.out_dim && row < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)row * args.top_k + slot;
            total = fma(partials[slot], selected_weights[assignment], total);
        }
        out[(ulong)row * args.out_dim + output] = total;
    }
}

kernel void kernel_qwen4_moe_q4_0_gate_up(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const qwen4_block_q4_0 *gate_weight [[buffer(1)]],
        device const qwen4_block_q4_0 *up_weight [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const int *selected [[buffer(4)]],
        device float *mid [[buffer(5)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    const uint slot = tgpig.z;
    if (output >= args.expert_dim || row >= args.n_rows || slot >= args.top_k)
        return;
    const ulong assignment = (ulong)row * args.top_k + slot;
    const int expert = selected[assignment];
    if (expert < 0 || (uint)expert >= args.n_experts) return;
    const uint row_blocks = args.in_dim / 32u;
    const ulong weight_row = (ulong)(uint)expert * args.expert_dim + output;
    const float2 values = qwen4_q4_0_pair_dot_f32(
        gate_weight + weight_row * row_blocks,
        up_weight + weight_row * row_blocks,
        x + (ulong)row * args.in_dim, args.in_dim, lane);
    if (lane == 0u)
        mid[assignment * args.expert_dim + output] =
            (values.x / (1.0f + exp(-values.x))) * values.y;
}

kernel void kernel_qwen4_moe_q4_0_down(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const qwen4_block_q4_0 *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint output = tgpig.x * (uint)nsg + (uint)sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint row_blocks = args.down_dim / 32u;
    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; slot++) {
        const ulong assignment = (ulong)row * args.top_k + slot;
        const int expert = selected[assignment];
        if (expert < 0 || (uint)expert >= args.n_experts) continue;
        const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
        const float sum = qwen4_q4_0_dot_f32(
            down_weight + weight_row * row_blocks,
            mid + assignment * args.expert_dim,
            args.expert_dim, lane);
        total = fma(sum, selected_weights[assignment], total);
    }
    if (lane == 0u) out[(ulong)row * args.out_dim + output] = total;
}

kernel void kernel_qwen4_moe_q4_0_down_split(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const qwen4_block_q4_0 *down_weight [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *selected_weights [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *partials [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x;
    const uint row = tgpig.y;
    if (sg < args.top_k) {
        const ulong assignment = (ulong)row * args.top_k + (uint)sg;
        const int expert = selected[assignment];
        float sum = 0.0f;
        if (output < args.out_dim && row < args.n_rows &&
            expert >= 0 && (uint)expert < args.n_experts) {
            const uint row_blocks = args.down_dim / 32u;
            const ulong weight_row = (ulong)(uint)expert * args.out_dim + output;
            sum = qwen4_q4_0_dot_f32(
                down_weight + weight_row * row_blocks,
                mid + assignment * args.expert_dim,
                args.expert_dim, lane);
        }
        if (lane == 0u) partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u && output < args.out_dim && row < args.n_rows) {
        float total = 0.0f;
        for (uint slot = 0u; slot < args.top_k; slot++) {
            const ulong assignment = (ulong)row * args.top_k + slot;
            total = fma(partials[slot], selected_weights[assignment], total);
        }
        out[(ulong)row * args.out_dim + output] = total;
    }
}

kernel void kernel_qwen4_shared_expert_add_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const float *routed [[buffer(1)]],
        device const float *shared [[buffer(2)]],
        device const float *raw_gate [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint row = gid.y;
    if (dim >= args.out_dim || row >= args.n_rows) return;
    const ulong at = (ulong)row * args.out_dim + dim;
    const float gate = 1.0f / (1.0f + exp(-raw_gate[row]));
    out[at] = fma(shared[at], gate, routed[at]);
}

kernel void kernel_qwen4_shared_expert_add_model_f32(
        constant qwen4_moe_args &args [[buffer(0)]],
        device const ushort *router_weight [[buffer(1)]],
        device const float *routed [[buffer(2)]],
        device const float *shared [[buffer(3)]],
        device const float *hidden [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *gate_value [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    // The unfused BF16 router assigns its only valid output to SIMD group 0.
    // Keep that exact 32-lane FMA and reduction order, then let the full
    // threadgroup apply the resulting scalar gate across the hidden row.
    if (sg == 0u) {
        device const float *xr = hidden + (ulong)row * args.out_dim;
        float sum = 0.0f;
        for (uint dim = lane; dim < args.out_dim; dim += 32u)
            sum = fma(qwen4_bf16_to_f32(router_weight[dim]), xr[dim], sum);
        sum = simd_sum(sum);
        if (lane == 0u) gate_value[0] = 1.0f / (1.0f + exp(-sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = gate_value[0];
    const ulong base = (ulong)row * args.out_dim;
    for (uint dim = tid; dim < args.out_dim; dim += 256u) {
        const ulong at = base + dim;
        out[at] = fma(shared[at], gate, routed[at]);
    }
}

struct qwen4_gdn_args {
    uint n_tokens;
    uint key_heads;
    uint value_heads;
    uint head_dim;
    uint has_mask;
    uint capture_slots;
};

struct qwen4_gdn_prepare_args {
    uint n_tokens;
    uint key_heads;
    uint value_heads;
    uint head_dim;
    uint conv_width;
    uint has_mask;
    uint capture_slots;
    float l2_eps;
};

// One thread owns one depthwise-convolution channel and walks the outer
// chunk in token order.  This preserves the exact causal state transition
// while keeping only four FP32 values live for the supported Qwen geometry.
template <bool Capture>
static inline void qwen4_gdn_conv_split(
        constant qwen4_gdn_prepare_args &args,
        device const float *mixed_qkv,
        device const ushort *conv_weight,
        device const uchar *mask,
        device float *conv_state,
        device float *q,
        device float *k,
        device float *v,
        device float *state_seq,
        uint channel) {
    const uint key_dim = args.key_heads * args.head_dim;
    const uint value_dim = args.value_heads * args.head_dim;
    const uint conv_dim = 2u * key_dim + value_dim;
    if (channel >= conv_dim || args.conv_width != 4u) return;

    float history[4];
    for (uint tap = 0; tap < 4u; tap++)
        history[tap] = conv_state[(ulong)channel * 4u + tap];

    for (uint token = 0; token < args.n_tokens; token++) {
        history[0] = history[1];
        history[1] = history[2];
        history[2] = history[3];
        const bool active = args.has_mask == 0u || mask[token] != 0u;
        history[3] = active
            ? mixed_qkv[(ulong)token * conv_dim + channel]
            : 0.0f;
        if (Capture && token < args.capture_slots) {
            const ulong capture_base =
                ((ulong)token * conv_dim + channel) * 4u;
            for (uint tap = 0; tap < 4u; tap++)
                state_seq[capture_base + tap] = history[tap];
        }
        float sum = 0.0f;
        for (uint tap = 0; tap < 4u; tap++)
            sum = fma(history[tap],
                      qwen4_bf16_to_f32(
                          conv_weight[(ulong)channel * 4u + tap]),
                      sum);
        const float activated = sum / (1.0f + exp(-sum));
        if (channel < key_dim) {
            q[(ulong)token * key_dim + channel] = activated;
        } else if (channel < 2u * key_dim) {
            k[(ulong)token * key_dim + channel - key_dim] = activated;
        } else {
            v[(ulong)token * value_dim + channel - 2u * key_dim] =
                activated;
        }
    }
    for (uint tap = 0; tap < 4u; tap++)
        conv_state[(ulong)channel * 4u + tap] = history[tap];
}

kernel void kernel_qwen4_gdn_conv_split_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *mixed_qkv [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device const uchar *mask [[buffer(3)]],
        device float *conv_state [[buffer(4)]],
        device float *q [[buffer(5)]],
        device float *k [[buffer(6)]],
        device float *v [[buffer(7)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_gdn_conv_split<false>(
        args, mixed_qkv, conv_weight, mask, conv_state,
        q, k, v, conv_state, channel);
}

kernel void kernel_qwen4_gdn_conv_split_capture_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *mixed_qkv [[buffer(1)]],
        device const ushort *conv_weight [[buffer(2)]],
        device const uchar *mask [[buffer(3)]],
        device float *conv_state [[buffer(4)]],
        device float *q [[buffer(5)]],
        device float *k [[buffer(6)]],
        device float *v [[buffer(7)]],
        device float *state_seq [[buffer(8)]],
        uint channel [[thread_position_in_grid]]) {
    qwen4_gdn_conv_split<true>(
        args, mixed_qkv, conv_weight, mask, conv_state,
        q, k, v, state_seq, channel);
}

kernel void kernel_qwen4_gdn_qk_l2norm_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device float *q [[buffer(1)]],
        device float *k [[buffer(2)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.key_heads) return;
    const ulong base = ((ulong)token * args.key_heads + head) * args.head_dim;
    float qsum = 0.0f;
    float ksum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float qv = q[base + dim];
        const float kv = k[base + dim];
        qsum = fma(qv, qv, qsum);
        ksum = fma(kv, kv, ksum);
    }
    qsum = simd_sum(qsum);
    ksum = simd_sum(ksum);
    if (lane == 0u) {
        partial[2u * sg] = qsum;
        partial[2u * sg + 1u] = ksum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        qsum = lane < nsg ? partial[2u * lane] : 0.0f;
        ksum = lane < nsg ? partial[2u * lane + 1u] : 0.0f;
        qsum = simd_sum(qsum);
        ksum = simd_sum(ksum);
        if (lane == 0u) {
            partial[0] = rsqrt(qsum + args.l2_eps) *
                         rsqrt((float)args.head_dim);
            partial[1] = rsqrt(ksum + args.l2_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        q[base + dim] *= partial[0];
        k[base + dim] *= partial[1];
    }
}

kernel void kernel_qwen4_gdn_gates_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *raw_decay [[buffer(1)]],
        device const float *raw_beta [[buffer(2)]],
        device const ushort *a_log [[buffer(3)]],
        device const ushort *dt_bias [[buffer(4)]],
        device const uchar *mask [[buffer(5)]],
        device float *decay [[buffer(6)]],
        device float *beta [[buffer(7)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint head = gid.x;
    const uint token = gid.y;
    if (head >= args.value_heads || token >= args.n_tokens) return;
    const ulong at = (ulong)token * args.value_heads + head;
    const bool active = args.has_mask == 0u || mask[token] != 0u;
    const float a = active ? raw_decay[at] : 0.0f;
    const float b = active ? raw_beta[at] : 0.0f;
    const float shifted = a + qwen4_bf16_to_f32(dt_bias[head]);
    const float softplus = shifted > 20.0f
        ? shifted
        : log(1.0f + exp(shifted));
    const float rate = exp(qwen4_bf16_to_f32(a_log[head]));
    decay[at] = exp(-rate * softplus);
    beta[at] = 1.0f / (1.0f + exp(-b));
}

struct qwen4_gdn_output_args {
    uint rows;
    uint head_dim;
    float eps;
};

kernel void kernel_qwen4_gdn_output_norm_f32(
        constant qwen4_gdn_output_args &args [[buffer(0)]],
        device const float *core [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *weight [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = core[base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = partial[0];
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float gate = 1.0f /
            (1.0f + exp(-raw_gate[base + dim]));
        out[base + dim] = core[base + dim] * inv_rms *
                          qwen4_bf16_to_f32(weight[dim]) * gate;
    }
}

static inline float qwen4_gdn_state_load(
        device const float *state,
        ulong index) {
    return state[index];
}

static inline float qwen4_gdn_state_load(
        device const ushort *state,
        ulong index) {
    return qwen4_bf16_to_f32(state[index]);
}

static inline void qwen4_gdn_state_store(
        device float *state,
        ulong index,
        float value) {
    state[index] = value;
}

static inline void qwen4_gdn_state_store(
        device ushort *state,
        ulong index,
        float value) {
    state[index] = qwen4_f32_to_bf16(value);
}

template <typename StateT, uint R, bool Capture>
static inline void qwen4_gdn_rows(
        constant qwen4_gdn_args &args,
        device const float      *q,
        device const float      *k,
        device const float      *v,
        device const float      *decay,
        device const float      *beta,
        device const uchar      *mask,
        device StateT           *state,
        device float            *out,
        device StateT           *state_seq,
        uint3 gid,
        ushort lane) {
    const uint row0 = gid.y * R;
    const uint hv = gid.z;
    if (lane >= 32u || row0 >= args.head_dim || hv >= args.value_heads) return;
    const uint hk = hv / (args.value_heads / args.key_heads);
    constexpr uint per_lane = 4u; // supported geometry has Dk=128.
    float local[R][per_lane];
    for (uint r = 0; r < R; r++) {
        for (uint i = 0; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index = ((ulong)hv * args.head_dim + row0 + r) *
                                args.head_dim + dk;
            local[r][i] = qwen4_gdn_state_load(state, index);
        }
    }
    for (uint t = 0; t < args.n_tokens; t++) {
        const bool active = args.has_mask == 0u || mask[t] != 0u;
        if (active) {
            const float d = decay[(ulong)t * args.value_heads + hv];
            const float b = beta[(ulong)t * args.value_heads + hv];
            for (uint r = 0; r < R; r++) {
                float kv = 0.0f;
                for (uint i = 0; i < per_lane; i++) {
                    const uint dk = lane * per_lane + i;
                    const float kk = k[((ulong)t * args.key_heads + hk) *
                                       args.head_dim + dk];
                    local[r][i] *= d;
                    kv = fma(local[r][i], kk, kv);
                }
                kv = simd_sum(kv);
                const float delta =
                    (v[((ulong)t * args.value_heads + hv) * args.head_dim +
                       row0 + r] - kv) * b;
                float y = 0.0f;
                for (uint i = 0; i < per_lane; i++) {
                    const uint dk = lane * per_lane + i;
                    const ulong qk_index =
                        ((ulong)t * args.key_heads + hk) * args.head_dim + dk;
                    local[r][i] = fma(k[qk_index], delta, local[r][i]);
                    y = fma(local[r][i], q[qk_index], y);
                }
                y = simd_sum(y);
                if (lane == 0u)
                    out[((ulong)t * args.value_heads + hv) * args.head_dim +
                        row0 + r] = y;
            }
        } else if (lane == 0u) {
            for (uint r = 0; r < R; r++)
                out[((ulong)t * args.value_heads + hv) * args.head_dim +
                    row0 + r] = 0.0f;
        }
        if (Capture && t < args.capture_slots) {
            const ulong state_elements =
                (ulong)args.value_heads * args.head_dim * args.head_dim;
            for (uint r = 0; r < R; r++) {
                for (uint i = 0; i < per_lane; i++) {
                    const uint dk = lane * per_lane + i;
                    const ulong index =
                        ((ulong)hv * args.head_dim + row0 + r) *
                        args.head_dim + dk;
                    qwen4_gdn_state_store(
                        state_seq, (ulong)t * state_elements + index,
                        local[r][i]);
                }
            }
        }
    }
    for (uint r = 0; r < R; r++) {
        for (uint i = 0; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index = ((ulong)hv * args.head_dim + row0 + r) *
                                args.head_dim + dk;
            qwen4_gdn_state_store(state, index, local[r][i]);
        }
    }
}

#define QWEN4_GDN_KERNEL(NAME, R)                                               \
kernel void NAME(                                                              \
        constant qwen4_gdn_args &args [[buffer(0)]],                           \
        device const float *q [[buffer(1)]],                                   \
        device const float *k [[buffer(2)]],                                   \
        device const float *v [[buffer(3)]],                                   \
        device const float *decay [[buffer(4)]],                               \
        device const float *beta [[buffer(5)]],                                \
        device const uchar *mask [[buffer(6)]],                                \
        device float *state [[buffer(7)]],                                     \
        device float *out [[buffer(8)]],                                       \
        uint3 gid [[thread_position_in_grid]],                                 \
        ushort lane [[thread_index_in_simdgroup]]) {                           \
    qwen4_gdn_rows<float, R, false>(                                            \
        args, q, k, v, decay, beta, mask, state, out, state, gid, lane);        \
}

#define QWEN4_GDN_CAPTURE_KERNEL(NAME, R)                                       \
kernel void NAME(                                                              \
        constant qwen4_gdn_args &args [[buffer(0)]],                           \
        device const float *q [[buffer(1)]],                                   \
        device const float *k [[buffer(2)]],                                   \
        device const float *v [[buffer(3)]],                                   \
        device const float *decay [[buffer(4)]],                               \
        device const float *beta [[buffer(5)]],                                \
        device const uchar *mask [[buffer(6)]],                                \
        device float *state [[buffer(7)]],                                     \
        device float *out [[buffer(8)]],                                       \
        device float *state_seq [[buffer(9)]],                                 \
        uint3 gid [[thread_position_in_grid]],                                 \
        ushort lane [[thread_index_in_simdgroup]]) {                           \
    qwen4_gdn_rows<float, R, true>(                                             \
        args, q, k, v, decay, beta, mask, state, out, state_seq, gid, lane);    \
}

QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r4_f32, 4)
QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r2_f32, 2)
QWEN4_GDN_KERNEL(kernel_qwen4_gdn_r1_f32, 1)
QWEN4_GDN_CAPTURE_KERNEL(kernel_qwen4_gdn_r4_capture_f32, 4)
QWEN4_GDN_CAPTURE_KERNEL(kernel_qwen4_gdn_r2_capture_f32, 2)
QWEN4_GDN_CAPTURE_KERNEL(kernel_qwen4_gdn_r1_capture_f32, 1)

#define QWEN4_GDN_BF16_KERNEL(NAME, R)                                          \
kernel void NAME(                                                              \
        constant qwen4_gdn_args &args [[buffer(0)]],                           \
        device const float *q [[buffer(1)]],                                   \
        device const float *k [[buffer(2)]],                                   \
        device const float *v [[buffer(3)]],                                   \
        device const float *decay [[buffer(4)]],                               \
        device const float *beta [[buffer(5)]],                                \
        device const uchar *mask [[buffer(6)]],                                \
        device ushort *state [[buffer(7)]],                                    \
        device float *out [[buffer(8)]],                                       \
        uint3 gid [[thread_position_in_grid]],                                 \
        ushort lane [[thread_index_in_simdgroup]]) {                           \
    qwen4_gdn_rows<ushort, R, false>(                                           \
        args, q, k, v, decay, beta, mask, state, out, state, gid, lane);        \
}

QWEN4_GDN_BF16_KERNEL(kernel_qwen4_gdn_r4_bf16_f32, 4)

#define QWEN4_GDN_BF16_CAPTURE_KERNEL(NAME, R)                                  \
kernel void NAME(                                                              \
        constant qwen4_gdn_args &args [[buffer(0)]],                           \
        device const float *q [[buffer(1)]],                                   \
        device const float *k [[buffer(2)]],                                   \
        device const float *v [[buffer(3)]],                                   \
        device const float *decay [[buffer(4)]],                               \
        device const float *beta [[buffer(5)]],                                \
        device const uchar *mask [[buffer(6)]],                                \
        device ushort *state [[buffer(7)]],                                    \
        device float *out [[buffer(8)]],                                       \
        device ushort *state_seq [[buffer(9)]],                                \
        uint3 gid [[thread_position_in_grid]],                                 \
        ushort lane [[thread_index_in_simdgroup]]) {                           \
    qwen4_gdn_rows<ushort, R, true>(                                            \
        args, q, k, v, decay, beta, mask, state, out, state_seq, gid, lane);    \
}

QWEN4_GDN_BF16_CAPTURE_KERNEL(kernel_qwen4_gdn_r4_bf16_capture_f32, 4)

/* Decode-only fusion of Q/K normalization and gate transforms into the BF16
 * recurrent update. The preceding convolution still writes raw activated
 * Q/K/V, but this removes two dispatches and avoids materializing decay/beta.
 * The two-level norm reduction mirrors kernel_qwen4_gdn_qk_l2norm_f32. */
kernel void kernel_qwen4_gdn_r4_bf16_raw_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device const float *raw_decay [[buffer(4)]],
        device const float *raw_beta [[buffer(5)]],
        device const ushort *a_log [[buffer(6)]],
        device const ushort *dt_bias [[buffer(7)]],
        device ushort *state [[buffer(8)]],
        device float *out [[buffer(9)]],
        uint3 gid [[thread_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint R = 4u;
    constexpr uint per_lane = 4u;
    const uint row0 = gid.y * R;
    const uint hv = gid.z;
    if (args.n_tokens != 1u || args.head_dim != 128u || lane >= 32u ||
        row0 >= args.head_dim || hv >= args.value_heads) return;
    const uint hk = hv / (args.value_heads / args.key_heads);
    const ulong qk_base = (ulong)hk * args.head_dim;

    float qpart[per_lane];
    float kpart[per_lane];
    for (uint i = 0u; i < per_lane; i++) {
        const uint dk = i * 32u + lane;
        const float qv = q[qk_base + dk];
        const float kv = k[qk_base + dk];
        qpart[i] = simd_sum(qv * qv);
        kpart[i] = simd_sum(kv * kv);
    }
    float qsum = lane < per_lane ? qpart[lane] : 0.0f;
    float ksum = lane < per_lane ? kpart[lane] : 0.0f;
    qsum = simd_sum(qsum);
    ksum = simd_sum(ksum);
    const float qscale = rsqrt(qsum + args.l2_eps) *
                         rsqrt((float)args.head_dim);
    const float kscale = rsqrt(ksum + args.l2_eps);

    const float shifted = raw_decay[hv] +
        qwen4_bf16_to_f32(dt_bias[hv]);
    const float softplus = shifted > 20.0f
        ? shifted : log(1.0f + exp(shifted));
    const float rate = exp(qwen4_bf16_to_f32(a_log[hv]));
    const float d = exp(-rate * softplus);
    const float b = 1.0f / (1.0f + exp(-raw_beta[hv]));

    float local[R][per_lane];
    for (uint r = 0u; r < R; r++) {
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index =
                ((ulong)hv * args.head_dim + row0 + r) *
                args.head_dim + dk;
            local[r][i] = qwen4_gdn_state_load(state, index) * d;
        }
    }
    for (uint r = 0u; r < R; r++) {
        float kv = 0.0f;
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            kv = fma(local[r][i], k[qk_base + dk] * kscale, kv);
        }
        kv = simd_sum(kv);
        const float delta =
            (v[(ulong)hv * args.head_dim + row0 + r] - kv) * b;
        float y = 0.0f;
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index =
                ((ulong)hv * args.head_dim + row0 + r) *
                args.head_dim + dk;
            local[r][i] = fma(k[qk_base + dk] * kscale,
                              delta, local[r][i]);
            y = fma(local[r][i], q[qk_base + dk] * qscale, y);
            qwen4_gdn_state_store(state, index, local[r][i]);
        }
        y = simd_sum(y);
        if (lane == 0u)
            out[(ulong)hv * args.head_dim + row0 + r] = y;
    }
}

/* Experimental full decode fusion. One 512-thread group owns a complete
 * value head: sixteen SIMDgroups update eight recurrent rows each, then the
 * group performs the exact four-SIMDgroup RMS reduction used by the standalone
 * output kernel and applies the sigmoid gate in place. */
kernel void kernel_qwen4_gdn_r8_bf16_raw_output_f32(
        constant qwen4_gdn_prepare_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device const float *raw_decay [[buffer(4)]],
        device const float *raw_beta [[buffer(5)]],
        device const ushort *a_log [[buffer(6)]],
        device const ushort *dt_bias [[buffer(7)]],
        device ushort *state [[buffer(8)]],
        device const float *raw_gate [[buffer(9)]],
        device const ushort *norm_weight [[buffer(10)]],
        device float *out [[buffer(11)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint3 gid [[thread_position_in_grid]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint R = 8u;
    constexpr uint per_lane = 4u;
    const uint row0 = gid.y * R;
    const uint hv = tgid.z;
    if (args.n_tokens != 1u || args.head_dim != 128u || lane >= 32u ||
        row0 >= args.head_dim || hv >= args.value_heads) return;
    const uint hk = hv / (args.value_heads / args.key_heads);
    const ulong qk_base = (ulong)hk * args.head_dim;

    float qpart[per_lane];
    float kpart[per_lane];
    for (uint i = 0u; i < per_lane; i++) {
        const uint dk = i * 32u + lane;
        const float qv = q[qk_base + dk];
        const float kv = k[qk_base + dk];
        qpart[i] = simd_sum(qv * qv);
        kpart[i] = simd_sum(kv * kv);
    }
    float qsum = lane < per_lane ? qpart[lane] : 0.0f;
    float ksum = lane < per_lane ? kpart[lane] : 0.0f;
    qsum = simd_sum(qsum);
    ksum = simd_sum(ksum);
    const float qscale = rsqrt(qsum + args.l2_eps) *
                         rsqrt((float)args.head_dim);
    const float kscale = rsqrt(ksum + args.l2_eps);

    const float shifted = raw_decay[hv] +
        qwen4_bf16_to_f32(dt_bias[hv]);
    const float softplus = shifted > 20.0f
        ? shifted : log(1.0f + exp(shifted));
    const float rate = exp(qwen4_bf16_to_f32(a_log[hv]));
    const float d = exp(-rate * softplus);
    const float b = 1.0f / (1.0f + exp(-raw_beta[hv]));

    float local[R][per_lane];
    for (uint r = 0u; r < R; r++) {
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index =
                ((ulong)hv * args.head_dim + row0 + r) *
                args.head_dim + dk;
            local[r][i] = qwen4_gdn_state_load(state, index) * d;
        }
    }
    for (uint r = 0u; r < R; r++) {
        float kv = 0.0f;
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            kv = fma(local[r][i], k[qk_base + dk] * kscale, kv);
        }
        kv = simd_sum(kv);
        const float delta =
            (v[(ulong)hv * args.head_dim + row0 + r] - kv) * b;
        float y = 0.0f;
        for (uint i = 0u; i < per_lane; i++) {
            const uint dk = lane * per_lane + i;
            const ulong index =
                ((ulong)hv * args.head_dim + row0 + r) *
                args.head_dim + dk;
            local[r][i] = fma(k[qk_base + dk] * kscale,
                              delta, local[r][i]);
            y = fma(local[r][i], q[qk_base + dk] * qscale, y);
            qwen4_gdn_state_store(state, index, local[r][i]);
        }
        y = simd_sum(y);
        if (lane == 0u) shared[row0 + r] = y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float square = tid < args.head_dim
        ? shared[tid] * shared[tid] : 0.0f;
    square = simd_sum(square);
    if (lane == 0u && sg < 4u) shared[128u + sg] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float total = lane < 4u ? shared[128u + lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0u)
            shared[132u] = rsqrt(
                total / (float)args.head_dim + args.l2_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.head_dim) {
        const ulong index = (ulong)hv * args.head_dim + tid;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[index]));
        out[index] = shared[tid] * shared[132u] *
            qwen4_bf16_to_f32(norm_weight[tid]) * gate;
    }
}

struct qwen4_qsa_prepare_args {
    uint cache_pos;
    uint n_tokens;
    uint cache_cap;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint index_heads;
    uint index_head_dim;
    uint ratio;
    uint rope_dim;
    uint pool_block_start;
    uint has_mrope;
    uint mrope_len;
    float rope_theta;
    float rms_eps;
};

static inline float qwen4_partial_rope(
        float current,
        float paired,
        uint dim,
        uint rope_dim,
        uint position,
        float theta) {
    if (dim >= rope_dim) return current;
    const uint half_dim = rope_dim / 2u;
    const uint freq = dim < half_dim ? dim : dim - half_dim;
    const float inv_freq = pow(theta,
        -2.0f * (float)freq / (float)rope_dim);
    const float angle = (float)position * inv_freq;
    const float c = cos(angle);
    const float s = sin(angle);
    return dim < half_dim ? fma(-paired, s, current * c)
                          : fma(paired, s, current * c);
}

static inline uint qwen4_rope_position(
        constant qwen4_qsa_prepare_args &args,
        device const int *mrope_positions,
        uint absolute,
        uint dim) {
    if (args.has_mrope == 0u) return absolute;
    const uint frequency = dim < args.rope_dim / 2u
        ? dim : dim - args.rope_dim / 2u;
    const uint axis = frequency % 3u;
    return (uint)mrope_positions[(ulong)axis * args.mrope_len + absolute];
}

kernel void kernel_qwen4_qsa_main_q_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *q_gate_raw [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device float *q [[buffer(3)]],
        device float *gate [[buffer(4)]],
        device const int *mrope_positions [[buffer(5)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.query_heads) return;
    const ulong raw_base =
        (ulong)token * args.query_heads * args.head_dim * 2u +
        (ulong)head * args.head_dim * 2u;
    const ulong out_base =
        ((ulong)token * args.query_heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = q_gate_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint absolute = args.cache_pos + token;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        float value = q_gate_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = q_gate_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        q[out_base + dim] = qwen4_partial_rope(
            value, paired, dim, args.rope_dim,
            qwen4_rope_position(args, mrope_positions, absolute, dim),
            args.rope_theta);
        gate[out_base + dim] = q_gate_raw[raw_base + args.head_dim + dim];
    }
}

kernel void kernel_qwen4_qsa_main_kv_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *key_raw [[buffer(1)]],
        device const float *value_raw [[buffer(2)]],
        device const ushort *norm_weight [[buffer(3)]],
        device ushort *key_cache [[buffer(4)]],
        device ushort *value_cache [[buffer(5)]],
        device const int *mrope_positions [[buffer(6)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.kv_heads) return;
    const ulong raw_base =
        ((ulong)token * args.kv_heads + head) * args.head_dim;
    const ulong cache_base =
        ((ulong)(args.cache_pos + token) * args.kv_heads + head) *
        args.head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        const float value = key_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(sum / (float)args.head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint absolute = args.cache_pos + token;
    for (uint dim = tid; dim < args.head_dim; dim += (uint)nsg * 32u) {
        float value = key_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = key_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        value = qwen4_partial_rope(
            value, paired, dim, args.rope_dim,
            qwen4_rope_position(args, mrope_positions, absolute, dim),
            args.rope_theta);
        key_cache[cache_base + dim] = qwen4_f32_to_bf16(value);
        value_cache[cache_base + dim] =
            qwen4_f32_to_bf16(value_raw[raw_base + dim]);
    }
}

kernel void kernel_qwen4_qsa_index_q_prepare_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *index_qk_raw [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device float *index_q [[buffer(3)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint token = group.x;
    const uint head = group.y;
    if (token >= args.n_tokens || head >= args.index_heads) return;
    const ulong raw_base =
        (ulong)token * (args.index_heads + 1u) * args.index_head_dim +
        (ulong)head * args.index_head_dim;
    const ulong out_base =
        ((ulong)token * args.index_heads + head) * args.index_head_dim;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        const float value = index_qk_raw[raw_base + dim];
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(
                sum / (float)args.index_head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint position = args.cache_pos + token;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        float value = index_qk_raw[raw_base + dim] * scale *
            qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = index_qk_raw[raw_base + pair] * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        index_q[out_base + dim] = qwen4_partial_rope(
            value, paired, dim, args.rope_dim, position, args.rope_theta);
    }
}

kernel void kernel_qwen4_qsa_index_key_store_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const float *index_qk_raw [[buffer(1)]],
        device ushort *raw_index_cache [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint dim = gid.x;
    const uint token = gid.y;
    if (dim >= args.index_head_dim || token >= args.n_tokens) return;
    const ulong raw_at =
        (ulong)token * (args.index_heads + 1u) * args.index_head_dim +
        (ulong)args.index_heads * args.index_head_dim + dim;
    raw_index_cache[(ulong)(args.cache_pos + token) *
                    args.index_head_dim + dim] =
        qwen4_f32_to_bf16(index_qk_raw[raw_at]);
}

static inline float qwen4_qsa_pooled_raw(
        device const ushort *raw_index_cache,
        uint block,
        uint ratio,
        uint head_dim,
        uint dim) {
    float sum = 0.0f;
    for (uint item = 0; item < ratio; item++)
        sum += qwen4_bf16_to_f32(
            raw_index_cache[((ulong)block * ratio + item) * head_dim + dim]);
    const float mean = sum / (float)ratio;
    return qwen4_bf16_to_f32(qwen4_f32_to_bf16(mean));
}

kernel void kernel_qwen4_qsa_index_pool_f32(
        constant qwen4_qsa_prepare_args &args [[buffer(0)]],
        device const ushort *raw_index_cache [[buffer(1)]],
        device const ushort *norm_weight [[buffer(2)]],
        device ushort *pooled_index_cache [[buffer(3)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint local_block [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = args.pool_block_start + local_block;
    float sum = 0.0f;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        const float value = qwen4_qsa_pooled_raw(
            raw_index_cache, block, args.ratio, args.index_head_dim, dim);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        sum = lane < nsg ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0u)
            partial[0] = rsqrt(
                sum / (float)args.index_head_dim + args.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = partial[0];
    const uint half_dim = args.rope_dim / 2u;
    const uint position = block * args.ratio;
    for (uint dim = tid; dim < args.index_head_dim;
         dim += (uint)nsg * 32u) {
        float value = qwen4_qsa_pooled_raw(
            raw_index_cache, block, args.ratio, args.index_head_dim, dim) *
            scale * qwen4_bf16_to_f32(norm_weight[dim]);
        float paired = 0.0f;
        if (dim < args.rope_dim) {
            const uint pair = dim < half_dim
                ? dim + half_dim : dim - half_dim;
            paired = qwen4_qsa_pooled_raw(
                raw_index_cache, block, args.ratio,
                args.index_head_dim, pair) * scale *
                qwen4_bf16_to_f32(norm_weight[pair]);
        }
        value = qwen4_partial_rope(
            value, paired, dim, args.rope_dim, position, args.rope_theta);
        pooled_index_cache[(ulong)block * args.index_head_dim + dim] =
            qwen4_f32_to_bf16(value);
    }
}

struct qwen4_qsa_attention_args {
    uint queries;
    uint cache_cap;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint top_k;
    uint ratio;
    uint max_selected;
    uint debug;   /* GQA MMA only: dump softmax stats to buffer(9) */
};

template <bool MEMOIZE_NUMERATORS>
static inline void qwen4_qsa_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        threadgroup float *scratch,
        uint2 group,
        uint tid,
        ushort lane,
        ushort sg,
        ushort nsg) {
    /* head_dim is pinned to 256 by the dispatch, so each simd lane owns one
     * contiguous eight-dim slice: 128-bit K/V loads and register value
     * accumulators replace the scratch score sheet and the serial per-token
     * output pass.  MEMOIZE_NUMERATORS no longer changes arithmetic (each
     * simdgroup consumes its own scores immediately); the template stays so
     * both pipeline names keep their bitwise-identical contract. */
    (void)tid;
    const uint query = group.x;
    const uint head = group.y;
    if (query >= args.queries || head >= args.query_heads) return;
    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count = min(
        min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;
    const ulong qbase =
        ((ulong)query * args.query_heads + head) * args.head_dim;
    const uint kv_head = head / (args.query_heads / args.kv_heads);
    if (selected == 0u) {
        for (uint dim = lane; dim < args.head_dim; dim += 32u)
            out[qbase + dim] = 0.0f;
        return;
    }

    const uint dim0 = (uint)lane * 8u;
    const device const float *qrow = q + qbase + dim0;
    const float4 q0 = *((device const float4 *)(qrow));
    const float4 q1 = *((device const float4 *)(qrow + 4u));

    float4 acc0 = 0.0f;
    float4 acc1 = 0.0f;
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint rank = sg; rank < selected; rank += (uint)nsg) {
        uint token;
        if (rank < block_count * args.ratio) {
            const uint block = selected_blocks[
                (ulong)query * args.top_k + rank / args.ratio];
            token = block * args.ratio + rank % args.ratio;
        } else {
            token = complete * args.ratio +
                    rank - block_count * args.ratio;
        }
        if (token >= visible) continue;
        const ulong kbase =
            ((ulong)token * args.kv_heads + kv_head) * args.head_dim;
        const uint4 kraw = *((device const uint4 *)(
            key_cache + kbase + dim0));
        thread const ushort *kq = (thread const ushort *)&kraw;
        float dot = q0.x * qwen4_bf16_to_f32(kq[0]) +
                    q0.y * qwen4_bf16_to_f32(kq[1]) +
                    q0.z * qwen4_bf16_to_f32(kq[2]) +
                    q0.w * qwen4_bf16_to_f32(kq[3]);
        dot += q1.x * qwen4_bf16_to_f32(kq[4]) +
               q1.y * qwen4_bf16_to_f32(kq[5]) +
               q1.z * qwen4_bf16_to_f32(kq[6]) +
               q1.w * qwen4_bf16_to_f32(kq[7]);
        dot = simd_sum(dot) * rsqrt((float)args.head_dim);

        /* The score is uniform across the simdgroup after simd_sum, so the
         * online softmax state and the rescale factor stay uniform too. */
        const float new_max = max(max_score, dot);
        const float factor = exp(max_score - new_max);
        const float p = exp(dot - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + p;
        acc0 *= factor;
        acc1 *= factor;

        const uint4 vraw = *((device const uint4 *)(
            value_cache + kbase + dim0));
        thread const ushort *vq = (thread const ushort *)&vraw;
        acc0 = float4(
            fma(p, qwen4_bf16_to_f32(vq[0]), acc0.x),
            fma(p, qwen4_bf16_to_f32(vq[1]), acc0.y),
            fma(p, qwen4_bf16_to_f32(vq[2]), acc0.z),
            fma(p, qwen4_bf16_to_f32(vq[3]), acc0.w));
        acc1 = float4(
            fma(p, qwen4_bf16_to_f32(vq[4]), acc1.x),
            fma(p, qwen4_bf16_to_f32(vq[5]), acc1.y),
            fma(p, qwen4_bf16_to_f32(vq[6]), acc1.z),
            fma(p, qwen4_bf16_to_f32(vq[7]), acc1.w));
    }

    /* Cross-simdgroup combine: partials in threadgroup memory, then every
     * lane rescales its own dim slice by exp(group_max - global_max). */
    threadgroup float *partials = scratch;
    threadgroup float *gmax = scratch + (uint)nsg * args.head_dim;
    threadgroup float *gsum = gmax + nsg;
    *((threadgroup float4 *)(partials +
        (ulong)sg * args.head_dim + dim0)) = acc0;
    *((threadgroup float4 *)(partials +
        (ulong)sg * args.head_dim + dim0 + 4u)) = acc1;
    if (lane == 0u) {
        gmax[sg] = max_score;
        gsum[sg] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = -INFINITY;
    for (uint g = 0u; g < (uint)nsg; g++)
        maximum = max(maximum, gmax[g]);
    if (!isfinite(maximum)) {
        for (uint dim = lane; dim < args.head_dim; dim += 32u)
            out[qbase + dim] = 0.0f;
        return;
    }
    float total = 0.0f;
    for (uint g = 0u; g < (uint)nsg; g++)
        total += gsum[g] * exp(gmax[g] - maximum);

    float4 r0 = 0.0f;
    float4 r1 = 0.0f;
    for (uint g = 0u; g < (uint)nsg; g++) {
        const float weight = gmax[g] == -INFINITY
            ? 0.0f : exp(gmax[g] - maximum);
        r0 += weight * *((threadgroup float4 *)(
            partials + (ulong)g * args.head_dim + dim0));
        r1 += weight * *((threadgroup float4 *)(
            partials + (ulong)g * args.head_dim + dim0 + 4u));
    }
    const float inv_sum = total > 0.0f ? 1.0f / total : 0.0f;
    const float4 inv0 = float4(inv_sum);
    const float4 inv1 = float4(inv_sum);
    float4 gated0 = r0 * inv0;
    float4 gated1 = r1 * inv1;
    gated0 *= float4(
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 0u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 1u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 2u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 3u])));
    gated1 *= float4(
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 4u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 5u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 6u])),
        1.0f / (1.0f + exp(-raw_gate[qbase + dim0 + 7u])));
    *((device float4 *)(out + qbase + dim0)) = gated0;
    *((device float4 *)(out + qbase + dim0 + 4u)) = gated1;
}

kernel void kernel_qwen4_qsa_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    qwen4_qsa_attention_bf16_f32<true>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, scratch, group, tid, lane, sg,
        nsg);
}

kernel void kernel_qwen4_qsa_attention_bf16_f32_legacy(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    qwen4_qsa_attention_bf16_f32<false>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, scratch, group, tid, lane, sg,
        nsg);
}

/* GQA tile-sharing matrix-core variant of the sparse QSA attention.
 *
 * Role-split organization (validated in speed-bench/gqa_mma_proto.metal):
 * one 192-thread threadgroup with six simdgroups owns one (query row,
 * KV head) pair.  simdgroups 0/1 are SCORE groups (eight head rows each:
 * full-width QK^T on the matrix units plus the online softmax, no O
 * accumulators); simdgroups 2..5 are PV groups owning eight rows x 128
 * dims each, so per-thread accumulator registers halve versus a
 * monolithic design and the matrix units can be fed at a far higher
 * rate.  Every gathered K/V tile is staged once in shared threadgroup
 * memory (dim-major for K so no operand load transposes), Q/K/V use F16
 * operands with F32 accumulation (the batched-kernel parity policy; Q
 * rounds F32->F16 and the BF16 cache converts losslessly to F16 outside
 * the subnormal tail), and the softmax, lazy running-max rescale, and
 * sigmoid gate stay F32 through threadgroup tiles.  The rescale tile
 * overlays the dead K/S staging regions.  The rank-to-token mapping,
 * masking, and selected-count semantics are identical to
 * kernel_qwen4_qsa_attention_bf16_f32; only the reduction order differs.
 */
template <bool QSPLIT>
static inline void qwen4_qsa_attention_gqa_mma_body(
        constant qwen4_qsa_attention_args &args,
        device const float *q,
        device const float *raw_gate,
        device const ushort *key_cache,
        device const ushort *value_cache,
        device const uint *selected_blocks,
        device const uint *selected_counts,
        device const uint *visible_tokens,
        device float *out,
        device float *dbg,
        threadgroup uchar *scratch,
        uint2 group,
        ushort lane,
        ushort sg) {
    /* Compiled geometry: 12 query heads per KV head as two padded 8-row
     * halves, head_dim 256, ratio-4 blocks, 64-token gather tiles with
     * 8-dim staging chunks.  The dispatch guards these. */
    constexpr uint GQA = 12u;
    constexpr uint D = 256u;
    constexpr uint BK = 64u;
    constexpr uint DC = 8u;
    constexpr uint DCHUNKS = D / DC;
    constexpr uint TK = BK / 8u;
    constexpr uint OD = (D / 2u) / 8u;   /* per PV simdgroup */
    constexpr uint TPT = 192u;

    threadgroup half *qs = (threadgroup half *)scratch;            /* 16*D */
    threadgroup half *qs_lo = qs + 16u * D;   /* 16*D, QSPLIT only */
    threadgroup half *kvs = qs + 16u * D * (QSPLIT ? 2u : 1u);     /* BK*DC */
    threadgroup float *s_tile =
        (threadgroup float *)(kvs + BK * DC);                      /* 16*BK */
    threadgroup half *p_tile = (threadgroup half *)(s_tile + 16u * BK);
    threadgroup int *sel = (threadgroup int *)(p_tile + 16u * BK);
    threadgroup float *stats = (threadgroup float *)(sel + BK);    /* 16*2 */
    threadgroup float *rfac = stats + 16u * 2u;                    /* 16   */
    threadgroup int *vote = (threadgroup int *)(rfac + 16u);
    /* One explicit 8x128 float O block (4 KB): each PV simdgroup passes
     * its accumulators through it in four barrier-separated phases for
     * the rare rescale and the final emit. */
    threadgroup float *oblock = (threadgroup float *)(vote + 2u);

    const uint query = group.x;
    const uint kv_head = group.y;
    if (query >= args.queries || kv_head >= args.kv_heads) return;
    const uint lane32 = (uint)lane;
    const uint tid = (uint)sg * 32u + lane32;
    const bool is_score = sg < 2u;
    const uint row_half = is_score ? (uint)sg : ((uint)sg - 2u) / 2u;
    const uint row_base = row_half * 8u;
    const uint dim_half = ((uint)sg - 2u) % 2u;   /* PV simdgroups only */

    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint complete = visible / args.ratio;
    const uint block_count =
        min(min(selected_counts[query], args.top_k), complete);
    const uint tail = visible - complete * args.ratio;
    const uint selected = block_count * args.ratio + tail;

    if (selected == 0u) {
        for (uint i = tid; i < GQA * D; i += TPT)
            out[((ulong)query * args.query_heads + kv_head * GQA + i / D) *
                args.head_dim + (i % D)] = 0.0f;
        return;
    }

    /* Stage the 12 query-head rows (+ zero padding) once as F16. */
    for (uint i = tid; i < 16u * (D / 8u); i += TPT) {
        const uint row = i / (D / 8u);
        const uint d8 = i % (D / 8u);
        float4 f0 = 0.0f;
        float4 f1 = 0.0f;
        if (row < GQA) {
            const device const float *qrow = q +
                ((ulong)query * args.query_heads + kv_head * GQA + row) *
                    args.head_dim +
                d8 * 8u;
            f0 = *((device const float4 *)qrow);
            f1 = *((device const float4 *)(qrow + 4u));
        }
        qs[row * D + d8 * 8u + 0u] = half(f0.x);
        qs[row * D + d8 * 8u + 1u] = half(f0.y);
        qs[row * D + d8 * 8u + 2u] = half(f0.z);
        qs[row * D + d8 * 8u + 3u] = half(f0.w);
        qs[row * D + d8 * 8u + 4u] = half(f1.x);
        qs[row * D + d8 * 8u + 5u] = half(f1.y);
        qs[row * D + d8 * 8u + 6u] = half(f1.z);
        qs[row * D + d8 * 8u + 7u] = half(f1.w);
        if (QSPLIT) {
            qs_lo[row * D + d8 * 8u + 0u] =
                half(f0.x - (float)half(f0.x));
            qs_lo[row * D + d8 * 8u + 1u] =
                half(f0.y - (float)half(f0.y));
            qs_lo[row * D + d8 * 8u + 2u] =
                half(f0.z - (float)half(f0.z));
            qs_lo[row * D + d8 * 8u + 3u] =
                half(f0.w - (float)half(f0.w));
            qs_lo[row * D + d8 * 8u + 4u] =
                half(f1.x - (float)half(f1.x));
            qs_lo[row * D + d8 * 8u + 5u] =
                half(f1.y - (float)half(f1.y));
            qs_lo[row * D + d8 * 8u + 6u] =
                half(f1.z - (float)half(f1.z));
            qs_lo[row * D + d8 * 8u + 7u] =
                half(f1.w - (float)half(f1.w));
        }
    }
    for (uint r = tid; r < 16u; r += TPT) {
        stats[r * 2u] = -INFINITY;
        stats[r * 2u + 1u] = 0.0f;
    }

    simdgroup_float8x8 ofrag[OD];
    if (!is_score)
        for (uint dd = 0u; dd < OD; dd++)
            ofrag[dd] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const float scale = rsqrt((float)args.head_dim);
    const uint n_tiles = (selected + BK - 1u) / BK;
    const ulong kv_base = (ulong)kv_head * args.head_dim;
    const ulong kv_token_stride = (ulong)args.kv_heads * args.head_dim;

    for (uint ktile = 0u; ktile < n_tiles; ktile++) {
        /* Resolve this tile's 128 ranks to token positions (-1 invalid).
         * The mapping matches kernel_qwen4_qsa_attention_bf16_f32. */
        for (uint k = tid; k < BK; k += TPT) {
            const uint rank = ktile * BK + k;
            uint token = UINT_MAX; /* sentinel for invalid */
            if (rank < block_count * args.ratio) {
                const uint block = selected_blocks[
                    (ulong)query * args.top_k + rank / args.ratio];
                token = block * args.ratio + rank % args.ratio;
            } else if (rank < selected) {
                token = complete * args.ratio +
                        rank - block_count * args.ratio;
            }
            sel[k] = (token == UINT_MAX || token >= visible)
                ? -1 : (int)token;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* QK^T: every simdgroup cooperates on staging the shared dim-
         * major K tile (no transpose loads); only the score simdgroups
         * run the matrix products. */
        {
            simdgroup_float8x8 sfrag[TK];
            if (is_score)
                for (uint ik = 0u; ik < TK; ik++)
                    sfrag[ik] =
                        make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint chunk = 0u; chunk < DCHUNKS; chunk++) {
                for (uint i = tid; i < BK * (DC / 8u); i += TPT) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            key_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[(d8 * 8u + e) * BK + k] =
                            half(qwen4_bf16_to_f32(elements[e]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint dd8 = 0u; dd8 < DC / 8u && is_score; dd8++) {
                    simdgroup_half8x8 qf;
                    simdgroup_load(qf, qs + row_base * D + chunk * DC +
                                       dd8 * 8u, D, 0, false);
                    simdgroup_half8x8 qfl;
                    if (QSPLIT)
                        simdgroup_load(qfl,
                                       qs_lo + row_base * D + chunk * DC +
                                           dd8 * 8u,
                                       D, 0, false);
                    for (uint ik = 0u; ik < TK; ik++) {
                        simdgroup_half8x8 kt;
                        simdgroup_load(kt,
                                       kvs + (dd8 * 8u) * BK + ik * 8u,
                                       BK, 0, false);
                        simdgroup_multiply_accumulate(
                            sfrag[ik], qf, kt, sfrag[ik]);
                        if (QSPLIT)
                            simdgroup_multiply_accumulate(
                                sfrag[ik], qfl, kt, sfrag[ik]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            /* Only the score simdgroups own populated sfrag fragments;
             * the PV groups share row_base row halves, so an unguarded
             * store would clobber the real scores with uninitialized
             * fragments (this exact bug shipped the kernel computing a
             * uniform softmax: it passed every small-magnitude fixture
             * because uniform weights approximate true weights there). */
            if (is_score)
                for (uint ik = 0u; ik < TK; ik++)
                    simdgroup_store(sfrag[ik],
                                    s_tile + row_base * BK + ik * 8u,
                                    BK, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Softmax on the score simdgroups.  Shuffle the UNMUTATED
         * per-lane partial: broadcasting from a lane whose accumulator
         * already absorbed an earlier iteration double-counts that
         * neighbor (max survives only by idempotence). */
        if (is_score) {
            const uint row = lane32 / 4u;
            /* Each of the row's four lanes covers BK/4 columns. */
            const uint col0 = (lane32 % 4u) * (BK / 4u);
            float local_max = -INFINITY;
            for (uint c = 0u; c < BK / 4u; c++) {
                float s = -INFINITY;
                if (sel[col0 + c] >= 0)
                    s = s_tile[(row_base + row) * BK + col0 + c] * scale;
                s_tile[(row_base + row) * BK + col0 + c] = s;
                local_max = max(local_max, s);
            }
            float row_max = local_max;
            for (uint j = 1u; j < 4u; j++)
                row_max = max(row_max,
                              simd_shuffle(local_max, row * 4u + j));
            const float old_max = stats[(row_base + row) * 2u];
            const float old_sum = stats[(row_base + row) * 2u + 1u];
            const float new_max = max(old_max, row_max);
            const bool grows = isfinite(old_max) && new_max > old_max;
            float local_sum = 0.0f;
            for (uint c = 0u; c < BK / 4u; c++) {
                const float s = s_tile[(row_base + row) * BK + col0 + c];
                const float p = isfinite(s) ? exp(s - new_max) : 0.0f;
                p_tile[(row_base + row) * BK + col0 + c] = half(p);
                local_sum += p;
            }
            float row_sum = local_sum;
            for (uint j = 1u; j < 4u; j++)
                row_sum += simd_shuffle(local_sum, row * 4u + j);
            if (lane32 % 4u == 0u) {
                const float factor = isfinite(old_max)
                    ? exp(old_max - new_max) : 1.0f;
                rfac[row_base + row] = factor;
                stats[(row_base + row) * 2u] = new_max;
                stats[(row_base + row) * 2u + 1u] =
                    old_sum * factor + row_sum;
            }
            if (lane32 == 0u) vote[sg] = grows ? 1 : 0;
            if (args.debug != 0u && query == 0u && kv_head == 0u &&
                sg == 0u && lane32 == 0u) {
                for (uint r = 0u; r < 8u; r++) {
                    dbg[(ulong)ktile * 2u * 8u + r * 2u] = stats[r * 2u];
                    dbg[(ulong)ktile * 2u * 8u + r * 2u + 1u] =
                        stats[r * 2u + 1u];
                }
                if (ktile == 0u) {
                    for (uint k = 0u; k < BK; k++)
                        dbg[2048u + k] = (float)sel[k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vote[0] != 0 || vote[1] != 0) {
            /* Lazy rescale: each PV simdgroup in turn brings its 8x128 O
             * block through the shared oblock; the other simdgroups wait
             * at the barriers. */
            for (uint phase = 0u; phase < 4u; phase++) {
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD; dd++)
                        simdgroup_store(ofrag[dd], oblock + dd * 8u,
                                        128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    const uint r = lane32 / 4u;
                    const float factor = rfac[row_base + r];
                    const uint col = (lane32 % 4u) * 32u;
                    for (uint c = 0u; c < 32u; c++)
                        oblock[r * 128u + col + c] *= factor;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (!is_score && (uint)sg == 2u + phase) {
                    for (uint dd = 0u; dd < OD; dd++)
                        simdgroup_load(ofrag[dd], oblock + dd * 8u,
                                       128u, 0, false);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* O += P @ V: every simdgroup cooperates on staging the shared V
         * tile; each PV simdgroup runs its dim half's products. */
        {
            for (uint chunk = 0u; chunk < DCHUNKS; chunk++) {
                const bool pv_chunk =
                    !is_score &&
                    chunk >= dim_half * (DCHUNKS / 2u) &&
                    chunk < (dim_half + 1u) * (DCHUNKS / 2u);
                for (uint i = tid; i < BK * (DC / 8u); i += TPT) {
                    const uint k = i / (DC / 8u);
                    const uint d8 = i % (DC / 8u);
                    uint4 raw = 0u;
                    if (sel[k] >= 0) {
                        raw = *((device const uint4 *)(
                            value_cache +
                            (ulong)sel[k] * kv_token_stride + kv_base +
                            chunk * DC + d8 * 8u));
                    }
                    thread const ushort *elements =
                        (thread const ushort *)&raw;
                    for (uint e = 0u; e < 8u; e++)
                        kvs[k * DC + d8 * 8u + e] =
                            half(qwen4_bf16_to_f32(elements[e]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (pv_chunk) {
                    for (uint ik = 0u; ik < TK; ik++) {
                        simdgroup_half8x8 pf;
                        simdgroup_load(pf,
                                       p_tile + row_base * BK + ik * 8u,
                                       BK, 0, false);
                        for (uint dd = 0u; dd < DC / 8u; dd++) {
                            simdgroup_half8x8 vf;
                            simdgroup_load(vf, kvs + ik * 8u * DC + dd * 8u,
                                           DC, 0, false);
                            simdgroup_multiply_accumulate(
                                ofrag[(chunk % (DCHUNKS / 2u)) *
                                          (DC / 8u) +
                                      dd],
                                pf, vf,
                                ofrag[(chunk % (DCHUNKS / 2u)) *
                                          (DC / 8u) +
                                      dd]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    /* Emit: each PV simdgroup normalizes and gates its 8x128 block.
     * Row half 1 owns only GQA-8 valid heads; its pad rows must not be
     * written (they alias the next KV group's heads and, at the last
     * group, run past the output buffer). */
    for (uint phase = 0u; phase < 4u; phase++) {
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA - 8u;
            for (uint dd = 0u; dd < OD; dd++)
                simdgroup_store(ofrag[dd], oblock + dd * 8u, 128u, 0,
                                false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!is_score && (uint)sg == 2u + phase) {
            const uint emit_rows = row_half == 0u ? 8u : GQA - 8u;
            for (uint i = lane32; i < emit_rows * (128u / 8u); i += 32u) {
                const uint r = i / (128u / 8u);
                const uint d8 = i % (128u / 8u);
                const uint head = kv_head * GQA + row_base + r;
                const ulong base =
                    ((ulong)query * args.query_heads + head) *
                    args.head_dim;
                const float m = stats[(row_base + r) * 2u];
                const float s = stats[(row_base + r) * 2u + 1u];
                const float inv =
                    isfinite(m) && s > 0.0f ? 1.0f / s : 0.0f;
                for (uint e = 0u; e < 8u; e++) {
                    const uint d = dim_half * 128u + d8 * 8u + e;
                    const float g =
                        1.0f / (1.0f + exp(-raw_gate[base + d]));
                    out[base + d] =
                        oblock[r * 128u + d8 * 8u + e] * inv * g;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* F16 hi/lo Q-split instantiation: the score simdgroups run a second QK
 * product per K chunk from the staged F16 rounding residual, restoring Q
 * to ~22 mantissa bits (K is already value-exact as BF16 -> F16), so the
 * scores match the scalar kernel's F32-Q numerics instead of drifting
 * with F16's 11-bit rounding at production score magnitudes.  Costs one
 * extra 16x256 F16 threadgroup tile (+8 KB) and doubles the score groups'
 * matrix-product count. */
kernel void kernel_qwen4_qsa_attention_gqa_mma_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        device float *dbg [[buffer(9)]],
        threadgroup uchar *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    qwen4_qsa_attention_gqa_mma_body<false>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, dbg, scratch, group, lane,
        sg);
}

kernel void kernel_qwen4_qsa_attention_gqa_mma_qsplit_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *selected_blocks [[buffer(5)]],
        device const uint *selected_counts [[buffer(6)]],
        device const uint *visible_tokens [[buffer(7)]],
        device float *out [[buffer(8)]],
        device float *dbg [[buffer(9)]],
        threadgroup uchar *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    qwen4_qsa_attention_gqa_mma_body<true>(
        args, q, raw_gate, key_cache, value_cache, selected_blocks,
        selected_counts, visible_tokens, out, dbg, scratch, group, lane,
        sg);
}



kernel void kernel_qwen4_qsa_dense_attention_bf16_f32(
        constant qwen4_qsa_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *raw_gate [[buffer(2)]],
        device const ushort *key_cache [[buffer(3)]],
        device const ushort *value_cache [[buffer(4)]],
        device const uint *visible_tokens [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    if (query >= args.queries || head >= args.query_heads) return;
    const uint visible = min(visible_tokens[query], args.cache_cap);
    const uint kv_head = head / (args.query_heads / args.kv_heads);
    const ulong qbase =
        ((ulong)query * args.query_heads + head) * args.head_dim;
    float qv[8];
    float acc[8];
    for (uint i = 0; i < 8u; i++) {
        qv[i] = q[qbase + (uint)lane + i * 32u];
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    const float scale = rsqrt((float)args.head_dim);
    for (uint token = 0; token < visible; token++) {
        const ulong base =
            ((ulong)token * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < 8u; i++)
            dot = fma(qv[i], qwen4_bf16_to_f32(
                key_cache[base + (uint)lane + i * 32u]), dot);
        dot = simd_sum(dot) * scale;
        const float next = max(maximum, dot);
        const float old_scale = maximum == -INFINITY
            ? 0.0f : exp(maximum - next);
        const float new_scale = exp(dot - next);
        denominator = denominator * old_scale + new_scale;
        for (uint i = 0; i < 8u; i++) {
            const uint d = (uint)lane + i * 32u;
            acc[i] = acc[i] * old_scale + new_scale *
                qwen4_bf16_to_f32(value_cache[base + d]);
        }
        maximum = next;
    }
    for (uint i = 0; i < 8u; i++) {
        const uint d = (uint)lane + i * 32u;
        const float value = visible == 0u ? 0.0f : acc[i] / denominator;
        out[qbase + d] = value /
            (1.0f + exp(-raw_gate[qbase + d]));
    }
}

struct qwen4_qsa_score_args {
    uint blocks;
    uint heads;
    uint head_dim;
    uint valid_blocks;
};

// Exact single-query QSA block scoring: one simdgroup owns one pooled key,
// retaining the same 32-lane FP32 reduction tree for every context length.
kernel void kernel_qwen4_qsa_score_m1_f32(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const float            *q [[buffer(1)]],
        device const float            *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma(q[(ulong)h * args.head_dim + d],
                            pooled_k[(ulong)block * args.head_dim + d],
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

// Low-precision cache variants retain FP32 products and the identical
// lane/reduction order used by the F32 reference specialization.
kernel void kernel_qwen4_qsa_score_m1_f16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const half             *q [[buffer(1)]],
        device const half             *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma((float)q[(ulong)h * args.head_dim + d],
                            (float)pooled_k[(ulong)block * args.head_dim + d],
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

kernel void kernel_qwen4_qsa_score_m1_bf16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const ushort           *q [[buffer(1)]],
        device const ushort           *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            dot_value = fma(qwen4_bf16_to_f32(
                                q[(ulong)h * args.head_dim + d]),
                            qwen4_bf16_to_f32(
                                pooled_k[(ulong)block * args.head_dim + d]),
                            dot_value);
        }
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

kernel void kernel_qwen4_qsa_score_m1_f32_bf16(
        constant qwen4_qsa_score_args &args [[buffer(0)]],
        device const float            *q [[buffer(1)]],
        device const ushort           *pooled_k [[buffer(2)]],
        device float                  *scores [[buffer(3)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint block = tgp * (uint)nsg + sg;
    if (block >= args.blocks) return;
    float score = 0.0f;
    for (uint h = 0; h < args.heads; h++) {
        float dot_value = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u)
            dot_value = fma(q[(ulong)h * args.head_dim + d],
                            qwen4_bf16_to_f32(
                                pooled_k[(ulong)block * args.head_dim + d]),
                            dot_value);
        dot_value = simd_sum(dot_value);
        if (lane == 0u) score += max(dot_value, 0.0f);
    }
    if (lane == 0u) {
        scores[block] = block < args.valid_blocks
            ? score * rsqrt((float)args.head_dim)
            : -INFINITY;
    }
}

struct qwen4_qsa_tile_args {
    uint queries;
    uint blocks;
    uint block_start;
    uint tile_blocks;
    uint heads;
    uint head_dim;
    uint top_k;
};

// Score one pooled-key tile.  The controller folds it into a persistent
// per-query heap immediately, bounding scratch to queries * tile_blocks.
// visible_blocks carries causal lengths as runtime data, replacing a full
// [queries, context] boolean mask.
kernel void kernel_qwen4_qsa_score_tile_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const float           *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint local_block = tgp.x * (uint)nsg + sg;
    const uint query = tgp.y;
    if (query >= args.queries || local_block >= args.tile_blocks) return;
    const uint block = args.block_start + local_block;
    float score = 0.0f;
    if (block < args.blocks && block < visible_blocks[query]) {
        for (uint h = 0; h < args.heads; ++h) {
            float dot = 0.0f;
            const ulong qbase = ((ulong)query * args.heads + h) * args.head_dim;
            const ulong kbase = (ulong)block * args.head_dim;
            for (uint d = lane; d < args.head_dim; d += 32u)
                dot = fma(q[qbase + d], pooled_k[kbase + d], dot);
            dot = simd_sum(dot);
            if (lane == 0u) score += max(dot, 0.0f);
        }
        if (lane == 0u) score *= rsqrt((float)args.head_dim);
    } else if (lane == 0u) {
        score = -INFINITY;
    }
    if (lane == 0u)
        tile_scores[(ulong)query * args.tile_blocks + local_block] = score;
}

kernel void kernel_qwen4_qsa_score_tile_bf16(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const ushort          *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint local_block = tgp.x * (uint)nsg + sg;
    const uint query = tgp.y;
    if (query >= args.queries || local_block >= args.tile_blocks) return;
    const uint block = args.block_start + local_block;
    float score = 0.0f;
    if (block < args.blocks && block < visible_blocks[query]) {
        for (uint h = 0; h < args.heads; ++h) {
            float dot = 0.0f;
            const ulong qbase = ((ulong)query * args.heads + h) * args.head_dim;
            const ulong kbase = (ulong)block * args.head_dim;
            for (uint d = lane; d < args.head_dim; d += 32u)
                dot = fma(q[qbase + d],
                          qwen4_bf16_to_f32(pooled_k[kbase + d]), dot);
            dot = simd_sum(dot);
            if (lane == 0u) score += max(dot, 0.0f);
        }
        if (lane == 0u) score *= rsqrt((float)args.head_dim);
    } else if (lane == 0u) {
        score = -INFINITY;
    }
    if (lane == 0u)
        tile_scores[(ulong)query * args.tile_blocks + local_block] = score;
}

/* Long-context restaging of the BF16 tile scorer.  The original grid gives
 * every four-block threadgroup its own full re-read of the query's index
 * vectors (2 KB of q per 1 KB of pooled keys), which at tens of thousands
 * of visible blocks makes q re-reads the dominant L2 traffic.  This
 * variant loads the query vectors into registers once per threadgroup and
 * scans eight blocks per simdgroup, keeping each (query, block) score's
 * lane arithmetic — fma chain order, simdgroup reduction, head max/sum,
 * rsqrt scaling, causal masking, and output placement — bit-identical to
 * kernel_qwen4_qsa_score_tile_bf16, so the selection and every downstream
 * consumer are unchanged. */
kernel void kernel_qwen4_qsa_score_tile_batch_bf16(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const ushort          *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr uint BLOCKS_PER_SIMDGROUP = 8u;
    const uint group_blocks = (uint)nsg * BLOCKS_PER_SIMDGROUP;
    const uint local_base = tgp.x * group_blocks;
    const uint query = tgp.y;
    if (query >= args.queries) return;
    const uint visible = visible_blocks[query];

    /* Per lane: q[head][lane + j*32] for j = 0..3 — exactly the lanes'
    * strided slice of the original d-loop. */
    float qv[4][4];
    for (uint h = 0; h < 4u; ++h) {
        const ulong qbase = ((ulong)query * args.heads + h) * args.head_dim;
        for (uint j = 0; j < 4u; ++j)
            qv[h][j] = q[qbase + j * 32u + lane];
    }

    for (uint b = 0; b < BLOCKS_PER_SIMDGROUP; ++b) {
        const uint local_block = local_base + (uint)sg *
            BLOCKS_PER_SIMDGROUP + b;
        float score = 0.0f;
        if (local_block < args.tile_blocks) {
            const uint block = args.block_start + local_block;
            if (block < args.blocks && block < visible) {
                const ulong kbase = (ulong)block * args.head_dim;
                for (uint h = 0; h < 4u; ++h) {
                    float dot = 0.0f;
                    dot = fma(qv[h][0], qwen4_bf16_to_f32(
                                  pooled_k[kbase + lane]), dot);
                    dot = fma(qv[h][1], qwen4_bf16_to_f32(
                                  pooled_k[kbase + 32u + lane]), dot);
                    dot = fma(qv[h][2], qwen4_bf16_to_f32(
                                  pooled_k[kbase + 64u + lane]), dot);
                    dot = fma(qv[h][3], qwen4_bf16_to_f32(
                                  pooled_k[kbase + 96u + lane]), dot);
                    dot = simd_sum(dot);
                    if (lane == 0u) score += max(dot, 0.0f);
                }
                if (lane == 0u) score *= rsqrt((float)args.head_dim);
            } else if (lane == 0u) {
                score = -INFINITY;
            }
            if (lane == 0u)
                tile_scores[(ulong)query * args.tile_blocks + local_block] =
                    score;
        }
    }
}

/* Tensor-core restaging of the BF16 tile scorer (drift-gated, NOT
 * byte-exact): the four index heads of one query fold into the rows of a
 * single GEMM A[queries*4, 128] @ B^T[128, tile_blocks] where B is the
 * shared pooled-key row of each block.  F32 query vectors and the BF16
 * pooled keys are staged to F16 threadgroup tiles exactly once (the
 * kernel_mul_mm operand-rounding policy) and the products accumulate in
 * F32 on the simdgroup matrix units.  Each 128-thread threadgroup owns a
 * 64x64 output tile (16 queries x 64 blocks); A and B^T are staged one
 * 64-wide K-chunk at a time into 16 KB of threadgroup memory (occupancy
 * is the lever — 32 KB variants measured 1.6x slower), four simdgroups
 * each compute a 32x32 subtile through 16 accumulator fragments, and the
 * results pass through a C tile overlaid on the dead staging region
 * (fragment element ownership is never indexed directly) into a
 * cooperative epilogue that reduces every query's four head rows with
 * the production max/sum/rsqrt arithmetic and causal masking.  Score
 * values differ from the scalar kernels only by the F16 operand
 * rounding; the selection order can flip at rounding-scale score ties. */
kernel void kernel_qwen4_qsa_score_tile_mm_bf16(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *q [[buffer(1)]],
        device const ushort          *pooled_k [[buffer(2)]],
        device const uint            *visible_blocks [[buffer(3)]],
        device float                 *tile_scores [[buffer(4)]],
        threadgroup uchar            *scratch [[threadgroup(0)]],
        uint2 tgp [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = 64u;   /* 16 queries x 4 head rows */
    constexpr uint BN = 64u;
    constexpr uint K = 128u;
    constexpr uint BK = 64u;
    constexpr uint TPT = 128u;

    /* One K-chunk of A and B^T is staged at a time (8 KB + 8 KB); the C
     * tile overlays the dead staging region after the MMA loop. */
    threadgroup half *sa = (threadgroup half *)scratch;       /* BM x BK */
    threadgroup half *sb = sa + BM * BK;                      /* BK x BN, K-major */
    threadgroup float *ctile = (threadgroup float *)sa;

    const uint m0 = tgp.x * BM;
    const uint n0 = tgp.y * BN;
    const uint tid = (uint)sg * 32u + (uint)lane;
    const uint rows = args.queries * 4u;

    simdgroup_half8x8 ma[BM / 16u];
    simdgroup_half8x8 mb[BN / 16u];
    simdgroup_float8x8 mc[(BM / 16u) * (BN / 16u)];
    for (uint i = 0u; i < (BM / 16u) * (BN / 16u); i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint m_half = ((uint)sg & 1u) * (BM / 2u);
    const uint n_half = ((uint)sg >> 1) * (BN / 2u);
    for (uint chunk = 0u; chunk < K / BK; chunk++) {
        for (uint i = tid; i < BM * (BK / 4u); i += TPT) {
            const uint row = i / (BK / 4u);
            const uint c4 = i % (BK / 4u);
            float4 v = 0.0f;
            if (m0 + row < rows)
                v = *((device const float4 *)q +
                      (ulong)(m0 + row) * (K / 4u) + chunk * (BK / 4u) + c4);
            *(threadgroup half4 *)(sa + row * BK + c4 * 4u) = (half4)v;
        }
        for (uint i = tid; i < BN * (BK / 8u); i += TPT) {
            const uint n = i / (BK / 8u);
            const uint k8 = i % (BK / 8u);
            const uint block = args.block_start + n0 + n;
            uint4 raw = 0u;
            if (n0 + n < args.tile_blocks && block < args.blocks)
                raw = *((device const uint4 *)pooled_k +
                        (ulong)block * (K / 8u) +
                        chunk * (BK / 8u) + k8);
            thread const ushort *el = (thread const ushort *)&raw;
            for (uint e = 0u; e < 8u; e++)
                sb[(k8 * 8u + e) * BN + n] =
                    half(qwen4_bf16_to_f32(el[e]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0u; i < BM / 16u; i++)
                simdgroup_load(ma[i], sa + (m_half + i * 8u) * BK + kk,
                               BK, 0, false);
            for (uint j = 0u; j < BN / 16u; j++)
                simdgroup_load(mb[j], sb + kk * BN + n_half + j * 8u,
                               BN, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0u; i < BM / 16u; i++)
                for (uint j = 0u; j < BN / 16u; j++)
                    simdgroup_multiply_accumulate(mc[j * (BM / 16u) + i],
                                                  ma[i], mb[j],
                                                  mc[j * (BM / 16u) + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0u; j < BN / 16u; j++)
        for (uint i = 0u; i < BM / 16u; i++)
            simdgroup_store(mc[j * (BM / 16u) + i],
                            ctile + (m_half + i * 8u) * BN + n_half + j * 8u,
                            BN, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = rsqrt((float)args.head_dim);
    for (uint o = tid; o < (BM / 4u) * BN; o += TPT) {
        const uint q_local = o / BN;
        const uint n = o % BN;
        const uint query = tgp.x * (BM / 4u) + q_local;
        if (query >= args.queries || n0 + n >= args.tile_blocks) continue;
        const uint block = args.block_start + n0 + n;
        float score = 0.0f;
        for (uint h = 0u; h < 4u; h++)
            score += max(ctile[(q_local * 4u + h) * BN + n], 0.0f);
        score *= scale;
        tile_scores[(ulong)query * args.tile_blocks + n0 + n] =
            (block < args.blocks && block < visible_blocks[query])
                ? score
                : -INFINITY;
    }
}

static inline bool qwen4_qsa_worse(float as, uint ai, float bs, uint bi) {
    return as < bs || (as == bs && ai > bi);
}

static inline void qwen4_qsa_heap_down(
        device float *scores,
        device uint *indices,
        uint count,
        uint root) {
    while (true) {
        const uint left = root * 2u + 1u;
        if (left >= count) return;
        uint worse = left;
        const uint right = left + 1u;
        if (right < count &&
            qwen4_qsa_worse(scores[right], indices[right],
                            scores[left], indices[left])) worse = right;
        if (!qwen4_qsa_worse(scores[worse], indices[worse],
                             scores[root], indices[root])) return;
        const float ts = scores[root];
        const uint ti = indices[root];
        scores[root] = scores[worse];
        indices[root] = indices[worse];
        scores[worse] = ts;
        indices[worse] = ti;
        root = worse;
    }
}

kernel void kernel_qwen4_qsa_heap_init_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device float                 *heap_scores [[buffer(1)]],
        device uint                  *heap_indices [[buffer(2)]],
        device uint                  *heap_counts [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)args.queries * args.top_k;
    if ((ulong)gid < total) {
        heap_scores[gid] = -INFINITY;
        heap_indices[gid] = 0xffffffffu;
    }
    if (gid < args.queries) heap_counts[gid] = 0u;
}

// Dot products above are parallel; only the compact top-k structure is
// serialized per query.  A min-heap makes each candidate O(log(top_k)).
kernel void kernel_qwen4_qsa_heap_merge_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device const float           *tile_scores [[buffer(1)]],
        device float                 *heap_scores [[buffer(2)]],
        device uint                  *heap_indices [[buffer(3)]],
        device uint                  *heap_counts [[buffer(4)]],
        uint query [[thread_position_in_grid]]) {
    if (query >= args.queries) return;
    device float *hs = heap_scores + (ulong)query * args.top_k;
    device uint *hi = heap_indices + (ulong)query * args.top_k;
    uint count = heap_counts[query];
    for (uint i = 0; i < args.tile_blocks; ++i) {
        const float score = tile_scores[(ulong)query * args.tile_blocks + i];
        if (!isfinite(score)) continue;
        const uint index = args.block_start + i;
        if (count < args.top_k) {
            uint child = count++;
            hs[child] = score;
            hi[child] = index;
            while (child != 0u) {
                const uint parent = (child - 1u) / 2u;
                if (!qwen4_qsa_worse(hs[child], hi[child], hs[parent], hi[parent])) break;
                const float ts = hs[parent];
                const uint ti = hi[parent];
                hs[parent] = hs[child]; hi[parent] = hi[child];
                hs[child] = ts; hi[child] = ti;
                child = parent;
            }
        } else if (!qwen4_qsa_worse(score, index, hs[0], hi[0]) &&
                   !(score == hs[0] && index == hi[0])) {
            hs[0] = score;
            hi[0] = index;
            qwen4_qsa_heap_down(hs, hi, count, 0u);
        }
    }
    heap_counts[query] = count;
}

// In-place heap sort leaves score/index pairs in descending score order with
// ascending block index as the deterministic tie break.
kernel void kernel_qwen4_qsa_heap_sort_f32(
        constant qwen4_qsa_tile_args &args [[buffer(0)]],
        device float                 *heap_scores [[buffer(1)]],
        device uint                  *heap_indices [[buffer(2)]],
        device const uint            *heap_counts [[buffer(3)]],
        uint query [[thread_position_in_grid]]) {
    if (query >= args.queries) return;
    device float *hs = heap_scores + (ulong)query * args.top_k;
    device uint *hi = heap_indices + (ulong)query * args.top_k;
    const uint count = heap_counts[query];
    for (uint end = count; end > 1u; --end) {
        const float ts = hs[0];
        const uint ti = hi[0];
        hs[0] = hs[end - 1u]; hi[0] = hi[end - 1u];
        hs[end - 1u] = ts; hi[end - 1u] = ti;
        qwen4_qsa_heap_down(hs, hi, end - 1u, 0u);
    }
}

struct qwen4_qsa_bitonic_args {
    uint src_entries;      /* scores/indices readable in the source */
    uint keep;             /* entries each chunk forwards (== top_k) */
    uint final_pass;       /* write the ordered output and count instead */
    uint src_implicit_index; /* level 0 derives the index from position */
};

struct qwen4_qsa_bitonic_stream_args {
    uint tile_blocks;   /* blocks scored into this tile */
    uint block_start;   /* absolute index of tile block zero */
    uint keep;          /* running candidate width (== top_k) */
    uint first_tile;    /* candidate slots start empty */
};

/* Streaming companion of the bitonic M=1 top-k for multi-query batches (the
 * verifier rows and prefill microtiles).  Each query owns one threadgroup
 * that merges its running top-k candidates with the freshly scored tile and
 * re-sorts, replacing the one-thread-per-query heap sift over every tile.
 * Candidates stay ordered after every invocation, so the last pass has
 * already produced the ordered output; the comparator and tie-breaks match
 * the heap path exactly. */
kernel void kernel_qwen4_qsa_bitonic_stream_f32(
        constant qwen4_qsa_bitonic_stream_args &args [[buffer(0)]],
        device const float *tile_scores [[buffer(1)]],
        device float       *cand_scores [[buffer(2)]],
        device uint        *cand_indices [[buffer(3)]],
        device uint        *counts [[buffer(4)]],
        threadgroup float  *tscores [[threadgroup(0)]],
        threadgroup uint   *tindices [[threadgroup(1)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort nthreads [[threads_per_threadgroup]]) {
    constexpr uint CHUNK = 2048u;
    for (uint i = tid; i < CHUNK; i += (uint)nthreads) {
        float score;
        uint index;
        if (i < args.keep) {
            if (args.first_tile) {
                score = -INFINITY;
                index = 0xFFFFFFFFu;
            } else {
                score = cand_scores[(ulong)tgp * args.keep + i];
                index = cand_indices[(ulong)tgp * args.keep + i];
            }
        } else if (i < args.keep + args.tile_blocks) {
            const uint t = i - args.keep;
            score = tile_scores[(ulong)tgp * args.tile_blocks + t];
            index = args.block_start + t;
        } else {
            score = -INFINITY;
            index = 0xFFFFFFFFu;
        }
        tscores[i] = score;
        tindices[i] = index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2u; size <= CHUNK; size <<= 1u) {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = tid; k < CHUNK; k += (uint)nthreads) {
                const uint j = k ^ stride;
                if (j <= k) continue;
                const bool ascending = (k & size) == 0u;
                if (ascending
                        ? qwen4_qsa_worse(tscores[k], tindices[k],
                                          tscores[j], tindices[j])
                        : qwen4_qsa_worse(tscores[j], tindices[j],
                                          tscores[k], tindices[k])) {
                    const float score = tscores[k];
                    const uint index = tindices[k];
                    tscores[k] = tscores[j];
                    tindices[k] = tindices[j];
                    tscores[j] = score;
                    tindices[j] = index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.keep; i += (uint)nthreads) {
        cand_scores[(ulong)tgp * args.keep + i] = tscores[i];
        cand_indices[(ulong)tgp * args.keep + i] = tindices[i];
    }
    threadgroup uint finite_count[1];
    if (tid == 0u) finite_count[0] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.keep; i += (uint)nthreads) {
        if (isfinite(tscores[i]))
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint *)finite_count, 1u,
                memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) counts[tgp] = finite_count[0];
}

/* Threshold-filtered merge-select for the streaming top-k at the
 * production geometry (keep = 512, tile <= 1024).  The full-sort merge
 * above orders all 2048 slots on every tile; this kernel instead drops
 * every tile entry that cannot reach the final top-512 — anything worse
 * than the running 512th-best is beaten by all 512 candidates and is
 * provably out — and only then sorts, so late tiles sort a handful of
 * survivors instead of 2048 slots.  The filter is exact, block indices
 * are unique, and the comparator is qwen4_qsa_worse, so the ordered
 * output is bit-identical to kernel_qwen4_qsa_bitonic_stream_f32.
 * Tiles that arrive before the threshold exists (more survivors than
 * fit) fall back to the full 2048-wide sort in-kernel. */
kernel void kernel_qwen4_qsa_merge_select_f32(
        constant qwen4_qsa_bitonic_stream_args &args [[buffer(0)]],
        device const float *tile_scores [[buffer(1)]],
        device float       *cand_scores [[buffer(2)]],
        device uint        *cand_indices [[buffer(3)]],
        device uint        *counts [[buffer(4)]],
        threadgroup float  *tscores [[threadgroup(0)]],
        threadgroup uint   *tindices [[threadgroup(1)]],
        threadgroup uint   *counter [[threadgroup(2)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort nthreads [[threads_per_threadgroup]]) {
    constexpr uint CHUNK = 2048u;
    constexpr uint KEEP = 512u;
    const bool production = args.keep == KEEP &&
        args.tile_blocks <= 1024u && !args.first_tile;
    /* Load the legacy layout: sorted running candidates (empty on the
     * first tile), then this tile's entries, then -INF padding. */
    for (uint i = tid; i < CHUNK; i += (uint)nthreads) {
        float score;
        uint index;
        if (i < args.keep) {
            if (args.first_tile) {
                score = -INFINITY;
                index = 0xFFFFFFFFu;
            } else {
                score = cand_scores[(ulong)tgp * args.keep + i];
                index = cand_indices[(ulong)tgp * args.keep + i];
            }
        } else if (i < args.keep + args.tile_blocks) {
            const uint t = i - args.keep;
            score = tile_scores[(ulong)tgp * args.tile_blocks + t];
            index = args.block_start + t;
        } else {
            score = -INFINITY;
            index = 0xFFFFFFFFu;
        }
        tscores[i] = score;
        tindices[i] = index;
    }
    if (tid == 0u) counter[0] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint survivors = CHUNK + 1u;  /* force the fallback unless proven fast */
    if (production) {
        /* Survivor region [KEEP, 1024) starts empty; passing tile entries
         * compact into it in arbitrary order (the survivor sort below is
         * independent of input order because every key is unique). */
        for (uint i = tid; i < 512u; i += (uint)nthreads) {
            tscores[KEEP + i] = -INFINITY;
            tindices[KEEP + i] = 0xFFFFFFFFu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float thr_s = tscores[KEEP - 1u];
        const uint thr_i = tindices[KEEP - 1u];
        for (uint t = tid; t < args.tile_blocks; t += (uint)nthreads) {
            const float s =
                tile_scores[(ulong)tgp * args.tile_blocks + t];
            const uint idx = args.block_start + t;
            if (!(s < thr_s || (s == thr_s && idx > thr_i))) {
                const uint slot = KEEP + atomic_fetch_add_explicit(
                    (threadgroup atomic_uint *)counter, 1u,
                    memory_order_relaxed);
                tscores[slot] = s;
                tindices[slot] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        survivors = counter[0];
    }

    if (survivors <= 512u) {
        /* Sort only the survivor span [KEEP, KEEP + W).  Padding beyond W
         * stays -INF at the tail of the descending order, so the whole
         * [KEEP, 1024) region is sorted when this finishes. */
        uint width = 2u;
        while (width < survivors) width <<= 1u;
        for (uint size = 2u; size <= width; size <<= 1u) {
            for (uint stride = size >> 1; stride > 0u; stride >>= 1u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = tid; k < width; k += (uint)nthreads) {
                    const uint j = k ^ stride;
                    if (j <= k) continue;
                    const uint ka = KEEP + k;
                    const uint ja = KEEP + j;
                    const bool ascending = (k & size) == 0u;
                    if (ascending
                            ? qwen4_qsa_worse(tscores[ka], tindices[ka],
                                              tscores[ja], tindices[ja])
                            : qwen4_qsa_worse(tscores[ja], tindices[ja],
                                              tscores[ka], tindices[ka])) {
                        const float ts = tscores[ka];
                        const uint ti = tindices[ka];
                        tscores[ka] = tscores[ja];
                        tindices[ka] = tindices[ja];
                        tscores[ja] = ts;
                        tindices[ja] = ti;
                    }
                }
            }
        }
        /* [K descending][reverse of [KEEP,1024)] is one bitonic sequence:
         * reversing puts every -INF pad at the block's front and the
         * survivors ascending behind it.  Ten merge stages over [0,1024)
         * then yield the fully descending order with the top 512 first.
         * Every (k, k^stride) pair has k < 1024, so the legacy ascending
         * term is constant here. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 256u; i += (uint)nthreads) {
            const uint a = KEEP + i;
            const uint b = 1023u - i;
            const float ts = tscores[a];
            const uint ti = tindices[a];
            tscores[a] = tscores[b];
            tindices[a] = tindices[b];
            tscores[b] = ts;
            tindices[b] = ti;
        }
        for (uint stride = 512u; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = tid; k < 1024u; k += (uint)nthreads) {
                const uint j = k ^ stride;
                if (j <= k) continue;
                if (qwen4_qsa_worse(tscores[k], tindices[k],
                                    tscores[j], tindices[j])) {
                    const float ts = tscores[k];
                    const uint ti = tindices[k];
                    tscores[k] = tscores[j];
                    tindices[k] = tindices[j];
                    tscores[j] = ts;
                    tindices[j] = ti;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        /* Early tiles (threshold missing or too many survivors) and any
         * non-production geometry: rebuild the legacy tile region from
         * global memory — the fast-path compaction may have copied
         * survivors over their own originals, so stale tail entries must
         * not survive into the sort — then run the full 2048-wide bitonic
         * network exactly like kernel_qwen4_qsa_bitonic_stream_f32. */
        for (uint i = tid; i < CHUNK - args.keep; i += (uint)nthreads) {
            float score;
            uint index;
            if (i < args.tile_blocks) {
                score = tile_scores[(ulong)tgp * args.tile_blocks + i];
                index = args.block_start + i;
            } else {
                score = -INFINITY;
                index = 0xFFFFFFFFu;
            }
            tscores[args.keep + i] = score;
            tindices[args.keep + i] = index;
        }
        for (uint size = 2u; size <= CHUNK; size <<= 1u) {
            for (uint stride = size >> 1; stride > 0u; stride >>= 1u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = tid; k < CHUNK; k += (uint)nthreads) {
                    const uint j = k ^ stride;
                    if (j <= k) continue;
                    const bool ascending = (k & size) == 0u;
                    if (ascending
                            ? qwen4_qsa_worse(tscores[k], tindices[k],
                                              tscores[j], tindices[j])
                            : qwen4_qsa_worse(tscores[j], tindices[j],
                                              tscores[k], tindices[k])) {
                        const float ts = tscores[k];
                        const uint ti = tindices[k];
                        tscores[k] = tscores[j];
                        tindices[k] = tindices[j];
                        tscores[j] = ts;
                        tindices[j] = ti;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < args.keep; i += (uint)nthreads) {
        cand_scores[(ulong)tgp * args.keep + i] = tscores[i];
        cand_indices[(ulong)tgp * args.keep + i] = tindices[i];
    }
    threadgroup uint finite_count[1];
    if (tid == 0u) finite_count[0] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.keep; i += (uint)nthreads) {
        if (isfinite(tscores[i]))
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint *)finite_count, 1u,
                memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) counts[tgp] = finite_count[0];
}


/* Parallel ordered top-k for the single-query decode path.  The single-thread
 * heap controller above serializes roughly blocks*log(top_k) global-memory
 * sift steps on one lane; this network sorts 2048-entry chunks in threadgroup
 * memory across the whole GPU and keeps each chunk's top_k, so the global
 * top-k survives every merge level.  The comparator is exactly
 * qwen4_qsa_worse (score descending, index ascending), and block indices are
 * unique, so the ordered result matches the heap path bit for bit. */
kernel void kernel_qwen4_qsa_bitonic_topk_f32(
        constant qwen4_qsa_bitonic_args &args [[buffer(0)]],
        device const float *src_scores [[buffer(1)]],
        device const uint  *src_indices [[buffer(2)]],
        device float       *dst_scores [[buffer(3)]],
        device uint        *dst_indices [[buffer(4)]],
        device uint        *out_count [[buffer(5)]],
        threadgroup float  *tscores [[threadgroup(0)]],
        threadgroup uint   *tindices [[threadgroup(1)]],
        uint tgp [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort nthreads [[threads_per_threadgroup]]) {
    constexpr uint CHUNK = 2048u;
    const uint base = tgp * CHUNK;
    for (uint i = tid; i < CHUNK; i += (uint)nthreads) {
        const uint src = base + i;
        if (src < args.src_entries) {
            tscores[i] = src_scores[src];
            tindices[i] = args.src_implicit_index ? src : src_indices[src];
        } else {
            tscores[i] = -INFINITY;
            tindices[i] = 0xFFFFFFFFu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2u; size <= CHUNK; size <<= 1u) {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = tid; k < CHUNK; k += (uint)nthreads) {
                const uint j = k ^ stride;
                if (j <= k) continue;
                const bool ascending = (k & size) == 0u;
                if (ascending
                        ? qwen4_qsa_worse(tscores[k], tindices[k],
                                          tscores[j], tindices[j])
                        : qwen4_qsa_worse(tscores[j], tindices[j],
                                          tscores[k], tindices[k])) {
                    const float score = tscores[k];
                    const uint index = tindices[k];
                    tscores[k] = tscores[j];
                    tindices[k] = tindices[j];
                    tscores[j] = score;
                    tindices[j] = index;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < args.keep; i += (uint)nthreads) {
        const uint dst = args.final_pass ? i : tgp * args.keep + i;
        dst_scores[dst] = tscores[i];
        dst_indices[dst] = tindices[i];
    }
    if (args.final_pass && tgp == 0u) {
        threadgroup uint finite_count[1];
        if (tid == 0u) finite_count[0] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < args.keep; i += (uint)nthreads) {
            if (isfinite(tscores[i]))
                atomic_fetch_add_explicit(
                    (threadgroup atomic_uint *)finite_count, 1u,
                    memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) out_count[0] = finite_count[0];
    }
}
