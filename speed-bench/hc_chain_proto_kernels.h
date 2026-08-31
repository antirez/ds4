R"METAL(
#include <metal_stdlib>
using namespace metal;

struct hc_args {
    uint32_t rows;
    uint32_t hidden;   /* 2560 */
    uint32_t streams;  /* 4 */
    uint32_t lowrank;  /* 320 */
    uint32_t groups;   /* persistent grid size */
};

struct block_q8_0p { half d; int8_t qs[32]; };

static inline float bf16h(uint16_t v) {
    return as_type<float>((uint32_t)v << 16);
}

static inline float dot_q8(device const block_q8_0p *row,
                            device const float *x,
                            uint in_dim, ushort lane) {
    float sum = 0.0f;
    for (uint base = (uint)lane * 8u; base < in_dim; base += 32u * 8u) {
        const uint bi = base / 32u;
        const uint ib = base & 31u;
        device const block_q8_0p *b = row + bi;
        float dq = 0.0f;
        for (uint i = 0u; i < 8u; i++)
            dq = fma((float)b->qs[ib + i], x[base + i], dq);
        sum = fma((float)b->d, dq, sum);
    }
    return simd_sum(sum);
}

/* Reference dispatch 1: RMS norm + weight per (token, stream) row. */
kernel void k_norm(
        constant hc_args &a [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const ushort *normw [[buffer(2)]],
        device float *normalized [[buffer(7)]],
        threadgroup float *red [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint row = tgroup;
    if (row >= a.rows * a.streams) return;
    const uint s = row % a.streams;
    device const float *xr = streams + (size_t)row * a.hidden;
    float sum2 = 0.0f;
    for (uint col = (uint)sg * 32u + lane; col < a.hidden;
         col += 32u * (uint)nsg)
        sum2 = fma(xr[col], xr[col], sum2);
    sum2 = simd_sum(sum2);
    if (lane == 0u) red[sg] = sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float t = 0.0f;
        for (uint g = 0; g < (uint)nsg; g++) t += red[g];
        if (lane == 0u) red[0] = rsqrt(t / (float)a.hidden + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = red[0];
    device const ushort *nw = normw + (size_t)s * a.hidden;
    device float *dst = normalized + (size_t)row * a.hidden;
    for (uint col = (uint)sg * 32u + lane; col < a.hidden;
         col += 32u * (uint)nsg)
        dst[col] = xr[col] * scale * bf16h(nw[col]);
}

/* Reference dispatch 2: down projection + SiLU, production grid. */
kernel void k_down_silu(
        constant hc_args &a [[buffer(0)]],
        device const block_q8_0p *downw [[buffer(4)]],
        device float *normalized [[buffer(7)]],
        device float *low [[buffer(8)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + (uint)sg;
    const uint token = tgpig.y;
    if (out_row >= a.lowrank || token >= a.rows) return;
    const uint hc_dim = a.streams * a.hidden;
    device const block_q8_0p *rowp =
        downw + (size_t)out_row * (hc_dim / 32u);
    const float v = dot_q8(rowp, normalized + (size_t)token * hc_dim,
                           hc_dim, lane);
    if (lane == 0u)
        low[(size_t)token * a.lowrank + out_row] = v / (1.0f + exp(-v));
}

/* Reference dispatch 3: up projection + stream mix, production grid. */
kernel void k_up_mix(
        constant hc_args &a [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const ushort *normw [[buffer(2)]],
        device const block_q8_0p *upw [[buffer(5)]],
        device float *low [[buffer(8)]],
        device float *normalized [[buffer(7)]],
        device float *out [[buffer(6)]],
        threadgroup float *raw_gate [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint dim = tgpig.x;
    const uint token = tgpig.y;
    if (dim >= a.hidden || token >= a.rows || sg >= a.streams) return;
    const uint hc_dim = a.streams * a.hidden;
    device const block_q8_0p *rowp =
        upw + ((size_t)sg * a.hidden + dim) * (a.lowrank / 32u);
    const float gate = dot_q8(rowp, low + (size_t)token * a.lowrank,
                              a.lowrank, lane);
    if (lane == 0u) raw_gate[sg] = gate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u || lane != 0u) return;
    float value = 0.0f;
    for (uint s = 0; s < a.streams; s++) {
        value = fma(normalized[(size_t)token * hc_dim +
                               (size_t)s * a.hidden + dim],
                    1.0f / (1.0f + exp(-raw_gate[s])), value);
    }
    out[(size_t)token * a.hidden + dim] = value / (float)a.streams;
}

/* Persistent whole-stage fusion: one dispatch for the entire chain.
 * Each of a.groups groups recomputes the four RMS scales it needs from
 * raw streams (avoiding a grid barrier before the down phase), computes
 * its slice of the low vector, meets every other group at a device
 * atomic counter, then computes its slice of the mixed output.  The grid
 * is sized far below occupancy so all groups are resident and the
 * software barrier cannot deadlock. */
kernel void hc_read_persistent(
        constant hc_args &a [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const ushort *normw [[buffer(2)]],
        device const block_q8_0p *downw [[buffer(4)]],
        device const block_q8_0p *upw [[buffer(5)]],
        device float *out [[buffer(6)]],
        device float *normalized [[buffer(7)]],
        device float *low [[buffer(8)]],
        device atomic_uint *barrier [[buffer(9)]],
        threadgroup float *red [[threadgroup(0)]],
        uint tgroup [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (tgroup >= a.groups) return;
    const uint hc_dim = a.streams * a.hidden;
    for (uint token = 0u; token < a.rows; token++) {
        /* phase 1: this group's copies of the four row scales */
        float scales[4];
        for (uint s = 0; s < a.streams; s++) {
            device const float *xr = streams + (size_t)token * hc_dim +
                                     (size_t)s * a.hidden;
            float sum2 = 0.0f;
            for (uint col = (uint)sg * 32u + lane; col < a.hidden;
                 col += 32u * (uint)nsg)
                sum2 = fma(xr[col], xr[col], sum2);
            sum2 = simd_sum(sum2);
            if (lane == 0u) red[sg] = sum2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float t = 0.0f;
                for (uint g = 0; g < (uint)nsg; g++) t += red[g];
                if (lane == 0u) red[16] = rsqrt(t / (float)a.hidden + 1e-6f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scales[s] = red[16];
            if (tgroup == 0u) {
                /* group zero materializes the normalized rows during
                 * phase 1; the phase-2/3 grid barrier publishes them. */
                device const float *xr = streams +
                    (size_t)token * hc_dim + (size_t)s * a.hidden;
                device float *dst = normalized +
                    (size_t)token * hc_dim + (size_t)s * a.hidden;
                device const ushort *nw = normw + (size_t)s * a.hidden;
                for (uint col = (uint)sg * 32u + lane; col < a.hidden;
                     col += 32u * (uint)nsg)
                    dst[col] = xr[col] * scales[s] * bf16h(nw[col]);
            }
        }
        /* phase 2: down slice (lowrank / groups outputs per group) */
        const uint per_low = a.lowrank / a.groups;
        for (uint o0 = 0u; o0 < per_low; o0 += (uint)nsg) {
            const uint o = tgroup * per_low + o0 + (uint)sg;
            if (o < (tgroup + 1u) * per_low && o < a.lowrank) {
                device const block_q8_0p *rowp =
                    downw + (size_t)o * (hc_dim / 32u);
                float v = 0.0f;
                for (uint s = 0; s < a.streams; s++) {
                    float part = 0.0f;
                    for (uint base = (uint)lane * 8u; base < a.hidden;
                         base += 32u * 8u) {
                        const uint bi = base / 32u;
                        const uint ib = base & 31u;
                        device const block_q8_0p *b =
                            rowp + (size_t)s * (a.hidden / 32u) + bi;
                        float dq = 0.0f;
                        for (uint i = 0u; i < 8u; i++) {
                            const float x =
                                streams[(size_t)token * hc_dim +
                                        (size_t)s * a.hidden + base + i] *
                                scales[s] *
                                bf16h(normw[(size_t)s * a.hidden + base + i]);
                            dq = fma((float)b->qs[ib + i], x, dq);
                        }
                        part = fma((float)b->d, dq, part);
                    }
                    part = simd_sum(part);
                    v += part;
                }
                if (lane == 0u)
                    low[(size_t)token * a.lowrank + o] =
                        v / (1.0f + exp(-v));
            }
        }
        /* grid barrier: every group must publish its low slice */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u && sg == 0u)
            atomic_fetch_add_explicit(barrier, 1u, memory_order_relaxed);
        if (lane == 0u && sg == 0u) {
            while (atomic_load_explicit(barrier, memory_order_relaxed) <
                   a.groups * (token + 1u)) {
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* phase 3: up + mix slice (hidden / groups dims per group) */
        const uint per_hid = a.hidden / a.groups;
        for (uint d0 = 0u; d0 < per_hid; d0 += (uint)nsg) {
            const uint dim = tgroup * per_hid + d0 + (uint)sg;
            if (dim < (tgroup + 1u) * per_hid && dim < a.hidden) {
                float value = 0.0f;
                for (uint s = 0; s < a.streams; s++) {
                    device const block_q8_0p *rowp =
                        upw + ((size_t)s * a.hidden + dim) *
                                  (a.lowrank / 32u);
                    const float gate = dot_q8(
                        rowp, low + (size_t)token * a.lowrank,
                        a.lowrank, lane);
                    value = fma(normalized[(size_t)token * hc_dim +
                                           (size_t)s * a.hidden + dim],
                                1.0f / (1.0f + exp(-gate)), value);
                }
                out[(size_t)token * a.hidden + dim] =
                    value / (float)a.streams;
            }
        }
    }
}
)METAL"
