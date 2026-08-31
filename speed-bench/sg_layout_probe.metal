#include <metal_stdlib>
using namespace metal;

/* Layout probe: load 8x8 tiles whose values encode (row*8+col) and dump each
 * thread's matrix elements so the host can print the per-thread element
 * coordinates of the float accumulator and half operand forms. */
kernel void sg_layout_probe(
        device float *out [[buffer(0)]],
        threadgroup float *tile [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint tid = (uint)sg * 32u + lane;
    if (tid < 64u) tile[tid] = (float)tid; /* float tile: r*8+c */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) { out[0] = -1.0f; return; }

    simdgroup_float8x8 mf;
    simdgroup_load(mf, tile, 8, 0, false);
    out[1 + (uint)lane * 3u + 0u] = (float)lane;
    out[1 + (uint)lane * 3u + 1u] = mf.thread_elements()[0];
    out[1 + (uint)lane * 3u + 2u] = mf.thread_elements()[1];

    /* half tile in a separate region: value r*8+c so the host decodes
     * (row, col) the same way */
    threadgroup half *htile = (threadgroup half *)(tile + 64u);
    if (tid < 64u) htile[tid] = (half)((float)((tid / 8u) * 8u + (tid % 8u)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 mh;
    simdgroup_load(mh, htile, 8, 0, false);
    for (uint e = 0; e < 8u; e++)
        out[1 + 96u + (uint)lane * 8u + e] = (float)mh.thread_elements()[e];
}
