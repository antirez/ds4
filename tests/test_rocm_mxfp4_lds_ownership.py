"""Execute the production LDSB kernel's control flow with host device shims.

The dot/reduction shims do not model GPU arithmetic or barriers. Instrumenting
the actual output store checks CTA ownership and tail bounds, including writes
of identical values that an output-parity test would miss.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "rocm/ds4_rocm_moe.cuh"
NAME = "moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel"


def production_kernel():
    source = SOURCE.read_text()
    start = source.index("__global__ static void " + NAME + "(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    kernel = source[start:end]
    store = "mid_out[off] ="
    assert kernel.count(store) == 1, "update instrumentation for changed store"
    return kernel.replace(store, "record_write(off); " + store)


SHIMS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
#define min(a,b) ((a)<(b)?(a):(b))
typedef struct { uint32_t x,y,z; } dims;
static dims blockIdx, blockDim, threadIdx;
typedef struct { int x,y,z,w; } int4;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } cuda_block_q8_K;
typedef struct { uint8_t e, qs[16]; } cuda_block_mxfp4;
_Alignas(16) char sB[16 * 272];
static uint32_t writes[1030 * 16];
static uint32_t active_count;
static void record_write(uint64_t off) {
    if (off >= (uint64_t)active_count * 16) {
        fprintf(stderr, "output out of bounds: %llu\n", (unsigned long long)off);
        exit(1);
    }
    writes[off]++;
}
static float quarter_warp_sum_f32(float v, uint32_t lane) {
    (void)lane; return v;
}
static void dev_dot_mxfp4_q8_K_block8(const cuda_block_mxfp4 *w,
        const cuda_block_q8_K *x0, const cuda_block_q8_K *x1,
        const cuda_block_q8_K *x2, const cuda_block_q8_K *x3,
        const cuda_block_q8_K *x4, const cuda_block_q8_K *x5,
        const cuda_block_q8_K *x6, const cuda_block_q8_K *x7,
        uint32_t np, float out[8]) {
    (void)w; (void)x0; (void)x1; (void)x2; (void)x3;
    (void)x4; (void)x5; (void)x6; (void)x7;
    for (uint32_t p=0; p<np; p++) out[p] += 1.0f;
}
"""


DRIVER = r"""
int main(void) {
    static const uint32_t cases[] = {1,7,8,31,32,33,127,128,129,255,256,257,513,1024};
    static float mid[1030 * 16], weights[1030];
    static cuda_block_q8_K xq[1030 * 2];
    _Alignas(16) static char gate[16 * 272], up[16 * 272];
    uint32_t pairs[1030], starts[9], experts[9] = {0}, counts[1], offsets[1] = {0};
    blockDim.x = 256;
    for (uint32_t c=0; c<sizeof(cases)/sizeof(cases[0]); c++) {
        active_count = counts[0] = cases[c];
        const uint32_t total = (active_count + 127) / 128;
        for (uint32_t p=0; p<active_count; p++) {
            pairs[p] = active_count - p - 1;
            weights[p] = 1.0f;
            for (uint32_t r=0; r<16; r++) writes[p*16+r] = 0;
        }
        for (uint32_t t=0; t<total; t++) starts[t] = t*128;
        for (blockIdx.y=0; blockIdx.y<total; blockIdx.y++)
        for (blockIdx.x=0; blockIdx.x<2; blockIdx.x++)
        for (threadIdx.x=0; threadIdx.x<256; threadIdx.x++) {
            moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel(
                NULL, NULL, mid, gate, up, xq, pairs, offsets, counts,
                &total, experts, starts, weights, sizeof(gate), 272,
                2, 16, 6, 0, 0, 7.0f);
        }
        for (uint32_t p=0; p<active_count; p++)
        for (uint32_t r=0; r<16; r++) {
            if (writes[p*16+r] != 1) {
                fprintf(stderr, "count=%u pair=%u row=%u has %u writers\n",
                        active_count,p,r,writes[p*16+r]);
                return 1;
            }
        }
    }
    puts("ROCm MXFP4 LDSB production ownership: 14 sizes, one writer per output");
    return 0;
}
"""


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-lds-") as tmp:
        source = Path(tmp) / "ownership.c"
        binary = Path(tmp) / "ownership"
        source.write_text(SHIMS + production_kernel() + DRIVER)
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-O2", "-Wno-unknown-pragmas", str(source),
            "-lm", "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)
