"""Native CUDA/HIP oracle for the production Q8_K register-bsum change.

Both arms execute on the GPU: the reference retains the former global-byte
bsum loop. All preceding scale/quantization code is shared so compiler-specific
rounding is held constant. Run --backend cuda or --backend hip on that host;
--emit writes the standalone source without claiming a native compilation.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def kernel(backend):
    path = ROOT / ("ds4_cuda.cu" if backend == "cuda" else "rocm/ds4_rocm_moe.cuh")
    source = path.read_text()
    signature = "__global__ static void q8_K_quantize_kernel("
    if source.count(signature) != 1:
        raise ValueError("expected one production quantizer")
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise ValueError("unterminated production quantizer")
    return source[start:end]


LEGACY_TAIL = r"""
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}
"""

PREAMBLE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
#if defined(__HIP_PLATFORM_AMD__)
// Match the production ROCm translation unit, including wave64 masks.
#define MASK_T uint64_t
#endif
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
#endif
#include "cuda/ds4_q8_k_bsum.h"
enum { CUDA_QK_K = 256, GUARD = 4 };
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } cuda_block_q8_K;
static_assert(sizeof(cuda_block_q8_K) == 292, "Q8_K ABI");
#define CHECK(call) do { if ((call) != GPU(Success)) { \
    fprintf(stderr, "GPU failure line %d: %s\n", __LINE__, #call); exit(1); \
} } while (0)
"""

DRIVER = r"""
static uint32_t random_u32(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static int run_case(uint32_t k, uint32_t rows, unsigned pattern,
                    GPU(Stream_t) stream) {
    const size_t elements = (size_t)k * rows;
    const size_t blocks = elements / CUDA_QK_K;
    const size_t bytes = (blocks + 2u * GUARD) * sizeof(cuda_block_q8_K);
    float *x = (float *)malloc(elements * sizeof(float));
    float *readback = (float *)malloc(elements * sizeof(float));
    unsigned char *baseline = (unsigned char *)malloc(bytes);
    unsigned char *candidate = (unsigned char *)malloc(bytes);
    if (!x || !readback || !baseline || !candidate) exit(1);
    uint32_t seed = 0x24f519u + pattern;
    for (size_t b = 0; b < blocks; b++) {
        const float scale = ldexpf(1.0f, (int)(b % 17u) - 8);
        for (unsigned i = 0; i < CUDA_QK_K; i++) {
            float v;
            if (pattern == 0) v = 0.0f;
            else if (pattern == 1) v = (i & 1u ? -127.0f : 127.0f) * scale;
            else if (pattern == 2) v = ((int)(random_u32(&seed) % 16383u) - 8191) * scale / 64.0f;
            else v = ((int)(i % 255u) - 127) * scale * 0.5f;
            x[b * CUDA_QK_K + i] = v;
        }
        /* Equal-magnitude opposite signs exercise the canonical tie tree;
         * half-step inputs exercise quantization rounding after that tree. */
        if (pattern == 3) {
            x[b * CUDA_QK_K + 0] = -127.0f * scale;
            x[b * CUDA_QK_K + 128] = 127.0f * scale;
        }
    }
    float *gx = NULL;
    cuda_block_q8_K *gold = NULL, *got = NULL;
    CHECK(GPU(Malloc)((void **)&gx, elements * sizeof(float)));
    CHECK(GPU(Malloc)((void **)&gold, bytes));
    CHECK(GPU(Malloc)((void **)&got, bytes));
    CHECK(GPU(Memcpy)(gx, x, elements * sizeof(float), GPU(MemcpyHostToDevice)));
    CHECK(GPU(MemsetAsync)(gold, 0xa5, bytes, stream));
    CHECK(GPU(MemsetAsync)(got, 0xa5, bytes, stream));
    /* Oversized grid also checks uniform block/row early returns. */
    const dim3 grid(k / CUDA_QK_K + 1u, rows + 1u, 1u);
    q8_K_quantize_reference<<<grid, 256, 0, stream>>>(gold + GUARD, gx, k, rows);
    CHECK(GPU(GetLastError)());
    q8_K_quantize_kernel<<<grid, 256, 0, stream>>>(got + GUARD, gx, k, rows);
    CHECK(GPU(GetLastError)());
    CHECK(GPU(StreamSynchronize)(stream));
    CHECK(GPU(Memcpy)(baseline, gold, bytes, GPU(MemcpyDeviceToHost)));
    CHECK(GPU(Memcpy)(candidate, got, bytes, GPU(MemcpyDeviceToHost)));
    CHECK(GPU(Memcpy)(readback, gx, elements * sizeof(float), GPU(MemcpyDeviceToHost)));
    int ok = memcmp(x, readback, elements * sizeof(float)) == 0;
    const size_t guard_bytes = GUARD * sizeof(cuda_block_q8_K);
    for (size_t i = 0; i < bytes; i++) {
        if (candidate[i] != baseline[i] ||
            ((i < guard_bytes || i >= bytes - guard_bytes) &&
             (candidate[i] != 0xa5 || baseline[i] != 0xa5))) {
            fprintf(stderr, "Q8_K mismatch K=%u rows=%u pattern=%u byte=%zu\n",
                    k, rows, pattern, i);
            ok = 0;
            break;
        }
    }
    CHECK(GPU(Free)(got)); CHECK(GPU(Free)(gold)); CHECK(GPU(Free)(gx));
    free(candidate); free(baseline); free(readback); free(x);
    return ok;
}

int main(void) {
    int devices = 0;
    CHECK(GPU(GetDeviceCount)(&devices));
    if (devices == 0) { fputs("Native GPU required\n", stderr); return 1; }
    CHECK(GPU(SetDevice)(0));
    GPU(Stream_t) stream;
    CHECK(GPU(StreamCreate)(&stream));
    const uint32_t ks[] = {256, 512, 1024, 4096, 8192};
    const uint32_t ns[] = {1, 9, 17, 128};
    unsigned cases = 0;
    for (unsigned k = 0; k < sizeof(ks) / sizeof(ks[0]); k++)
    for (unsigned n = 0; n < sizeof(ns) / sizeof(ns[0]); n++)
    for (unsigned p = 0; p < 4; p++) {
        if (!run_case(ks[k], ns[n], p, stream)) return 1;
        cases++;
    }
    CHECK(GPU(StreamDestroy)(stream));
    printf("Q8_K native parity PASS: %u cases, full bytes/guards/source/non-default stream\n", cases);
    return 0;
}
"""


def source_for(backend):
    actual = kernel(backend)
    marker = "    int qv = 0;"
    if actual.count(marker) != 1 or actual.count("ds4_q8_K_bsum16(") != 1:
        raise ValueError("update native reference extraction for changed quantizer")
    reference = actual[:actual.index(marker)] + LEGACY_TAIL
    reference = reference.replace("q8_K_quantize_kernel", "q8_K_quantize_reference", 1)
    return PREAMBLE + actual + "\n" + reference + DRIVER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--emit", type=Path)
    args = parser.parse_args()
    source = source_for(args.backend)
    if args.emit:
        args.emit.write_text(source)
        print(f"Wrote {args.emit}; native compilation/execution not performed")
        return
    variable, default = ("NVCC", "nvcc") if args.backend == "cuda" else ("HIPCC", "hipcc")
    compiler = shlex.split(os.environ.get(variable, default))
    if not compiler or not shutil.which(compiler[0]):
        parser.error(f"{variable} compiler required; use --emit only to inspect the source")
    flags = (shlex.split(os.environ.get("NVCCFLAGS", "-O3 --use_fast_math"))
             if args.backend == "cuda" else
             shlex.split(os.environ.get("ROCM_CFLAGS", "-O3 -ffast-math -fno-finite-math-only")))
    with tempfile.TemporaryDirectory(prefix="ds4-q8-native-") as tmp:
        path = Path(tmp) / ("oracle.cu" if args.backend == "cuda" else "oracle.hip")
        binary = Path(tmp) / "oracle"
        path.write_text(source)
        subprocess.run(compiler + flags + ["-std=c++17", "-I", str(ROOT),
                       str(path), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
