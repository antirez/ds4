"""Check CUDA/ROCm raw KV indexing and writer ownership with host shims.

The extracted production kernels run with a 256-thread grid and instrumented
output stores. Conversion shims preserve test integers in [0, 1023], all exactly
representable in F16. They do not model half bit patterns, rounding, GPU
arithmetic, or scheduling; duplicate writers are detected even if values agree.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "cuda": REPO / "ds4_cuda.cu",
    "rocm": REPO / "rocm/ds4_rocm_fp8_kv.cuh",
}


def production_kernel(source, backend):
    signature = "__global__ static void store_raw_kv_batch_kernel("
    if source.count(signature) != 1:
        raise AssertionError(f"expected one {backend} raw KV kernel definition")
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise AssertionError(f"unterminated {backend} raw KV kernel definition")
    kernel = source[start:end].replace(
        "store_raw_kv_batch_kernel", f"store_raw_kv_batch_{backend}", 1
    )
    kernel, stores = re.subn(
        r"\braw\[([^\]\n]+)\]\s*=",
        lambda match: f"record_write({match[1]}); {match[0]}",
        kernel,
    )
    if stores != 1:
        raise AssertionError(f"update instrumentation for changed {backend} raw store")
    return kernel


SHIMS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define __global__
#define min(a, b) ((a) < (b) ? (a) : (b))
typedef struct { uint32_t x; } dim;
static dim blockIdx, blockDim, threadIdx;
static const char *backend_name;
static uint32_t case_cap, case_n, case_pos;
static uint64_t raw_count;
static unsigned *writes;

static void fail(const char *reason, uint64_t index) {
    fprintf(stderr, "%s cap=%u N=%u pos=%u index=%llu: %s\n", backend_name,
        case_cap, case_n, case_pos, (unsigned long long)index, reason);
    exit(1);
}

static void record_write(uint64_t index) {
    if (index >= raw_count) fail("output out of bounds", index);
    if (++writes[index] != 1) fail("multiple writers", index);
}

/* Deliberately preserve exact small integers, not their IEEE half encoding. */
static float __float2half(float value) { return value; }
static float __half2float(float value) { return value; }
static uint16_t f32_to_f16_bits_hip_round(float value) { return (uint16_t)value; }
static float f16_bits_to_f32(uint16_t value) { return (float)value; }
"""


DRIVER = r"""
typedef void (*kernel_fn)(float *, const float *, uint32_t, uint32_t, uint32_t, uint32_t);

static float source_value(uint64_t index, unsigned pattern) {
    /* Together these two exact integer patterns uniquely identify every input. */
    return (float)((index >> (pattern * 10u)) & 1023u);
}

static void run_case(kernel_fn kernel, uint32_t cap, uint32_t n, uint32_t pos) {
    const uint32_t head_dim = 17;
    const uint64_t source_count = (uint64_t)n * head_dim;
    const uint32_t stored = n < cap ? n : cap;
    const uint64_t work_count = (uint64_t)stored * head_dim;
    const uint64_t blocks = (work_count + 255u) / 256u;
    const float sentinel = -12345.0f;
    case_cap = cap;
    case_n = n;
    case_pos = pos;
    raw_count = (uint64_t)cap * head_dim;
    float *raw = malloc(raw_count * sizeof(*raw));
    float *reference = malloc(raw_count * sizeof(*reference));
    float *source = malloc((source_count ? source_count : 1u) * sizeof(*source));
    unsigned char *touched = calloc(raw_count, sizeof(*touched));
    writes = calloc(raw_count, sizeof(*writes));
    if (!raw || !reference || !source || !touched || !writes) fail("allocation failed", 0);

    for (unsigned pattern = 0; pattern < 2; pattern++) {
        memset(writes, 0, raw_count * sizeof(*writes));
        memset(touched, 0, raw_count * sizeof(*touched));
        for (uint64_t i = 0; i < raw_count; i++) raw[i] = reference[i] = sentinel;
        for (uint64_t i = 0; i < source_count; i++) source[i] = source_value(i, pattern);

        /* Independent oracle inserts every token sequentially with 64-bit positions. */
        for (uint32_t t = 0; t < n; t++) {
            const uint32_t row = ((uint64_t)pos + t) % cap;
            for (uint32_t d = 0; d < head_dim; d++) {
                const uint64_t index = (uint64_t)row * head_dim + d;
                reference[index] = source[(uint64_t)t * head_dim + d];
                touched[index] = 1;
            }
        }

        /* Match the wrapper's launch rounding, including inactive tail threads. */
        blockDim.x = 256;
        for (blockIdx.x = 0; blockIdx.x < blocks; blockIdx.x++)
            for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++)
                kernel(raw, source, cap, pos, n, head_dim);

        uint64_t total_writes = 0;
        for (uint64_t i = 0; i < raw_count; i++) {
            if (raw[i] != reference[i]) fail("value/source index differs from sequential insertion", i);
            if (writes[i] != touched[i]) fail("incorrect writer count for touched/untouched row", i);
            total_writes += writes[i];
        }
        if (total_writes != work_count) fail("incorrect surviving suffix length", total_writes);
        for (uint64_t i = 0; i < source_count; i++)
            if (source[i] != source_value(i, pattern)) fail("source was modified", i);
    }
    free(raw);
    free(reference);
    free(source);
    free(touched);
    free(writes);
}

int main(void) {
    static const uint32_t capacities[] = {1, 31, 256};
    const kernel_fn kernels[] = {store_raw_kv_batch_cuda, store_raw_kv_batch_rocm};
    const char *names[] = {"CUDA", "ROCm"};
    unsigned cases = 0;
    for (unsigned backend = 0; backend < 2; backend++) {
        backend_name = names[backend];
        for (unsigned c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
            const uint32_t cap = capacities[c];
            const uint32_t counts[] = {cap - 1u, cap, cap + 1u, 2u * cap + 17u};
            const uint32_t positions[] = {0, cap - 1u, UINT32_MAX - 5u};
            for (unsigned n = 0; n < sizeof(counts) / sizeof(counts[0]); n++)
                for (unsigned p = 0; p < sizeof(positions) / sizeof(positions[0]); p++) {
                    run_case(kernels[backend], cap, counts[n], positions[p]);
                    cases++;
                }
        }
    }
    printf("CUDA/ROCm raw KV ownership: %u cases, two source-index patterns passed\n", cases);
    return 0;
}
"""


class RawKvOwnershipTests(unittest.TestCase):
    def test_production_kernel_ownership(self):
        kernels = "\n".join(
            production_kernel(path.read_text(), backend)
            for backend, path in SOURCES.items()
        )
        with tempfile.TemporaryDirectory(prefix="ds4-raw-kv-ownership-") as tmp:
            source = Path(tmp) / "ownership.c"
            source.write_text(SHIMS + kernels + DRIVER)
            compiler = shlex.split(os.environ.get("CC", "cc"))
            for optimization in ("-O0", "-O2"):
                with self.subTest(optimization=optimization):
                    binary = Path(tmp) / ("ownership" + optimization)
                    subprocess.run(compiler + [
                        "-std=c11", optimization, "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address", "-fno-omit-frame-pointer", "-g",
                        str(source), "-o", str(binary),
                    ], check=True)
                    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
