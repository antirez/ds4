"""Check the production ROCm raw-KV ring store and its launch policy on the host.

Extract the kernel and wrapper, instrument actual writes, and emulate independent
thread schedules with ASan/UBSan. The oracle applies all input rows sequentially
and checks that the kernel writes each ring cell at most once. Covers batches
larger than the ring, 32-bit position overflow, guards and invalid arguments.
This checks ownership and indexing; it does not execute HIP or model GPU timing.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production_bodies():
    kernel = extract_function(
        (ROOT / "rocm/ds4_rocm_fp8_kv.cuh").read_text(),
        "__global__ static void store_raw_kv_batch_kernel(")
    store = "raw[(uint64_t)row * head_dim + d] = f16_bits_to_f32(hb);"
    if kernel.count(store) != 1:
        raise AssertionError("expected one raw-KV output store to instrument")
    kernel = kernel.replace("__global__ ", "").replace(
        store, "note_store((uint64_t)row * head_dim + d);\n    " + store)
    wrapper = extract_function(
        (ROOT / "rocm/ds4_rocm_attention_launch.cuh").read_text(),
        'extern "C" int ds4_gpu_store_raw_kv_batch_tensor(')
    launch = "store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>("
    if wrapper.count(launch) != 1:
        raise AssertionError("expected one raw-KV launch to lower to the host shim")
    wrapper = wrapper.replace('extern "C" ', "").replace(
        launch, "launch_store((n + 255) / 256, ")
    return kernel, wrapper


SHIM = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_gpu.h"
#define min(a, b) ((a) < (b) ? (a) : (b))
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
static struct { uint64_t x; } blockIdx, threadIdx, blockDim = {256};
static unsigned *writes;
static uint64_t cells, launched_blocks, launches;
static unsigned schedule;
static uint16_t f32_to_f16_bits_hip_round(float x) {
    _Float16 h = (_Float16)x;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}
static float f16_bits_to_f32(uint16_t bits) {
    _Float16 h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}
static void note_store(uint64_t index) {
    assert(index < cells);
    assert(++writes[index] == 1);  /* Fail even if duplicate stores agree. */
}
static int cudaGetLastError(void) { return 0; }
static int cuda_ok(int error, const char *label) { (void)label; return error == 0; }
'''

LAUNCHER = r'''
static void launch_store(uint64_t blocks, float *raw, const float *kv,
                         uint32_t cap, uint32_t pos, uint32_t tokens,
                         uint32_t dim) {
    ++launches;
    launched_blocks = blocks;
    /* Also execute inactive threads beyond the legal grid: the kernel must
     * guard them. Three schedules expose any repeated ring-cell ownership. */
    const uint64_t threads = (blocks + 2) * 256;
    uint64_t *order = malloc((size_t)threads * sizeof(*order));
    assert(order);
    for (uint64_t i = 0; i < threads; ++i) order[i] = i;
    if (schedule == 2) {
        uint32_t seed = 71;
        for (uint64_t i = threads - 1; i; --i) {
            seed = seed * 1664525u + 1013904223u;
            const uint64_t j = seed % (i + 1);
            const uint64_t swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
    }
    for (uint64_t i = 0; i < threads; ++i) {
        const uint64_t tid = schedule == 1 ? threads - 1 - i : order[i];
        blockIdx.x = tid / 256;
        threadIdx.x = tid % 256;
        store_raw_kv_batch_kernel(raw, kv, cap, pos, tokens, dim);
    }
    free(order);
}
'''

DRIVER = r'''
static void run_case(uint32_t cap, uint32_t tokens, uint32_t dim, uint32_t pos) {
    enum {GUARD = 16};
    const float sentinel = -98765.0f;
    cells = (uint64_t)cap * dim;
    const uint64_t inputs = (uint64_t)tokens * dim;
    float *storage = malloc((size_t)(cells + 2 * GUARD) * sizeof(float));
    float *expected = malloc((size_t)(cells + 2 * GUARD) * sizeof(float));
    float *input = malloc((size_t)inputs * sizeof(float));
    float *original = malloc((size_t)inputs * sizeof(float));
    writes = calloc((size_t)cells, sizeof(*writes));
    assert(storage && expected && input && original && writes);
    for (uint64_t i = 0; i < inputs; ++i)
        input[i] = (float)(int)(i % 20001) / 32.0f - 125.0f;
    memcpy(original, input, (size_t)inputs * sizeof(float));
    for (uint64_t i = 0; i < cells + 2 * GUARD; ++i) expected[i] = sentinel;
    /* Independent sequential ring oracle retains the newest row in each slot.
     * It never truncates the input batch or reuses the suffix-index formula. */
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint64_t row = ((uint64_t)pos + t) % cap;
        for (uint32_t d = 0; d < dim; ++d)
            expected[GUARD + row * dim + d] = (float)(_Float16)input[(uint64_t)t * dim + d];
    }
    ds4_gpu_tensor out = {storage + GUARD, cells * sizeof(float)};
    ds4_gpu_tensor in = {input, inputs * sizeof(float)};
    for (schedule = 0; schedule < 3; ++schedule) {
        for (uint64_t i = 0; i < cells + 2 * GUARD; ++i) storage[i] = sentinel;
        memset(writes, 0, (size_t)cells * sizeof(*writes));
        assert(ds4_gpu_store_raw_kv_batch_tensor(&out, &in, cap, pos, tokens, dim));
        const uint64_t stored = (tokens < cap ? tokens : cap) * (uint64_t)dim;
        assert(launched_blocks == (stored + 255) / 256);
        assert(!memcmp(storage, expected, (size_t)(cells + 2 * GUARD) * sizeof(float)));
        assert(!memcmp(input, original, (size_t)inputs * sizeof(float)));
        uint64_t count = 0;
        for (uint64_t i = 0; i < cells; ++i) count += writes[i];
        assert(count == stored);
    }
    free(writes);
    free(original);
    free(input);
    free(expected);
    free(storage);
}

static void invalid_arguments(void) {
    float values[64] = {0};
    ds4_gpu_tensor out = {values, sizeof(values)}, in = {values, sizeof(values)};
    const uint64_t before = launches;
    assert(!ds4_gpu_store_raw_kv_batch_tensor(NULL, &in, 4, 0, 4, 4));
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, NULL, 4, 0, 4, 4));
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, &in, 0, 0, 4, 4));
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, &in, 4, 0, 0, 4));
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, &in, 4, 0, 4, 0));
    out.bytes = 63;
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, &in, 4, 0, 4, 4));
    out.bytes = sizeof(values);
    in.bytes = 63;
    assert(!ds4_gpu_store_raw_kv_batch_tensor(&out, &in, 4, 0, 4, 4));
    assert(launches == before);
}
int main(void) {
    const uint32_t caps[] = {1, 2, 4, 7, 17, 64};
    const uint32_t tokens[] = {1, 2, 3, 8, 19, 66, 131};
    const uint32_t dims[] = {1, 3, 32, 65, 512};
    const uint32_t positions[] = {0, 3, UINT32_MAX - 2, UINT32_MAX};
    unsigned cases = 0;
    for (unsigned c = 0; c < sizeof(caps) / sizeof(*caps); ++c)
    for (unsigned t = 0; t < sizeof(tokens) / sizeof(*tokens); ++t)
    for (unsigned d = 0; d < sizeof(dims) / sizeof(*dims); ++d)
    for (unsigned p = 0; p < sizeof(positions) / sizeof(*positions); ++p) {
        run_case(caps[c], tokens[t], dims[d], positions[p]);
        ++cases;
    }
    invalid_arguments();
    printf("PASS ROCm raw-KV store: %u shapes, 3 schedules each, newest-row "
           "oracle, single writers, position overflow, launch bounds and guards. HIP unverified.\n", cases);
    return 0;
}
'''


def main():
    kernel, wrapper = production_bodies()
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-raw-kv-") as tmp:
        source = Path(tmp) / "raw_kv.c"
        binary = Path(tmp) / "raw_kv"
        source.write_text(SHIM + kernel + LAUNCHER + wrapper + DRIVER)
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc")) +
            ["-std=c11", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
             "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
             "-fno-omit-frame-pointer", "-I", str(ROOT), str(source),
             "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
