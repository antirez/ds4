"""Fault-inject the actual GB10 F16 compressor dispatcher and cache on the host.

CUDA calls are replaced with observable stubs; kernel arithmetic is covered by
test_cuda_f16_compressor.py. This verifies enqueue/fallback and cache lifetime
contracts, not GPU scheduling or CUDA driver behavior.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4_cuda.cu").read_text()
SIGNATURES = [
    "static void cuda_f16_pair_chunk32_release_all(",
    "static void cuda_model_range_release_all(",
    "static int cuda_f16_pair_chunk32_get(",
    "static bool cuda_f16_compressor_ranges_overlap(",
    'extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(',
]

STUBS = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <unordered_map>
using __half = uint16_t;
using __half2 = uint32_t;
using cudaStream_t = int;
using cudaError_t = int;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorLaunchFailure = 4 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };
static unsigned allocations, frees, repacks, packed_writers, ordered_writers;
static unsigned synchronizations, invalidations, resolve_calls, checks;
static bool fail_alloc, fail_repack, fail_sync;
static int pending_error, active_device = 7;
static cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
static int g_decode_graph_capturing, g_quality_mode, g_n_gpus = 1;
static int g_cuda_is_gb10[] = {1, 0};
static struct { int device_id; } g_gpu[] = {{7}, {9}};
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device; };
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) { return t->device; }
static cudaStream_t cuda_decode_stream() { return g_decode_graph_capturing ? 13 : 0; }
static int cudaGetDevice(int *d) { *d = active_device; return 0; }
static int cudaSetDevice(int d) { active_device = d; return 0; }
static int cudaGetLastError() { int e = pending_error; pending_error = 0; return e; }
static const char *cudaGetErrorString(int) { return "injected error"; }
static int cuda_ok(int e, const char *) { return e == 0; }
static int cudaStreamIsCapturing(int, cudaStreamCaptureStatus *s) { *s = capture_status; return 0; }
static int cudaStreamSynchronize(int) { ++synchronizations; return fail_sync ? 4 : 0; }
static int cudaMalloc(__half2 **p, size_t bytes) {
    ++allocations;
    if (fail_alloc) return cudaErrorMemoryAllocation;
    *p = (__half2 *)malloc(bytes); return *p ? 0 : cudaErrorMemoryAllocation;
}
static int cudaFree(void *p) { ++frees; free(p); return 0; }
static int cudaHostUnregister(void *) { return 0; }
static void ds4_gpu_decode_graphs_invalidate() { ++invalidations; }
static const char *cuda_resolve_weight_ptr(const void *base, uint64_t offset,
        uint64_t, int, const char *) {
    ++resolve_calls; return (const char *)((uintptr_t)base + offset);
}
template<class... T> static void f16_pair_chunk32_repack_kernel(T...) {
    ++repacks; if (fail_repack) pending_error = cudaErrorLaunchFailure;
}
template<class... T> static void matmul_f16_pair_compressor_store_chunk32_prefetch8_kernel(T...) {
    ++packed_writers;
}
template<class... T> static void matmul_f16_pair_compressor_store_ordered_chunks_kernel(T...) {
    ++ordered_writers;
}
struct cuda_f16_pair_chunk32_range {
    const void *host_base; uint64_t weight0_offset, weight1_offset, in_dim;
    uint32_t width; __half2 *device_ptr; int device_id;
};
static std::vector<cuda_f16_pair_chunk32_range> g_f16_pair_chunk32_ranges;
static std::mutex g_f16_pair_chunk32_mutex;
static uint64_t g_f16_pair_chunk32_bytes;
static int g_f16_pair_chunk32_disabled_after_oom;
struct cuda_model_range {
    bool host_registered; void *registered_base; void *device_ptr;
    bool arena_allocated, borrowed;
};
struct cuda_model_arena { void *device_ptr; };
static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static uint64_t g_model_range_bytes;
static void cuda_model_load_progress_reset() {}
static void check(bool ok, const char *why) {
    ++checks;
    if (!ok) { fprintf(stderr, "FAIL F16 compressor policy: %s\n", why); exit(1); }
}
'''

CASES = r'''
struct fixture {
    ds4_gpu_tensor out0{(void *)0x10000000, 4096, 0};
    ds4_gpu_tensor out1{(void *)0x10010000, 4096, 0};
    ds4_gpu_tensor state0{(void *)0x20000000, 262144, 0};
    ds4_gpu_tensor state1{(void *)0x30000000, 262144, 0};
    ds4_gpu_tensor x{(void *)0x40000000, 16384, 0};
    const void *model = (void *)0x100000000ull;
    uint64_t model_size = 1ull << 30, w0 = 0, w1 = 16u << 20, ape = 32u << 20;
    uint64_t in_dim = 4096;
    uint32_t width = 1024, ratio = 4, ape_type = 0;
    int run() {
        return ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            &out0, &out1, &state0, &state1, model, model_size, w0, w1, ape,
            ape_type, in_dim, width, &x, ratio, UINT32_MAX);
    }
};
static void reset() {
    cuda_model_range_release_all();
    allocations = frees = repacks = packed_writers = ordered_writers = 0;
    synchronizations = invalidations = resolve_calls = 0;
    fail_alloc = fail_repack = fail_sync = false; pending_error = 0;
    g_decode_graph_capturing = g_quality_mode = 0;
    capture_status = cudaStreamCaptureStatusNone;
    g_n_gpus = g_cuda_is_gb10[0] = 1; active_device = 7;
}
static void no_writer(const char *why) {
    check(allocations == 0 && repacks == 0 && packed_writers == 0 &&
          ordered_writers == 0, why);
}
int main() {
    for (unsigned shape = 0; shape < 3; ++shape) {
        reset(); fixture f;
        f.width = shape == 0 ? 256 : shape == 1 ? 1024 : 512;
        f.ratio = shape == 2 ? 128 : 4; f.ape_type = shape & 1;
        check(f.run() == 1 && allocations == 1 && repacks == 1 &&
              synchronizations == 1 && packed_writers == 1, "all three decode shapes admitted");
        const unsigned alloc = allocations;
        g_decode_graph_capturing = 1;
        check(f.run() == 1 && allocations == alloc && repacks == 1 &&
              synchronizations == 1 && packed_writers == 2, "captured hot cache does not allocate");
        g_decode_graph_capturing = 0;
        cuda_model_range_release_all();
        check(frees == 1 && invalidations == 1 && g_f16_pair_chunk32_ranges.empty() &&
              g_f16_pair_chunk32_bytes == 0 && active_device == 7,
              "model release invalidates graphs, frees cache and preserves device");
    }
    for (unsigned mode = 0; mode < 2; ++mode) {
        reset(); fixture f;
        if (mode == 0) g_decode_graph_capturing = 1;
        else capture_status = cudaStreamCaptureStatusActive;
        check(f.run() == 1 && allocations == 0 && repacks == 0 &&
              ordered_writers == 1, "cold capture uses ordered fused kernel");
    }
    reset(); fixture f;
    fail_alloc = true;
    check(f.run() == 1 && allocations == 1 && repacks == 0 && ordered_writers == 1,
          "OOM before enqueue keeps ordered fallback");
    fail_alloc = false;
    check(f.run() == 1 && allocations == 1 && ordered_writers == 2,
          "OOM disables repeated allocation attempts");
    cuda_model_range_release_all();
    check(!g_f16_pair_chunk32_disabled_after_oom, "model release resets OOM policy");
    check(f.run() == 1 && allocations == 2 && packed_writers == 1, "new model retries cache");
    for (unsigned mode = 0; mode < 2; ++mode) {
        reset(); fixture f; fail_repack = mode == 0; fail_sync = mode == 1;
        check(f.run() == -1 && allocations == 1 && frees == 1 && repacks == 1 &&
              ordered_writers == 0 && packed_writers == 0 &&
              g_f16_pair_chunk32_ranges.empty(), "failed repack never submits output writer");
    }
    reset(); g_f16_pair_chunk32_bytes = 1ull << 30;
    check(f.run() == 1 && allocations == 0 && ordered_writers == 1,
          "cache capacity retains ordered fallback");
    for (unsigned kind = 0; kind < 14; ++kind) {
        reset(); fixture f;
        switch (kind) {
        case 0: f.width = 512; break;
        case 1: f.width = 256; f.ratio = 128; break;
        case 2: f.in_dim = 2048; break;
        case 3: f.ratio = 0; break;
        case 4: f.in_dim = UINT64_MAX; break;
        case 5: f.width = UINT32_MAX; break;
        case 6: g_quality_mode = 1; break;
        case 7: g_n_gpus = 2; break;
        case 8: g_cuda_is_gb10[0] = 0; break;
        case 9: f.x.device = 1; break;
        case 10: f.out1.device = 1; break;
        case 11: f.state0.device = 1; break;
        case 12: f.state1.device = 1; break;
        case 13: f.out0.device = 1; break;
        }
        check(f.run() == 0, "unsupported shape/policy/tier falls back");
        no_writer("unsupported path submits no work");
    }
    for (unsigned kind = 0; kind < 12; ++kind) {
        reset(); fixture f;
        switch (kind) {
        case 0: f.x.bytes = 16383; break;
        case 1: f.out0.bytes = 4095; break;
        case 2: f.out1.bytes = 4095; break;
        case 3: f.state0.bytes = 32767; break;
        case 4: f.state1.bytes = 32767; break;
        case 5: f.model_size = f.w1 + 4096*1024*2 - 1; break;
        case 6: f.w0 = UINT64_MAX; break;
        case 7: f.w1 = UINT64_MAX; break;
        case 8: f.ape = UINT64_MAX; break;
        case 9: f.ape_type = 2; break;
        case 10: f.out0.ptr = nullptr; break;
        case 11: active_device = 9; break;
        }
        check(f.run() == -1, "malformed tensor or device rejects");
        no_writer("malformed argument submits no work");
    }
    for (unsigned kind = 0; kind < 10; ++kind) {
        reset(); fixture f;
        switch (kind) {
        case 0: f.out0.ptr = f.out1.ptr; break;
        case 1: f.out0.ptr = (void *)((uintptr_t)f.out1.ptr+4); break;
        case 2: f.out1.ptr = f.state0.ptr; break;
        case 3: f.state0.ptr = f.state1.ptr; break;
        case 4: f.out0.ptr = f.x.ptr; break;
        case 5: f.state1.ptr = f.x.ptr; break;
        case 6: f.out0.ptr = (void *)f.model; break;
        case 7: f.out1.ptr = (void *)((uintptr_t)f.model+f.w1); break;
        case 8: f.state0.ptr = (void *)((uintptr_t)f.model+f.ape); break;
        case 9: f.state1.ptr = (void *)((uintptr_t)f.x.ptr+16380); break;
        }
        check(f.run() == 0, "aliased producer/consumer uses fallback");
        no_writer("alias rejection precedes allocation and writer");
    }
    for (const char *flag : {
            "DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE", "DS4_CUDA_NO_F16_PAIR_MATMUL",
            "DS4_CUDA_SERIAL_F16_MATMUL", "DS4_CUDA_SERIAL_ROUTER",
            "DS4_CUDA_NO_ORDERED_F16_MATMUL"}) {
        reset(); setenv(flag, "0", 1); fixture f;
        check(f.run() == 0, "presence rollback remains authoritative");
        no_writer("rollback submits no work"); unsetenv(flag);
    }
    for (const char *flag : {"DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE",
                            "DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE_PREFETCH8"}) {
        reset(); setenv(flag, "0", 1); fixture f;
        check(f.run() == 1 && allocations == 0 && repacks == 0 && ordered_writers == 1,
              "transpose rollback retains ordered fused kernel"); unsetenv(flag);
    }
    reset(); fixture a, b; b.model = (void *)0x200000000ull;
    check(a.run() == 1 && b.run() == 1 && allocations == 2,
          "cache identity includes model mapping");
    b.w0 += 128;
    check(b.run() == 1 && allocations == 3, "cache identity includes weight range");
    reset();
    check(!cuda_f16_compressor_ranges_overlap((void *)16, 0, (void *)0, 32),
          "empty ranges do not alias");
    check(!cuda_f16_compressor_ranges_overlap((void *)16, 16, (void *)32, 16),
          "adjacent ranges do not alias");
    printf("CUDA F16 compressor policy PASS: %u checks\n", checks);
}
'''


def main():
    bodies = "\n\n".join(extract_function(SOURCE, signature) for signature in SIGNATURES)
    bodies = re.sub(r"<<<.*?>>>", "", bodies, flags=re.S)
    env = {key: value for key, value in os.environ.items() if not key.startswith("DS4_")}
    with tempfile.TemporaryDirectory(prefix="ds4-f16-compressor-policy-") as directory:
        source = Path(directory) / "policy.cpp"
        binary = Path(directory) / "policy"
        source.write_text(STUBS + bodies + CASES)
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O2", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", str(source), "-o", str(binary),
        ]
        subprocess.run(command, check=True, env=env)
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()
