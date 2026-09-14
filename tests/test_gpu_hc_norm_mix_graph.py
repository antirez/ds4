"""Exercise actual HC graph admission/fallback with an instrumented backend.

This checks control flow and argument propagation, including launch failures;
it does not simulate GPU execution or measure performance.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]

SHIM = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#undef __APPLE__
#if TEST_APPLE
#define __APPLE__ 1
#endif
#define DS4_RMS_EPS 1.0e-6f
typedef struct { int id; } ds4_gpu_tensor;
typedef struct { void *map; uint64_t size; } ds4_model;
typedef struct { uint64_t abs_offset; } ds4_tensor;
static ds4_gpu_tensor output = {1}, normalized = {2}, input = {3};
static ds4_tensor weight = {128};
static ds4_model model = {&weight, 819200};
static int reference, disabled, old_apple, new_apple, available, probes;
static int fused_result, norm_result, project_result;
static unsigned trace;
bool metal_graph_use_reference_hc_decode(void) { return reference; }
char *test_getenv(const char *key) {
    assert(strcmp(key, "DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE") == 0);
    return disabled ? (char *)"1" : NULL;
}
#define getenv test_getenv
bool ds4_gpu_device_is_pre_m5_apple_silicon(void) { return old_apple; }
bool ds4_gpu_device_is_m5_apple_silicon(void) { return new_apple; }
int ds4_gpu_hc_rms_norm_mix_f16_available(void) { ++probes; return available; }
int ds4_gpu_hc_rms_norm_mix_f16_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t offset, uint32_t n, uint32_t m, float eps) {
    assert(out == &output && x == &input && map == model.map);
    assert(size == model.size && offset == weight.abs_offset);
    assert(n == 16384 && m == 24 && eps == DS4_RMS_EPS);
    trace = trace * 10 + 1;
    return fused_result;
}
int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x, uint32_t n, float eps) {
    assert(out == &normalized && x == &input && n == 16384);
    assert(eps == DS4_RMS_EPS);
    trace = trace * 10 + 2;
    return norm_result;
}
bool metal_graph_matmul_plain_tensor(ds4_gpu_tensor *out,
        const ds4_model *mdl, const ds4_tensor *w, uint64_t n, uint64_t m,
        const ds4_gpu_tensor *x, uint64_t tokens) {
    assert(out == &output && mdl == &model && w == &weight);
    assert(n == 16384 && m == 24 && x == &normalized && tokens == 1);
    trace = trace * 10 + 3;
    return project_result;
}
'''

CASES = r'''
int main(void) {
    unsigned cases = 0;
    for (reference = 0; reference < 2; ++reference)
    for (disabled = 0; disabled < 2; ++disabled)
    for (old_apple = 0; old_apple < 2; ++old_apple)
    for (new_apple = 0; new_apple < 2; ++new_apple)
    for (available = 0; available < 2; ++available) {
        probes = 0;
        bool eligible = !reference;
#if TEST_APPLE
        eligible = eligible && !disabled && (old_apple || new_apple);
#endif
        assert(metal_graph_hc_norm_mix_decode_available() ==
               (eligible && available));
        assert(probes == (int)eligible);
        ++cases;
    }
    for (unsigned enabled = 0; enabled < 2; ++enabled)
    for (fused_result = -1; fused_result <= 1; ++fused_result)
    for (norm_result = 0; norm_result < 2; ++norm_result)
    for (project_result = 0; project_result < 2; ++project_result) {
        trace = 0;
        const bool got = metal_graph_hc_norm_mix_or_reference(
                &output, &normalized, &input, &model, &weight,
                16384, 24, enabled);
        if (enabled && fused_result != 0) {
            assert(got == (fused_result > 0));
            assert(trace == 1); /* No replay after submission or success. */
        } else {
            assert(got == (norm_result && project_result));
            assert(trace == (norm_result ? (enabled ? 123u : 23u)
                                        : (enabled ? 12u : 2u)));
        }
        ++cases;
    }
    printf("HC graph %s: PASS (%u admission/fallback/failure cases)\n",
           TEST_APPLE ? "Metal" : "CUDA/ROCm", cases);
}
'''


def main():
    source = (ROOT / "ds4.c").read_text()
    bodies = "\n".join(extract_function(source, signature) for signature in (
        "static bool metal_graph_hc_norm_mix_decode_available(",
        "static bool metal_graph_hc_norm_mix_or_reference(",
    ))
    # Both attention and FFN must use the tested failure-handling helper.
    assert source.count("ok = metal_graph_hc_norm_mix_or_reference(") == 2
    with tempfile.TemporaryDirectory(prefix="ds4-hc-graph-") as directory:
        path = Path(directory)
        unit = path / "graph.c"
        unit.write_text(SHIM + bodies + CASES)
        for apple in (0, 1):
            executable = path / f"graph-{apple}"
            subprocess.run(shlex.split(os.environ.get("CC", "clang")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                f"-DTEST_APPLE={apple}", str(unit), "-o", str(executable),
            ], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
