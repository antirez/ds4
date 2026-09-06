"""Run extracted CUDA load and dispatch code against host contracts.

Requires a C compiler with AddressSanitizer. This checks the production four-byte
load's bounds and the Markov dispatcher's device selection and error cleanup;
it does not compile CUDA, execute a GPU kernel, or model GPU arithmetic.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "ds4_cuda.cu"


def extract_function(source, signature):
    if source.count(signature) != 1:
        raise AssertionError(f"expected one production definition: {signature}")
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise AssertionError(f"unterminated production definition: {signature}")
    return source[start:end]


def production_functions():
    source = SOURCE.read_text()
    device_idx = extract_function(
        source, "static inline int ds4_tensor_device_idx("
    )
    loader = extract_function(
        source,
        "__device__ __forceinline__ static uint32_t ldu32_half_aligned(",
    ).replace("__device__ __forceinline__ ", "", 1)
    dispatch = extract_function(
        source, 'extern "C" int ds4_gpu_dspark_markov_argmax_tensor('
    ).replace('extern "C" ', "", 1)
    dispatch, replacements = re.subn(
        r"dspark_markov_argmax_kernel\s*<<<\s*128\s*,\s*256\s*>>>",
        "dspark_markov_argmax_kernel", dispatch,
    )
    if replacements != 1 or "<<<" in dispatch:
        raise AssertionError("update host shim for changed Markov kernel launch")
    return device_idx + "\n" + loader + "\n" + dispatch


SHIMS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *test_name;
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s: line %d: %s\n", test_name, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

typedef struct { void *ptr; uint64_t bytes; int device_id; } ds4_gpu_tensor;
static struct { int device_id; } g_gpu[2];
static int g_n_gpus = 2;
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 1 };
static unsigned long long out_value;
static float logits[3];
static unsigned char model[512];
static ds4_gpu_tensor out_tensor, logits_tensor;
static int active_device, saved_device, target_device, logical_device;
static int get_calls, set_calls, resolve_calls, memset_calls, kernel_calls;
static int last_error_calls, fail_get, fail_set, fail_resolve, fail_memset, fail_launch;
static char events[32];
static unsigned event_count;

static void record(char event) {
    CHECK(event_count + 1 < sizeof(events));
    events[event_count++] = event;
    events[event_count] = '\0';
}

static cudaError_t cudaGetDevice(int *device) {
    record('G');
    get_calls++;
    if (fail_get) return cudaErrorUnknown;
    *device = active_device;
    return cudaSuccess;
}

static cudaError_t cudaSetDevice(int device) {
    record('S');
    set_calls++;
    CHECK(device == (set_calls == 1 ? target_device : saved_device));
    if (fail_set && set_calls == 1) return cudaErrorUnknown;
    active_device = device;
    return cudaSuccess;
}

static void *cuda_resolve_weight_ptr(const void *map, uint64_t offset,
        uint64_t bytes, int tier, const char *label) {
    resolve_calls++;
    record((char)('0' + resolve_calls));
    CHECK(resolve_calls <= 2);
    CHECK(get_calls == 1 && active_device == target_device);
    CHECK(map == model && tier == logical_device);
    CHECK(offset == (resolve_calls == 1 ? 34u : 128u));
    CHECK(bytes == (resolve_calls == 1 ? 34u : 102u));
    CHECK(strcmp(label, resolve_calls == 1 ? "markov_w1_row" : "markov_w2") == 0);
    if (fail_resolve == resolve_calls) return NULL;
    return model + offset;
}

static cudaError_t cudaMemsetAsync(void *ptr, int value, size_t size) {
    record('M');
    memset_calls++;
    CHECK(active_device == target_device && resolve_calls == 2);
    CHECK(ptr == &out_value && value == 0 && size == sizeof(out_value));
    if (fail_memset) return cudaErrorUnknown;
    memset(ptr, value, size);
    return cudaSuccess;
}

static void dspark_markov_argmax_kernel(unsigned long long *out,
        const float *row, const unsigned char *w1, const unsigned char *w2,
        uint32_t vocab, uint32_t rank_blocks, const void *extra1, const void *extra2) {
    record('K');
    kernel_calls++;
    CHECK(active_device == target_device && memset_calls == 1);
    CHECK(out == &out_value && row == logits && out_value == 0);
    CHECK(w1 == model + 34 && w2 == model + 128);
    CHECK(vocab == 3 && rank_blocks == 1 && !extra1 && !extra2);
}

static cudaError_t cudaGetLastError(void) {
    record('E');
    last_error_calls++;
    CHECK(active_device == target_device && kernel_calls == 1);
    return fail_launch ? cudaErrorUnknown : cudaSuccess;
}

static int cuda_ok(cudaError_t error, const char *operation) {
    CHECK(strcmp(operation, "dspark markov argmax launch") == 0);
    return error == cudaSuccess;
}
"""


DRIVER = r"""
/* Keep the tested load observable at both optimization levels. */
__attribute__((noinline)) static uint32_t invoke_load(const uint8_t *p) {
    return ldu32_half_aligned(p);
}

static void test_loads(void) {
    static const size_t sizes[] = {4, 6, 8, 10, 34, 68, 102};
    const uint32_t endian_probe = 1;
    test_name = "four-byte Q8 load";
    CHECK(*(const uint8_t *)&endian_probe == 1);
    for (size_t n = 0; n < sizeof(sizes) / sizeof(sizes[0]); n++) {
        size_t size = sizes[n];
        uint8_t *data = malloc(size);
        CHECK(data && ((uintptr_t)data & 3u) == 0);
        for (unsigned seed = 0; seed < 256; seed++) {
            for (size_t i = 0; i < size; i++)
                data[i] = (uint8_t)(seed + i * 37u + i * i);
            for (size_t offset = 0; offset + 4 <= size; offset += 2) {
                const uint8_t *p = data + offset;
                uint32_t expected = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
                    ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
                CHECK(invoke_load(p) == expected);
            }
        }
        /* size 34 includes the final word at byte 30 of an unpadded Q8 block. */
        free(data);
    }
}

static void reset_dispatch(const char *name, int logical, int active) {
    test_name = name;
    g_n_gpus = 2;
    g_gpu[0].device_id = 1;
    g_gpu[1].device_id = 0;
    logical_device = logical;
    target_device = logical >= 0 && logical < 2 ? g_gpu[logical].device_id : -1;
    active_device = saved_device = active;
    get_calls = set_calls = resolve_calls = memset_calls = kernel_calls = 0;
    last_error_calls = fail_get = fail_set = fail_resolve = fail_memset = fail_launch = 0;
    events[0] = '\0';
    event_count = 0;
    out_value = 123;
    out_tensor = (ds4_gpu_tensor){&out_value, sizeof(out_value), logical};
    logits_tensor = (ds4_gpu_tensor){logits, sizeof(logits), logical};
}

static int dispatch(void) {
    return ds4_gpu_dspark_markov_argmax_tensor(
        &out_tensor, &logits_tensor, model, sizeof(model), 0, 128, 1, 3, 32);
}

static void check_restored(void) {
    CHECK(active_device == saved_device);
    CHECK(set_calls == (saved_device != target_device ? 2 : 0));
}

static void check_rejected(void) {
    CHECK(dispatch() == 0);
    CHECK(event_count == 0 && active_device == saved_device && out_value == 123);
}

static void test_dispatch(void) {
    for (int logical = 0; logical < 2; logical++) {
        reset_dispatch("logical tier maps to physical device", logical, logical);
        CHECK(dispatch() == 1);
        CHECK(strcmp(events, "GS12MKES") == 0);
        check_restored();

        reset_dispatch("target device already active", logical, 1 - logical);
        CHECK(dispatch() == 1);
        CHECK(strcmp(events, "G12MKE") == 0);
        check_restored();
    }

    reset_dispatch("output tier mismatch", 0, 0);
    out_tensor.device_id = 1;
    check_rejected();

    reset_dispatch("untagged tensors use legacy logical tier zero", 0, 0);
    out_tensor.device_id = logits_tensor.device_id = -1;
    CHECK(dispatch() == 1);
    CHECK(strcmp(events, "GS12MKES") == 0);
    check_restored();

    reset_dispatch("null tensor rejects before device resolution", 0, 0);
    CHECK(ds4_gpu_dspark_markov_argmax_tensor(
        NULL, &logits_tensor, model, sizeof(model), 0, 128, 1, 3, 32) == 0);
    CHECK(ds4_gpu_dspark_markov_argmax_tensor(
        &out_tensor, NULL, model, sizeof(model), 0, 128, 1, 3, 32) == 0);
    CHECK(event_count == 0 && active_device == saved_device && out_value == 123);

    reset_dispatch("logical tier beyond GPU count", 2, 0);
    check_rejected();
    reset_dispatch("no initialized GPUs", 0, 0);
    g_n_gpus = 0;
    check_rejected();

    for (int failing_weight = 1; failing_weight <= 2; failing_weight++) {
        reset_dispatch("weight resolution failure restores device", 0, 0);
        fail_resolve = failing_weight;
        CHECK(dispatch() == 0);
        CHECK(resolve_calls >= failing_weight && resolve_calls <= 2);
        CHECK(memset_calls == 0 && kernel_calls == 0 && last_error_calls == 0);
        CHECK(events[event_count - 1] == 'S');
        check_restored();
    }

    reset_dispatch("memset failure restores device", 0, 0);
    fail_memset = 1;
    CHECK(dispatch() == 0);
    CHECK(strcmp(events, "GS12MS") == 0);
    CHECK(kernel_calls == 0 && last_error_calls == 0);
    check_restored();

    reset_dispatch("launch failure restores device", 0, 0);
    fail_launch = 1;
    CHECK(dispatch() == 0);
    CHECK(strcmp(events, "GS12MKES") == 0);
    check_restored();

    reset_dispatch("device query failure stops dispatch", 0, 0);
    fail_get = 1;
    CHECK(dispatch() == 0);
    CHECK(strcmp(events, "G") == 0 && active_device == saved_device);

    reset_dispatch("device switch failure stops dispatch", 0, 0);
    fail_set = 1;
    CHECK(dispatch() == 0);
    CHECK(strcmp(events, "GS") == 0 && active_device == saved_device);
}

int main(void) {
    test_loads();
    test_dispatch();
    puts("CUDA host contracts: load parity/bounds and Markov device dispatch passed");
    return 0;
}
"""


class CudaKernelContractTests(unittest.TestCase):
    def test_production_host_contracts(self):
        with tempfile.TemporaryDirectory(prefix="ds4-cuda-contracts-") as tmp:
            source = Path(tmp) / "contracts.c"
            source.write_text(SHIMS + production_functions() + DRIVER)
            compiler = shlex.split(os.environ.get("CC", "cc"))
            for optimization in ("-O0", "-O2"):
                with self.subTest(optimization=optimization):
                    binary = Path(tmp) / ("contracts" + optimization)
                    subprocess.run(compiler + [
                        "-std=c11", optimization, "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address", "-fno-omit-frame-pointer", "-g",
                        str(source), "-o", str(binary),
                    ], check=True)
                    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
