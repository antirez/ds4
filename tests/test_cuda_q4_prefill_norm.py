"""Compile actual CUDA prefill RMS bodies as C with a staged reduction shim.

The shim collects the production per-lane sums, compares the actual reduction
helper with the independent legacy tree, then resumes the production epilogue.
It checks arithmetic and ownership, not CUDA scheduling, barriers or FTZ.
Use --emit-cuda PATH to prepare the same kernels and a native bitwise oracle,
or --cuda to compile and execute that oracle with NVCC and a CUDA device.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from test_cuda_kernel_contracts import extract_function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4_cuda.cu").read_text()
NAMES = ["dsv4_qkv_rms_norm_rows_kernel",
         "dsv4_qkv_rms_norm_rows_kv_rope_kernel",
         "head_rms_norm_rope_tail_kernel"]
BODIES = [extract_function(SOURCE, "__global__ static void " + n + "(")
          for n in NAMES]
RAMP = extract_function(SOURCE, "__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {")


def host_body(body, index):
    start = body.index("    __shared__ float partial[256];")
    end = body.index("    const float scale", start)
    reduction = body[start:end]
    condition = "Q4_PREFILL_REDUCE" if index == 2 else "rows > 8u"
    # Fail if the kernel no longer has the reviewed two-path reduction.
    assert f"if ({condition})" in reduction
    assert reduction.count("ds4_q4_prefill_reduce_256(sum, partial)") == 1
    assert reduction.count("partial[threadIdx.x] += partial[threadIdx.x + stride]") == 1
    body = body[:start] + f"    const float total = reduce_stage(sum, {condition});\n" + body[end:]
    return body.replace("__global__ ", "")


COMMON = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static void check(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL CUDA prefill RMS: %s\n", message); exit(1); }
}
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float add(float a, float b) { volatile float r = a + b; return r; }
static unsigned random_word(unsigned *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
'''

HOST = r'''
#include <setjmp.h>
#include "cuda/ds4_q4_prefill_reduce.h"
static struct { unsigned x, y; } threadIdx, blockIdx, blockDim = {256, 1};
static int Q4_PREFILL_REDUCE, collecting, reference, saw_optimized;
static float leaves[256], total;
static jmp_buf collected;
static float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static float reduce_stage(float sum, int optimized) {
    saw_optimized = optimized;
    if (collecting) { leaves[threadIdx.x] = sum; longjmp(collected, 1); }
    return total;
}
static void finish_reduce(void) {
    float baseline[256], warp[32];
    memcpy(baseline, leaves, sizeof(leaves));
    for (unsigned stride = 128; stride; stride >>= 1)
        for (unsigned lane = 0; lane < stride; ++lane)
            baseline[lane] = add(baseline[lane], baseline[lane + stride]);
    for (unsigned lane = 0; lane < 32; ++lane)
        warp[lane] = ds4_q4_prefill_reduce_lane(leaves, lane);
    for (unsigned stride = 16; stride; stride >>= 1)
        for (unsigned lane = 0; lane < stride; ++lane)
            warp[lane] = ds4_q4_prefill_add(warp[lane], warp[lane + stride]);
    check(bits(baseline[0]) == bits(warp[0]), "actual partials / exact tree");
    total = reference || !saw_optimized ? baseline[0] : warp[0];
}
'''

# One CTA is emulated at the final token. Other rows and both end guards must
# remain untouched. In-place normalization shares its input/output pointer.
CALLS = r'''
static unsigned mode, rows, q_n, kv_n, heads, width, rotary, seed_case;
static int in_place;
static float *q, *kv, *qw, *kw, *qo, *ko;
static void run_lane(void) {
    const float ext = seed_case & 1 ? 1.0f : 0.0f;
    const float eps = seed_case & 2 ? 1e-6f : 1e-5f;
    if (mode == 0) dsv4_qkv_rms_norm_rows_kernel(qo, q, qw, q_n,
        ko, kv, kw, kv_n, rows, eps);
    else if (mode == 1) dsv4_qkv_rms_norm_rows_kv_rope_kernel(
        qo, q, qw, q_n, ko, kv, kw, kv_n, rows, heads, width, rotary,
        7919, 131072, seed_case & 4, 10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
    else head_rms_norm_rope_tail_kernel(qo, rows, heads, width, rotary,
        7919, 131072, seed_case & 4, 10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
}
static void run_cta(unsigned which) {
    blockIdx.y = which;
    blockIdx.x = mode == 2 ? rows * heads - 1 : rows - 1;
    collecting = 1;
    for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x)
        if (!setjmp(collected)) run_lane();
    check(saw_optimized == (rows > 8), "decode/prefill boundary");
    finish_reduce();
    collecting = 0;
    for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x) run_lane();
}
static void run_case(unsigned case_id) {
    enum {GUARD = 16};
    const float sentinel = 123456.0f;
    seed_case = case_id;
    const size_t qc = (size_t)rows * q_n, kc = (size_t)rows * kv_n;
    float *initial_q = malloc((qc + 2*GUARD) * 4), *initial_k = malloc((kc + 2*GUARD) * 4);
    float *store_q = malloc((qc + 2*GUARD) * 4), *store_k = malloc((kc + 2*GUARD) * 4);
    float *expect_q = malloc((qc + 2*GUARD) * 4), *expect_k = malloc((kc + 2*GUARD) * 4);
    float *input_q = malloc(qc * 4), *input_k = malloc(kc * 4);
    qw = malloc(q_n * 4); kw = malloc(kv_n * 4);
    check(initial_q && initial_k && store_q && store_k && expect_q && expect_k &&
          input_q && input_k && qw && kw, "allocation");
    unsigned seed = case_id + 97;
    for (size_t i = 0; i < qc; ++i) input_q[i] =
        ldexpf((int)(random_word(&seed) % 1025) - 512, (int)(i % 9) - 7);
    for (size_t i = 0; i < kc; ++i) input_k[i] =
        ldexpf((int)(random_word(&seed) % 1025) - 512, (int)(i % 11) - 8);
    if (case_id % 6 == 0 || case_id % 6 == 1) {
        for (size_t i = 0; i < qc; ++i)
            input_q[i] = case_id % 6 == 0 ? 0.0f : ldexpf((int)(i % 127) - 63, -140);
        for (size_t i = 0; i < kc; ++i)
            input_k[i] = case_id % 6 == 0 ? -0.0f : ldexpf((int)(i % 113) - 56, -140);
    }
    for (unsigned i = 0; i < q_n; ++i) qw[i] = ((int)(i % 47) - 23) / 32.0f;
    for (unsigned i = 0; i < kv_n; ++i) kw[i] = ((int)(i % 61) - 30) / 32.0f;
    for (size_t i = 0; i < qc + 2*GUARD; ++i) initial_q[i] = sentinel;
    for (size_t i = 0; i < kc + 2*GUARD; ++i) initial_k[i] = sentinel;
    if (in_place || mode == 2) {
        memcpy(initial_q + GUARD, input_q, qc * 4);
        memcpy(initial_k + GUARD, input_k, kc * 4);
    }
    for (reference = 1; reference >= 0; --reference) {
        memcpy(store_q, initial_q, (qc + 2*GUARD) * 4);
        memcpy(store_k, initial_k, (kc + 2*GUARD) * 4);
        qo = store_q + GUARD; ko = store_k + GUARD;
        q = in_place ? qo : input_q; kv = in_place ? ko : input_k;
        Q4_PREFILL_REDUCE = rows > 8;
        run_cta(0);
        if (mode != 2) run_cta(1);
        if (reference) {
            memcpy(expect_q, store_q, (qc + 2*GUARD) * 4);
            memcpy(expect_k, store_k, (kc + 2*GUARD) * 4);
        } else {
            check(!memcmp(expect_q, store_q, (qc + 2*GUARD) * 4), "Q/head bitwise epilogue");
            check(!memcmp(expect_k, store_k, (kc + 2*GUARD) * 4), "KV bitwise epilogue");
        }
        const size_t qs = mode == 2 ? qc - width : qc - q_n;
        const size_t ks = kc - kv_n;
        for (size_t i = 0; i < qc + 2*GUARD; ++i)
            if (i < GUARD + qs || i >= GUARD + qc)
                check(bits(store_q[i]) == bits(initial_q[i]), "Q guard / other rows");
        for (size_t i = 0; i < kc + 2*GUARD; ++i)
            if (mode == 2 || i < GUARD + ks || i >= GUARD + kc)
                check(bits(store_k[i]) == bits(initial_k[i]), "KV guard / other rows");
    }
    free(kw); free(qw); free(input_k); free(input_q); free(expect_k); free(expect_q);
    free(store_k); free(store_q); free(initial_k); free(initial_q);
}
int main(void) {
    const unsigned counts[] = {1, 8, 9, 32};
    const unsigned widths[] = {32, 34, 254, 258, 512, 514, 1024, 1536, 4096};
    unsigned cases = 0;
    for (mode = 0; mode < 3; ++mode)
    for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); ++wi)
    for (unsigned ni = 0; ni < sizeof(counts)/sizeof(counts[0]); ++ni)
    for (unsigned r = 0; r < 3; ++r)
    for (in_place = 0; in_place < 2; ++in_place) {
        width = widths[wi]; rows = counts[ni]; heads = wi & 1 ? 2 : 1;
        rotary = r == 0 ? 0 : r == 1 ? (width < 64 ? width : 64) : width;
        kv_n = width * heads; q_n = mode == 2 ? kv_n : 1024;
        run_case(cases++);
    }
    printf("PASS host: %u production RMS/RoPE cases, exact reduction, Q/KV arms, "
           "in-place, widths/tails, decode threshold and guards. CUDA unverified.\n", cases);
    return 0;
}
'''

NATIVE = r'''
#define CUDA_CHECK(expr) check((expr) == cudaSuccess, #expr)
static void native_case(unsigned mode, unsigned rows, unsigned heads,
                        unsigned width, unsigned rotary, int in_place, unsigned seed) {
    enum {GUARD = 16};
    const unsigned qn = mode == 2 ? heads * width : 1024, kn = heads * width;
    const size_t qc = (size_t)rows * qn, kc = (size_t)rows * kn;
    const size_t qb = (qc + 2*GUARD) * 4, kb = (kc + 2*GUARD) * 4;
    const float sentinel = 123456.0f;
    float *hq = (float *)malloc(qb), *hk = (float *)malloc(kb);
    float *expected_q = (float *)malloc(qb), *expected_k = (float *)malloc(kb);
    float *got_q = (float *)malloc(qb), *got_k = (float *)malloc(kb);
    float *hwq = (float *)malloc(qn * 4), *hwk = (float *)malloc(kn * 4);
    check(hq && hk && expected_q && expected_k && got_q && got_k && hwq && hwk, "host allocation");
    const unsigned case_id = seed;
    for (size_t i = 0; i < qc + 2*GUARD; ++i) hq[i] = sentinel;
    for (size_t i = 0; i < kc + 2*GUARD; ++i) hk[i] = sentinel;
    for (size_t i = 0; i < qc; ++i) hq[GUARD+i] = case_id % 6 == 0 ? 0.0f :
        ldexpf((int)(random_word(&seed) % 1025) - 512, case_id % 6 == 1 ? -140 : (int)(i % 9) - 7);
    for (size_t i = 0; i < kc; ++i) hk[GUARD+i] = case_id % 6 == 0 ? -0.0f :
        ldexpf((int)(random_word(&seed) % 1025) - 512, case_id % 6 == 1 ? -140 : (int)(i % 11) - 8);
    for (unsigned i = 0; i < qn; ++i) hwq[i] = ((int)(i % 47) - 23) / 32.0f;
    for (unsigned i = 0; i < kn; ++i) hwk[i] = ((int)(i % 61) - 30) / 32.0f;
    float *dq, *dk, *wq, *wk, *oq, *ok;
    CUDA_CHECK(cudaMalloc((void **)&dq, qb)); CUDA_CHECK(cudaMalloc((void **)&dk, kb));
    CUDA_CHECK(cudaMalloc((void **)&oq, qb)); CUDA_CHECK(cudaMalloc((void **)&ok, kb));
    CUDA_CHECK(cudaMalloc((void **)&wq, qn * 4)); CUDA_CHECK(cudaMalloc((void **)&wk, kn * 4));
    CUDA_CHECK(cudaMemcpy(wq, hwq, qn*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wk, hwk, kn*4, cudaMemcpyHostToDevice));
    const float ext = case_id & 1 ? 1.0f : 0.0f, eps = case_id & 2 ? 1e-6f : 1e-5f;
    for (unsigned candidate = 0; candidate < 2; ++candidate) {
        CUDA_CHECK(cudaMemcpy(dq, hq, qb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dk, hk, kb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(oq, hq, qb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ok, hk, kb, cudaMemcpyHostToDevice));
        const float *iq = in_place ? oq + GUARD : dq + GUARD;
        const float *ik = in_place ? ok + GUARD : dk + GUARD;
        const dim3 grid(rows, 2, 1);
        if (mode == 0) {
            if (candidate) dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                oq+GUARD, iq, wq, qn, ok+GUARD, ik, wk, kn, rows, eps);
            else dsv4_qkv_rms_norm_rows_kernel_reference<<<grid, 256>>>(
                oq+GUARD, iq, wq, qn, ok+GUARD, ik, wk, kn, rows, eps);
        } else if (mode == 1) {
            if (candidate) dsv4_qkv_rms_norm_rows_kv_rope_kernel<<<grid, 256>>>(
                oq+GUARD, iq, wq, qn, ok+GUARD, ik, wk, kn, rows, heads, width, rotary,
                7919, 131072, case_id & 4, 10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
            else dsv4_qkv_rms_norm_rows_kv_rope_kernel_reference<<<grid, 256>>>(
                oq+GUARD, iq, wq, qn, ok+GUARD, ik, wk, kn, rows, heads, width, rotary,
                7919, 131072, case_id & 4, 10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
        } else {
            if (candidate && rows > 8) head_rms_norm_rope_tail_kernel<true><<<rows*heads, 256>>>(
                oq+GUARD, rows, heads, width, rotary, 7919, 131072, case_id & 4,
                10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
            else head_rms_norm_rope_tail_kernel<false><<<rows*heads, 256>>>(
                oq+GUARD, rows, heads, width, rotary, 7919, 131072, case_id & 4,
                10000.0f, 0.125f, ext, 1.0f, 32.0f, 1.0f, eps);
        }
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got_q, dq, qb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_k, dk, kb, cudaMemcpyDeviceToHost));
        check(!memcmp(got_q, hq, qb) && !memcmp(got_k, hk, kb), "input unchanged");
        CUDA_CHECK(cudaMemcpy(got_q, oq, qb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_k, ok, kb, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < GUARD; ++i) {
            check(got_q[i] == sentinel && got_q[GUARD+qc+i] == sentinel, "Q native guards");
            check(got_k[i] == sentinel && got_k[GUARD+kc+i] == sentinel, "KV native guards");
        }
        if (candidate) {
            check(!memcmp(got_q, expected_q, qb), "native Q/RoPE bitwise");
            check(!memcmp(got_k, expected_k, kb), "native KV/RoPE bitwise");
        } else { memcpy(expected_q, got_q, qb); memcpy(expected_k, got_k, kb); }
    }
    CUDA_CHECK(cudaFree(dq)); CUDA_CHECK(cudaFree(dk)); CUDA_CHECK(cudaFree(oq));
    CUDA_CHECK(cudaFree(ok)); CUDA_CHECK(cudaFree(wq)); CUDA_CHECK(cudaFree(wk));
    free(hq); free(hk); free(expected_q); free(expected_k); free(got_q); free(got_k); free(hwq); free(hwk);
}
int main(void) {
    const unsigned counts[] = {1, 8, 9, 32};
    const unsigned widths[] = {32, 34, 254, 258, 512, 514, 1024, 1536, 4096};
    unsigned cases = 0;
    for (unsigned mode = 0; mode < 3; ++mode)
    for (unsigned wi = 0; wi < sizeof(widths)/sizeof(widths[0]); ++wi)
    for (unsigned ni = 0; ni < sizeof(counts)/sizeof(counts[0]); ++ni)
    for (unsigned r = 0; r < 3; ++r)
    for (int in_place = 0; in_place < 2; ++in_place) {
        const unsigned width = widths[wi], rotary = r == 0 ? 0 : r == 1 ? (width < 64 ? width : 64) : width;
        native_case(mode, counts[ni], wi & 1 ? 2 : 1, width, rotary, in_place, cases++);
    }
    printf("PASS CUDA: %u production RMS/RoPE cases, native legacy-tree bitwise parity and guards.\n", cases);
    return 0;
}
'''


def native_source():
    emitted = '#include <cuda_runtime.h>\n' + COMMON
    emitted += '#include "cuda/ds4_q4_prefill_reduce.h"\n' + RAMP + '\n'
    emitted += BODIES[0] + '\n' + BODIES[1] + '\n'
    emitted += 'template<bool Q4_PREFILL_REDUCE>\n' + BODIES[2] + '\n'
    for name, body in zip(NAMES[:2], BODIES[:2]):
        emitted += body.replace(name, name + '_reference').replace(
            'if (rows > 8u)', 'if (false)') + '\n'
    return emitted + NATIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cuda", type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.emit_cuda:
        args.emit_cuda.write_text(native_source())
        print(f"Wrote native candidate/reference oracle to {args.emit_cuda}; no CUDA execution.")
        return
    if args.cuda:
        compiler = shlex.split(os.environ.get('NVCC', 'nvcc'))
        if not compiler or not shutil.which(compiler[0]):
            parser.error('NVCC compiler required; use --emit-cuda to inspect the native oracle')
        flags = shlex.split(os.environ.get('NVCCFLAGS', '-O3 --use_fast_math'))
        with tempfile.TemporaryDirectory(prefix='ds4-cuda-q4-norm-native-') as tmp:
            source, binary = Path(tmp) / 'norm.cu', Path(tmp) / 'norm'
            source.write_text(native_source())
            subprocess.run(compiler + flags +
                           ['-std=c++17', '-I', str(ROOT), str(source),
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
        return
    program = COMMON + HOST + RAMP.replace('__device__ ', '') + '\n'
    program += '\n'.join(host_body(b, i) for i, b in enumerate(BODIES)) + CALLS
    with tempfile.TemporaryDirectory(prefix='ds4-cuda-q4-norm-') as tmp:
        source = Path(tmp) / 'norm.c'
        source.write_text(program)
        for flags in (['-O2'], ['-O3', '-ffast-math']):
            binary = Path(tmp) / 'norm'
            subprocess.run(shlex.split(os.environ.get('CC', 'cc')) + flags +
                           ['-std=c11', '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            '-I', str(ROOT), str(source), '-lm', '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
