"""Exercise the actual scalar CUDA/HIP Q4 dequantizer through a C host shim.

Checks the packed-layout lowering, all scale/min codes, varied row strides,
half rounding, inactive threads and exact allocations with ASan/UBSan.
Use --emit-gpu PATH to prepare the native legacy-vs-flat oracle, or --gpu to
compile and execute it. Host execution does not establish GPU performance.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from test_cuda_kernel_contracts import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production_body(backend):
    if backend == 'rocm':
        source = (ROOT / 'rocm/ds4_rocm_q4_qb_sidecar.cuh').read_text()
        name = 'rocm_dequant_q4_K_attn_q_b_f16_kernel'
        scale_signature = '__device__ __forceinline__ static void\nrocm_q4_attn_q_b_get_scale_min('
    else:
        source = (ROOT / 'ds4_cuda.cu').read_text()
        name = 'dequant_q4_K_to_f16_kernel'
        scale_signature = '__device__ static void dev_q4_K_get_scale_min('
    signature = '__global__ static void ' + name + '('
    body = extract_function(source[source.rindex(signature):], signature)
    scale = extract_function(source, scale_signature)
    return scale, body.replace(name, 'dequant_q4_K_to_f16_kernel')


def native_program(backend):
    # Reuse the existing guarded native fixture, graph replay and ABBA timing.
    # Its independent scalar_reference retains the original row/column walk;
    # replace only the vector candidate with the actual flat scalar body.
    program = (ROOT / 'tests/test_q4_prefill_dequant.cpp').read_text()
    scale, body = production_body(backend)
    marker = '// Intentionally independent scalar copy of the original backend algorithm.'
    assert program.count(marker) == 1
    declarations = ('using cuda_block_q4_K = BlockQ4;\n'
                    '__device__ static float dev_f16_to_f32(uint16_t bits) {\n'
                    '    return __half2float(__ushort_as_half(bits));\n}\n')
    program = program.replace(marker, declarations + scale + '\n' + body + '\n' + marker)
    candidate = ('ds4_q4_dequant_f16_vec16_kernel<<<unsigned((chunks+255u)/256u),256,0,stream>>>(\n'
                 '            y.ptr<__half>(),x.ptr<BlockQ4>(),blocks);')
    assert program.count(candidate) == 1, 'native fixture candidate changed'
    program = program.replace(candidate,
        'dequant_q4_K_to_f16_kernel<<<unsigned((chunks+255u)/256u),256,0,stream>>>(\n'
        '            y.ptr<__half>(),x.ptr<BlockQ4>(),k,m);')
    # Scalar production dequantization has no GB10/gfx1151 device restriction.
    timing_gate = 'require(!bench || target,"timing requires the production target (GB10/gfx1151)");'
    assert program.count(timing_gate) == 1
    program = program.replace(timing_gate, '(void)target;')
    return ('// Generated actual production flat scalar vs independent legacy scalar.\n' +
            program.replace('#include "../cuda/', '#include "cuda/')
            .replace('for (unsigned skew : {0u,16u,32u,48u})',
                     'for (unsigned skew : {0u,2u,4u,16u,32u,48u})')
            .replace('v ? "vec16" : "scalar16"', 'v ? "flat16" : "reference16"')
            .replace('scalar/vector F16 parity', 'legacy/flat F16 parity')
            .replace('test_q4_prefill_dequant [--bench]', 'test_q4_dequant_flat [--bench]'))

shim = r'''
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef _Float16 __half;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } cuda_block_q4_K;
_Static_assert(sizeof(cuda_block_q4_K) == 144, "GGUF layout");
static struct { unsigned x; } blockIdx, threadIdx, blockDim = {256};
static float dev_f16_to_f32(uint16_t bits) {
    __half value; memcpy(&value, &bits, 2); return (float)value;
}
static __half __float2half_rn(float value) { return (__half)value; }
static __half __float2half(float value) { return (__half)value; }
static __half __ushort_as_half(unsigned short bits) {
    __half value; memcpy(&value, &bits, 2); return value;
}
static float __half2float(__half value) { return (float)value; }
static unsigned random_word(unsigned *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL GPU flat Q4 dequant: %s\n", what); exit(1); }
}
'''

driver = r'''
static unsigned test_case(uint64_t K, uint64_t M, unsigned seed) {
    enum {GUARD = 32};
    const uint64_t count = K * M, nb = count / 256;
    const size_t bytes = (size_t)nb * sizeof(cuda_block_q4_K);
    cuda_block_q4_K *weights = malloc(bytes), *original = malloc(bytes);
    __half *storage = malloc((size_t)(count + 2*GUARD) * sizeof(__half));
    __half *expected = malloc((size_t)(count + 2*GUARD) * sizeof(__half));
    check(weights && original && storage && expected, "allocate");
    const uint16_t special[] = {0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x3c00, 0xbc00, 0x7bff};
    for (uint64_t b = 0; b < nb; ++b) {
        weights[b].d = b < 9 ? special[b] : random_word(&seed) % 0x7c00;
        weights[b].dmin = b < 9 ? special[8 - b] : random_word(&seed) % 0x7c00;
        for (unsigned j = 0; j < 12; ++j) weights[b].scales[j] = random_word(&seed);
        for (unsigned j = 0; j < 128; ++j) weights[b].qs[j] = random_word(&seed);
        // Every scale/min code in both encoding halves, across the large fixture.
        const unsigned group = (b / 64) % 8, sc = b % 64, mn = (b * 17) % 64;
        if (group < 4) {
            weights[b].scales[group] = (weights[b].scales[group] & 0xc0) | sc;
            weights[b].scales[group + 4] = (weights[b].scales[group + 4] & 0xc0) | mn;
        } else {
            weights[b].scales[group + 4] = (sc & 15) | ((mn & 15) << 4);
            weights[b].scales[group - 4] = (weights[b].scales[group - 4] & 63) | ((sc >> 4) << 6);
            weights[b].scales[group] = (weights[b].scales[group] & 63) | ((mn >> 4) << 6);
        }
    }
    memcpy(original, weights, bytes);
    for (uint64_t i = 0; i < count + 2*GUARD; ++i) storage[i] = expected[i] = (__half)1234.0f;
    // Independent scalar decoder follows rows/columns and the GGUF scale map.
    for (uint64_t row = 0; row < M; ++row)
    for (uint64_t col = 0; col < K; ++col) {
        const cuda_block_q4_K *b = weights + row * (K / 256) + col / 256;
        unsigned group = (col % 256) / 32, within = col % 32;
        unsigned sc = group < 4 ? b->scales[group] & 63 :
            (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
        unsigned mn = group < 4 ? b->scales[group + 4] & 63 :
            (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        unsigned packed = b->qs[(group / 2) * 32 + within];
        unsigned q = group % 2 ? packed >> 4 : packed & 15;
        float d = dev_f16_to_f32(b->d), dmin = dev_f16_to_f32(b->dmin);
        expected[GUARD + row * K + col] = __float2half_rn(
            (d * (float)sc) * (float)q - dmin * (float)mn);
    }
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        const uint64_t grid = (count / 16 + 255) / 256 + 3;
        for (uint64_t bid = 0; bid < grid; ++bid) {
            blockIdx.x = repeat & 1 ? (unsigned)(grid - 1 - bid) : (unsigned)bid;
            for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x)
                dequant_q4_K_to_f16_kernel(storage + GUARD, weights, K, M);
        }
        check(!memcmp(storage, expected, (size_t)(count + 2*GUARD) * sizeof(__half)),
              "bitwise half values, row/tile tails and output guards");
        check(!memcmp(weights, original, bytes), "source unchanged");
    }
    free(expected); free(storage); free(original); free(weights);
    return 1;
}
int main(void) {
    const uint64_t widths[] = {256, 512, 768, 1024, 2048, 4096, 8192};
    const uint64_t rows[] = {1, 3, 17, 65};
    unsigned cases = 0;
    for (unsigned k = 0; k < sizeof(widths)/sizeof(widths[0]); ++k)
    for (unsigned m = 0; m < sizeof(rows)/sizeof(rows[0]); ++m)
        cases += test_case(widths[k], rows[m], 701 + 97*k + m);
    // Full 32768x1024 Q-B matrix, still only 18 MiB packed and 64 MiB expanded.
    cases += test_case(1024, 32768, 812);
    printf("PASS host: %u production flat Q4 dequant shapes, 3 schedules each, "
           "all scale/min codes, half boundary values and guards. GPU unverified.\n", cases);
    return 0;
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cuda', 'rocm'), default='cuda')
    parser.add_argument('--emit-gpu', type=Path)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    if args.bench and not args.gpu:
        parser.error('--bench requires --gpu')
    if args.emit_gpu:
        args.emit_gpu.write_text(native_program(args.backend))
        print(f'Wrote {args.backend.upper()} native oracle to {args.emit_gpu}; no GPU execution.')
        return
    if args.gpu:
        env_name, fallback = ('HIPCC', 'hipcc') if args.backend == 'rocm' else ('NVCC', 'nvcc')
        compiler = shlex.split(os.environ.get(env_name, fallback))
        if not compiler or not shutil.which(compiler[0]):
            parser.error(f'{env_name} compiler required; use --emit-gpu to inspect the oracle')
        flags = (shlex.split(os.environ.get('ROCM_CFLAGS',
                                            '-O3 -ffast-math -fno-finite-math-only'))
                 if args.backend == 'rocm' else
                 shlex.split(os.environ.get('NVCCFLAGS', '-O3 --use_fast_math')))
        with tempfile.TemporaryDirectory(prefix='ds4-gpu-q4-flat-') as tmp:
            source, binary = Path(tmp) / 'flat.cu', Path(tmp) / 'flat'
            source.write_text(native_program(args.backend))
            subprocess.run(compiler + flags + ['-std=c++17', '-I', str(ROOT), str(source),
                                               '-o', str(binary)], check=True)
            subprocess.run([str(binary)] + (['--bench'] if args.bench else []), check=True)
        return
    scale, body = production_body(args.backend)
    print(f'Testing production {args.backend.upper()} scalar dequantizer', flush=True)
    with tempfile.TemporaryDirectory(prefix='ds4-cuda-q4-flat-') as tmp:
        cfile = Path(tmp) / 'flat.c'
        cfile.write_text(shim + scale.replace('__device__ ', '').replace('__forceinline__ ', '') + '\n' +
                         body.replace('__global__ ', '') + driver)
        for flags in (['-O2'], ['-O3', '-ffast-math']):
            binary = Path(tmp) / 'flat'
            subprocess.run(shlex.split(os.environ.get('CC', 'cc')) + flags +
                           ['-std=c11', '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            str(cfile), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
