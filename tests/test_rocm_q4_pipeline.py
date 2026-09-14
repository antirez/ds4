#!/usr/bin/env python3
"""Benchmark-only Q4 K128 lookahead pipeline: host oracle, --rocm [--bench].

Default checks actual staging helpers with strict/fast ASan/UBSan and verifies
that weight decoding, WMMA order and stores still match the production kernel.
Native execution requires gfx1151; --bench prints paired F32 and copy+F16 timings
plus LDS/VGPR/local-memory/occupancy. The production dispatcher is unchanged.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function
from test_rocm_q4_activation import build_source as activation_source

ROOT = Path(__file__).resolve().parents[1]


def check_math_source():
    original = (ROOT / 'rocm/ds4_rocm_q4.cuh').read_text()
    candidate = (ROOT / 'rocm/ds4_rocm_q4_pipeline.cuh').read_text()
    old = extract_function(original, '__global__ static void\n'
        'rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel(')
    new = extract_function(candidate, '__global__ static void\n'
        'rocm_matmul_q4_K_prefill_wmma_k128_pipeline_rowtile_strided_kernel(')
    # Compare actual source bodies; accidental edits to either baseline or
    # candidate decoding/MMA must make this candidate review fail explicitly.
    signature = 'for (uint32_t qpair_offset = 0u;'
    assert extract_function(old, signature) == extract_function(new, signature), 'weight decode/MMA diverged'
    store = '    for (uint32_t token_tile = 0u; token_tile < 4u; token_tile++) {'
    assert old[old.rindex(store):] == new[new.rindex(store):], 'output stores diverged'
    assert original.count('rocm_matmul_q4_K_prefill_wmma_k128_pipeline_rowtile_strided_kernel<') == 2
    print('PASS identical Q4 weight decoding, MMA order and output stores; benchmark hook only', flush=True)


def native_source():
    prefix = activation_source(True).split('static void native_case(')[0]
    fixture = (ROOT / 'tests/test_rocm_q4_pipeline.cpp').read_text().split('// NATIVE_TEST\n')[1]
    q4 = (ROOT / 'rocm/ds4_rocm_q4.cuh').read_text()
    norm = (ROOT / 'rocm/ds4_rocm_norm_rope.cuh').read_text()
    device = extract_function(norm, 'static bool rocm_q4_qb_gfx1151_wave32_device(')
    for old, new in [('cudaGetDeviceProperties','hipGetDeviceProperties'),
                     ('cudaDeviceProp','hipDeviceProp_t'),
                     ('cudaGetDevice','hipGetDevice'),('cudaSuccess','hipSuccess')]:
        device = device.replace(old,new)
    hook = extract_function(q4, 'extern "C" int ds4_rocm_bench_q4_K_wmma_k128_pipeline_enqueue(')
    return prefix + fixture.replace('// PRODUCTION_PIPELINE_HOOK', device + '\n' + hook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--emit-native', type=Path, help='emit native fixture without requiring HIP')
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm')
    check_math_source()
    if args.emit_native:
        args.emit_native.write_text(native_source())
        return
    with tempfile.TemporaryDirectory(prefix='ds4-q4-pipeline-') as temporary:
        temporary = Path(temporary)
        binary = temporary / 'pipeline'
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC','hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and a gfx1151 GPU are required; no silent fallback')
            source = temporary / 'pipeline.cpp'
            source.write_text(native_source())
            flags = shlex.split(os.environ.get('ROCM_CFLAGS','-O3 --offload-arch=gfx1151'))
            subprocess.run(compiler + flags + ['-x','hip','-std=c++17','-I',str(ROOT),
                str(source),'-o',str(binary)],check=True)
            subprocess.run([str(binary),str(args.device)] + (['--bench'] if args.bench else []),check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX','clang++'))
            for flags in (['-O2'],['-O3','-ffast-math','-fno-finite-math-only']):
                print('Q4 pipeline host ' + ' '.join(flags),flush=True)
                subprocess.run(compiler + flags + ['-std=c++17','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-I',str(ROOT),
                    str(ROOT/'tests/test_rocm_q4_pipeline.cpp'),'-o',str(binary)],check=True)
                subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    main()
