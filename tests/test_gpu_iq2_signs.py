#!/usr/bin/env python3
"""Test the actual CUDA/ROCm IQ2 helpers against their production sign table.

Default: strict/fast host builds with ASan+UBSan and modeled GPU intrinsics.
--cuda/--rocm: compile the same extracted helpers and run the oracle on-device.
Host results establish neither GPU compilation nor a performance improvement.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = (
    'dev_f16_to_f32', 'dev_unpack_iq2_signs', 'dev_iq2_dp4a_8',
    'dev_dot_iq2_pair_16', 'dev_iq2_i8x8_lut',
    'dev_dot_iq2_xxs_q8_K_block_lut', 'dev_dot_iq2_xxs_q8_K_block',
    'dev_dot_iq2_xxs_q8_K_block8_deq_lut',
    'dev_dot_iq2_xxs_q8_K_block4', 'dev_dot_iq2_xxs_q8_K_block8',
)


def function(source, name):
    matches = re.findall(r'^.*\b' + name + r'\(', source, re.M)
    definitions = [s for s in matches if '__device__' in s]
    assert len(definitions) == 1, f'expected one definition of {name}'
    return extract_function(source, definitions[0])


def make_source(backend, native):
    source = (ROOT / ('ds4_cuda.cu' if backend == 'cuda' else 'rocm/ds4_rocm_moe.cuh')).read_text()
    types = (ROOT / ('ds4_cuda.cu' if backend == 'cuda' else 'ds4_rocm.cu')).read_text()
    parts = ['#define CUDA_QK_K 256\n#define DS4_CUDA_UNUSED\n#define DS4_ROCM_UNUSED\n']
    for name in ('cuda_block_iq2_xxs', 'cuda_block_q8_K'):
        matches = re.findall(r'typedef struct \{[^{}]*\} ' + name + ';', types)
        assert len(matches) == 1, f'production block ABI: {name}'
        parts += matches
    tables = (ROOT / 'ds4_iq2_tables_cuda.inc').read_text()
    for name in ('cuda_ksigns_iq2xs[128]', 'cuda_iq2xxs_grid[256]'):
        start = tables.rfind('\n', 0, tables.index(name)) + 1
        parts += [extract_function(tables, tables[start:tables.index('{', start) + 1]) + ';']
    names = FUNCTIONS + (('iq2_xxs_dequant_256_direct',) if backend == 'rocm' else ())
    helpers = '\n'.join(function(source, name) for name in names)
    unpack = function(source, 'dev_unpack_iq2_signs')
    assert '__popc(v)' in unpack, 'production raw sign parity helper missing'
    assert 'cuda_ksigns_iq2xs[' not in helpers, 'redundant production sign lookup remains'
    # Preserve all production floating-point arithmetic and replace ONLY the
    # sign expansion in the reference. No historical checkout is needed.
    table_unpack = unpack[:unpack.index('{')] + '{ return uint32_t(cuda_ksigns_iq2xs[v]) * 0x01010101u; }'
    parts += ['namespace production {\n' + helpers + '\n}',
              'namespace reference {\n' + helpers.replace(unpack, table_unpack) + '\n}']
    if backend == 'rocm':
        parts.insert(0, '#define TEST_ROCM\n')
        compat = (ROOT / 'ds4_rocm.h').read_text()
        intrinsics = '\n'.join(function(compat, name) for name in ('__vcmpne4', '__vsub4', '__dp4a'))
        # On host the actual compatibility helpers select their scalar branch;
        # on HIP this compiles the production amd_mixed_dot instruction path.
        parts.insert(0, intrinsics)
    text = '\n'.join(parts)
    if not native:
        text = text.replace('__device__', '').replace('__constant__', 'const').replace('__forceinline__', 'inline')
    fixture = (ROOT / 'tests/test_gpu_iq2_signs.cpp').read_text()
    return ('#define TEST_ROCM_HOST\n' if backend == 'rocm' and not native else '') + fixture.replace('// PRODUCTION_SOURCE', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cuda', action='store_true')
    mode.add_argument('--rocm', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    native = args.cuda or args.rocm
    backends = ('cuda',) if args.cuda else ('rocm',) if args.rocm else ('cuda', 'rocm')
    with tempfile.TemporaryDirectory(prefix='ds4-iq2-signs-') as tmp:
        for backend in backends:
            source = Path(tmp) / ('iq2.cu' if native else 'iq2.cpp')
            source.write_text(make_source(backend, native))
            binary = Path(tmp) / 'iq2'
            if native:
                env = 'NVCC' if args.cuda else 'HIPCC'
                compiler = shlex.split(os.environ.get(env, 'nvcc' if args.cuda else 'hipcc'))
                if not compiler or not shutil.which(compiler[0]):
                    parser.error(f'{env} and a visible device are required; native validation cannot be skipped')
                flags = shlex.split(os.environ.get('NVCCFLAGS' if args.cuda else 'ROCM_CFLAGS', '-O3'))
                language = 'cu' if args.cuda else 'hip'
                subprocess.run(compiler + flags + ['-x', language, '-std=c++17', str(source), '-o', str(binary)], check=True)
                subprocess.run([str(binary), str(args.device)], check=True)
            else:
                compiler = shlex.split(os.environ.get('CXX', 'clang++'))
                for flags in (['-O2'], ['-O3', '-ffast-math', '-fno-finite-math-only']):
                    print(f'{backend} host {" ".join(flags)}', flush=True)
                    subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-Wno-unknown-pragmas', '-fno-strict-aliasing', '-fsanitize=address,undefined', str(source), '-o', str(binary)], check=True)
                    subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
