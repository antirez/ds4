#!/usr/bin/env python3
"""Check the ROCm MMQ token-quantization reuse path.

Host mode executes the actual gather and scope policy with ASan/UBSan; it
checks raw record bits and guards, without emulating GPU quantization.
--rocm compares the original gathered quantizer with token quantization plus
this gather on AMD. --bench also times their complete GPU preparation paths,
including the existing output memset. Allocations stay outside GPU timings.
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


def make_source(native):
    header = (ROOT / 'cuda/mmq/ds4_mmq_quant_reuse.cuh').read_text().replace('#pragma once', '')
    wrapper = (ROOT / 'cuda/mmq/ds4_mmq.cu').read_text()
    # Scope remains HIP-only, while the compact producer must report its
    # launch error before its output is consumed. This is not a GPU test.
    include = '#if defined(GGML_USE_HIP)\n#include "ds4_mmq_quant_reuse.cuh"\n#endif'
    assert include in wrapper
    branch = extract_function(wrapper, '        else if (reuse_token_quant) {')
    assert branch.index('err = cudaGetLastError();') < branch.index('ds4_mmq_gather_q8_1_records_kernel<<<')
    assert 'return -3;' in branch and 'ybuf_memset' not in branch
    assert 'type == GGML_TYPE_IQ2_XXS' in wrapper and 'cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151, direct_gateup_q8' in wrapper
    parts = [header]
    if native:
        quant = (ROOT / 'cuda/mmq/quantize.cu').read_text()
        mmq = (ROOT / 'cuda/mmq/mmq.cuh').read_text()
        assert 'int base = cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 ? 64 : hardware_max;' in mmq, 'native timing tail must match gfx1151 default'
        ds_layout = extract_function(mmq, 'static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(')
        iq2_case = ds_layout[ds_layout.index('case GGML_TYPE_IQ2_XXS:'):]
        assert iq2_case[iq2_case.index('return '):].startswith('return MMQ_Q8_1_DS_LAYOUT_D4;'), 'IQ2 native fixture requires production D4 layout'
        layout = extract_function(mmq, 'enum mmq_q8_1_ds_layout {') + ';'
        block = extract_function(mmq, 'struct block_q8_1_mmq {') + ';'
        kernel = extract_function(quant, 'static __global__ void quantize_mmq_q8_1(')
        parts += [layout, block, 'template <mmq_q8_1_ds_layout ds_layout>\n' + kernel]
        # The production HIP vendor header disables *_sync builtins and
        # maps this shuffle to __shfl_xor. Preserve that exact macro.
        vendor = (ROOT / 'cuda/mmq/vendors/hip.h').read_text()
        macro = re.findall(r'^#define __shfl_xor_sync.*$', vendor, re.M)
        assert len(macro) == 1
        parts.insert(0, macro[0])
    else:
        parts = [s.replace('__global__', '') for s in parts]
    fixture = (ROOT / 'tests/test_rocm_mmq_quant_reuse.cpp').read_text()
    return ('#define TEST_NATIVE\n' if native else '') + fixture.replace('// PRODUCTION_SOURCE', '\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm')
    with tempfile.TemporaryDirectory(prefix='ds4-mmq-quant-reuse-') as tmp:
        src, binary = Path(tmp) / 'reuse.cpp', Path(tmp) / 'reuse'
        src.write_text(make_source(args.rocm))
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC', 'hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and an AMD GPU are required; native validation cannot be skipped')
            flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3'))
            subprocess.run(compiler + flags + ['-x', 'hip', '-std=c++17', str(src), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(args.device)] + (['--bench'] if args.bench else []), check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX', 'clang++'))
            for flags in (['-O2'], ['-O3', '-ffast-math', '-fno-finite-math-only']):
                print('MMQ reuse host ' + ' '.join(flags), flush=True)
                subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-fno-strict-aliasing', '-fsanitize=address,undefined', str(src), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
