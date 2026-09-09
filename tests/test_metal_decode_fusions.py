#!/usr/bin/env python3
"""Native bitwise oracles for fused Metal decode kernels.

Build a minimal library from the checkout's actual shaders. The reference uses
unchanged ordinary F16/Q8 matvecs followed by an independent state store or the
ordinary HC expansion. No historical checkout or frozen optimized kernel is
required. Run on macOS from any working directory.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def shader_source():
    dense = (ROOT / 'metal/dense.metal').read_text()
    hc = (ROOT / 'metal/dsv4_hc.metal').read_text()
    prefix = '''#include <metal_stdlib>
using namespace metal;
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
struct block_q8_0 { half d; int8_t qs[32]; };
'''
    # Include unchanged ordinary matvecs as independent numerical references.
    prefix += dense[:dense.index('// Q8_0 matvec whose output')]
    begin = dense.index('// DS4 compressor projections always')
    compound = extract_function(dense,
        'kernel void kernel_dsv4_qkv_pair_quad_compressor_store_q8_0(')
    end = dense.index(compound) + len(compound)
    prefix += dense[begin:end] + '\n'
    prefix += extract_function(hc, 'struct ds4_metal_args_dsv4_hc_expand {') + ';\n'
    for name in ('kernel_dsv4_hc_expand4',
                 'kernel_dsv4_shared_down_hc_expand4_q8_0',
                 'kernel_dsv4_q8_hc_expand4_q8_0',
                 'kernel_dsv4_q8_hc_expand4_q8_0_vec_hc'):
        prefix += extract_function(hc, 'kernel void ' + name + '(') + '\n'
    prefix += '''
// Reference has a separate dispatch and reloads the materialized projections.
kernel void reference_compressor_store(
    constant ds4_metal_args_compressor_pair_store &s,
    device const float *kv, device const float *score,
    device const char *ape, device float *state_kv,
    device float *state_score, uint col [[thread_position_in_grid]]) {
    if (col >= s.width || !s.ratio) return;
    uint slot = s.pos % s.ratio;
    uint row = s.ratio == 4 ? 4 + slot : slot;
    float a = s.ape_type == 1 ? float(((device const half *)ape)[slot*s.width+col])
                            : ((device const float *)ape)[slot*s.width+col];
    state_kv[row*s.width+col] = kv[col];
    state_score[row*s.width+col] = score[col] + a;
}
'''
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-source', type=Path)
    parser.add_argument('--binary', type=Path, help='use an already compiled harness')
    args = parser.parse_args()
    source = shader_source()
    if args.emit_source:
        args.emit_source.write_text(source)
        return
    with tempfile.TemporaryDirectory(prefix='ds4-metal-decode-') as directory:
        directory = Path(directory)
        shader = directory / 'decode.metal'
        shader.write_text(source)
        binary = args.binary or directory / 'test_metal_decode_fusions'
        if not args.binary:
            subprocess.run(['clang', '-O2', '-fobjc-arc', '-Wall', '-Wextra',
                            '-Werror', '-framework', 'Foundation', '-framework',
                            'Metal', str(ROOT / 'tests/test_metal_decode_fusions.m'),
                            '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(shader)], check=True)


if __name__ == '__main__':
    main()
