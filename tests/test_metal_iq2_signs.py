#!/usr/bin/env python3
"""Exact native IQ2_XXS sign-decode oracle using production Metal wrappers.

Compile the common paired helper with its production popcount macro disabled
and enabled; compare routed ID, six-slot, address and masked-address outputs in
strict and fast math. No GGUF or historical checkout is required.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
MACRO = 'DS4_METAL_IQ2_PAIR_POPCOUNT'


def check_sign_table(moe):
    table = extract_function(
        moe, 'static constant uchar ds4_metal_ksigns_iq2xs[128] = {')
    values = [int(value) for value in re.findall(r'\d+', table.split('{', 1)[1])]
    assert len(values) == 128, 'production sign table length changed'
    for code, value in enumerate(values):
        # Independent integer oracle: the stored eighth sign gives even parity.
        expected = code | ((sum((code >> bit) & 1 for bit in range(7)) & 1) << 7)
        assert value == expected, f'sign table parity differs for code {code}'
    print('PASS: all 128 production IQ2 sign codes have exact even parity.', flush=True)


def shader_source(moe):
    dense = (ROOT / 'metal/dense.metal').read_text()
    helper = extract_function(moe, 'template<int nr0>\nvoid kernel_mul_mv_iq2_xxs_pair_f32_impl(')
    assert MACRO in helper, 'production IQ2 popcount specialization is missing'
    source = '''#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define N_R0_IQ2_XXS 4
constant short FC_mul_mv_nsg [[function_constant(600)]];
'''
    source += extract_function(dense, 'struct ds4_metal_args_mul_mv {') + ';\n'
    for definition in (
        'static constant uchar ds4_metal_kmask_iq2xs[8] = {',
        'static constant uchar ds4_metal_ksigns_iq2xs[128] = {',
        'static constant ulong ds4_metal_iq2xxs_grid[256] = {',
        'struct block_iq2_xxs {',
        'struct ds4_metal_args_mul_mv_id {',
        'struct ds4_metal_dsv4_moe_swiglu_weight_args {',
        'struct ds4_metal_stream_expert_split_args {',
    ):
        source += extract_function(moe, definition) + ';\n'
    source += extract_function(moe, 'static inline bool ds4_tp_owns_expert(') + '\n'
    source += helper + '\n'
    for name in (
        'kernel_mul_mv_id_iq2_xxs_pair_f32',
        'kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32',
        'kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32',
        'kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32',
    ):
        source += extract_function(moe, 'kernel void ' + name + '(') + '\n'
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-source', type=Path, help='emit shader without using GPU')
    parser.add_argument('--host-only', action='store_true', help='check all 128 sign codes only')
    parser.add_argument('--binary', type=Path, help='use an already compiled harness')
    args = parser.parse_args()
    moe = (ROOT / 'metal/moe.metal').read_text()
    check_sign_table(moe)
    if args.host_only:
        return
    source = shader_source(moe)
    if args.emit_source:
        args.emit_source.write_text(source)
        return
    with tempfile.TemporaryDirectory(prefix='ds4-metal-iq2-signs-') as directory:
        directory = Path(directory)
        shader = directory / 'iq2_signs.metal'
        shader.write_text(source)
        binary = args.binary or directory / 'test_metal_iq2_signs'
        if not args.binary:
            subprocess.run(['clang', '-O2', '-fobjc-arc', '-Wall', '-Wextra',
                            '-Werror', '-framework', 'Foundation', '-framework',
                            'Metal', str(ROOT / 'tests/test_metal_iq2_signs.m'),
                            '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(shader)], check=True)


if __name__ == '__main__':
    main()
