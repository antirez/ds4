#!/usr/bin/env python3
"""Native IQ2 paired-MoE F16 activation reuse oracle, without a GGUF.

Extract the actual production route map, IQ2 decoder, fused pair and contiguous
F32-to-F16 copy. Compare the original F32 RHS with copy + token-compact F16 RHS
in strict and fast math. --bench measures copy + pair, with the common route map
outside timing; results describe this stage, not end-to-end prefill.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def shader_source():
    moe = (ROOT / 'metal/moe.metal').read_text()
    cpy = (ROOT / 'metal/cpy.metal').read_text()
    runtime = (ROOT / 'ds4_metal.m').read_text()
    fixture = (ROOT / 'tests/test_metal_moe_activation.m').read_text()
    signature = 'static NSUInteger ds4_gpu_cpy_threads('
    assert extract_function(runtime, signature) == extract_function(fixture, signature), \
        'fixture copy launch shape differs from production'
    source = '''#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define QK_NL 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid
'''
    for definition in (
        'static constant uchar ds4_metal_kmask_iq2xs[8] = {',
        'static constant uchar ds4_metal_ksigns_iq2xs[128] = {',
        'static constant ulong ds4_metal_iq2xxs_grid[256] = {',
        'struct block_iq2_xxs {',
        'struct ds4_metal_args_mul_mm_id_map0 {',
        'struct ds4_metal_args_mul_mm_id {',
        'struct ds4_metal_dsv4_moe_swiglu_weight_args {',
    ):
        source += extract_function(moe, definition) + ';\n'
    source += extract_function(moe, 'template <typename type4x4>\nvoid dequantize_iq2_xxs(') + '\n'
    source += extract_function(moe, 'template<short ne20>\nkernel void kernel_mul_mm_id_map0(') + '\n'
    source += 'typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;\n'
    signature = ('template<typename block_q, short nl, void (*dequantize_func)'
                 '(device const block_q *, short, thread half4x4 &), '
                 'bool CULL_TAIL_SIMDGROUPS = false, typename T1 = float, '
                 'typename T1_2x4 = float2x4>\n'
                 'kernel void kernel_mul_mm_id_pair_swiglu_f16_impl(')
    source += extract_function(moe, signature) + '\n'
    for line in moe.splitlines():
        if (line.startswith('typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<block_iq2_xxs') or
                line.startswith('template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16') or
                line.startswith('template [[host_name("kernel_mul_mm_id_map0_ne20_6"')):
            source += line + '\n'
    source += extract_function(cpy, 'kernel void kernel_cpy_contig_f32_f16_4(') + '\n'
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-source', type=Path, help='emit actual shaders without using GPU')
    parser.add_argument('--binary', type=Path, help='use an already compiled harness')
    parser.add_argument('--bench', action='store_true', help='benchmark full-shape paired MoE after correctness')
    args = parser.parse_args()
    source = shader_source()
    if args.emit_source:
        args.emit_source.write_text(source)
        return
    with tempfile.TemporaryDirectory(prefix='ds4-metal-moe-activation-') as directory:
        directory = Path(directory)
        shader = directory / 'moe_activation.metal'
        shader.write_text(source)
        binary = args.binary or directory / 'test_metal_moe_activation'
        if not args.binary:
            subprocess.run(['clang', '-O2', '-fobjc-arc', '-Wall', '-Wextra',
                            '-Werror', '-framework', 'Foundation', '-framework',
                            'Metal', str(ROOT / 'tests/test_metal_moe_activation.m'),
                            '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(shader)] + (['--bench'] if args.bench else []), check=True)


if __name__ == '__main__':
    main()
