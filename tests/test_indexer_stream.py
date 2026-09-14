#!/usr/bin/env python3
"""Actual Metal score/index leaf and streaming merge on host, ASan/UBSan.

Only address spaces and barrier scheduling are adapted. Global leaf width and
the full legacy tie order are retained, including a final chunk shorter than K.
No GPU compilation, scoring-kernel math or performance is tested here.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from kernel_source import extract_function
from test_indexer_topk import actual_leaf_source

ROOT = Path(__file__).resolve().parents[1]


def source():
    shader = (ROOT/'metal/indexer_stream.metal').read_text()
    parts = [extract_function(shader, 'struct '+name+' {')+';' for name in (
        'ds4_metal_indexer_stream_pair', 'ds4_metal_args_indexer_stream_leaf',
        'ds4_metal_args_indexer_stream_merge')]
    leaf = extract_function(shader, 'kernel void kernel_indexer_stream_leaf_f32_i32(')
    leaf = re.sub(r'\[\[.*?\]\]', '', leaf)
    leaf = re.sub(r'\bconstant\b', 'const', leaf)
    leaf = re.sub(r'\b(kernel|device|threadgroup)\b', '', leaf)
    leaf = leaf.replace('int32_t *shmem_i32 ', 'int32_t *shmem_i32, float *shmem_f32 ')
    leaf = leaf.replace('    const int col = tpitg.x;', '    (void)tpitg;')
    leaf = leaf.replace('    float *shmem_f32 = ( float *)(shmem_i32 + ntg.x);', '')
    # Preserve production statements while supplying separate typed arrays for
    # the two Metal threadgroup regions; C++ strict aliasing then stays valid.
    assert leaf.count('float *shmem_f32') == 1, 'staging declaration adaptation changed'
    leaf = leaf.replace('    shmem_i32[col] =', '    for (int col=0; col<ntg.x; ++col) {\n    shmem_i32[col] =')
    leaf = leaf.replace('    threadgroup_barrier(mem_flags::mem_threadgroup);', '    }', 1)
    leaf = leaf.replace('            int ixj = col ^ j;',
                        '            for (int col=0; col<ntg.x; ++col) {\n            int ixj = col ^ j;')
    leaf = leaf.replace('            threadgroup_barrier(mem_flags::mem_threadgroup);', '            }')
    leaf = leaf.replace('    if ((uint)col < kept) {',
                        '    for (int col=0; col<ntg.x; ++col) {\n    if ((uint)col < kept) {')
    leaf = leaf[:-1]+'    }\n}'
    assert 'threadgroup_barrier' not in leaf
    merge = extract_function(shader, 'kernel void kernel_indexer_stream_merge(')
    merge = re.sub(r'\[\[.*?\]\]', '', merge)
    merge = re.sub(r'\bconstant\b', 'const', merge)
    merge = re.sub(r'\b(kernel|device)\b', '', merge)
    parts += [actual_leaf_source((ROOT/'metal/argsort.metal').read_text()),
              leaf, 'template<bool final_output>\n'+merge]
    return (ROOT/'tests/test_indexer_stream.cpp').read_text().replace('// PRODUCTION_SOURCE', '\n'.join(parts))


def main():
    compiler = shlex.split(os.environ.get('CXX', 'clang++'))
    with tempfile.TemporaryDirectory(prefix='ds4-indexer-stream-') as directory:
        path, binary = Path(directory)/'test.cpp', Path(directory)/'test'
        path.write_text(source())
        for label, flags in (('strict',['-O2']),('fast',['-O3','-ffast-math','-fno-finite-math-only'])):
            print('Indexer stream host '+label,flush=True)
            subprocess.run(compiler+flags+['-std=c++17','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-I'+str(ROOT),str(path),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__ == '__main__':
    main()
