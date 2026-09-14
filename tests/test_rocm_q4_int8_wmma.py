#!/usr/bin/env python3
"""Isolated test-only Q4_K x Q8_K INT8 WMMA feasibility fixture.

Host models fragment ownership and exact block/reduction arithmetic against
the production dot header. --rocm executes the prototype and production-dot
oracle on gfx1151; it is not wired into inference or a speedup benchmark.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ds4-q4-int8-wmma-') as tmp:
        binary = str(Path(tmp) / 'oracle')
        source = str(ROOT / 'tests/test_rocm_q4_int8_wmma.cpp')
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC') or 'hipcc')
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and a gfx1151 GPU are required')
            flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151'))
            subprocess.run(compiler + flags + ['-std=c++17', '-x', 'hip', '-DTEST_NATIVE', '-I', str(ROOT), source, '-o', binary], check=True)
            subprocess.run([binary], check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX', 'clang++'))
            for math in (['-ffp-contract=off'], ['-ffast-math', '-fno-finite-math-only']):
                subprocess.run(compiler + ['-std=c++17', '-O2', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT)] + math + [source, '-o', binary], check=True)
                subprocess.run([binary], check=True)


if __name__ == '__main__':
    main()
