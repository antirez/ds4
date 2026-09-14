#!/usr/bin/env python3
"""Validate coordinate packing independently of rocWMMA register ordering.

Default: strict/fast host permutation oracle with ASan/UBSan. --rocm: compile
and run load -> identity MMA -> store diagnostics on gfx1151, strict and fast.
This is not a performance benchmark or proof of full-score numerical parity;
the separate test_gpu_indexer_prepared fixture compares actual score kernels.
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
    key, default = ('HIPCC', 'hipcc') if args.rocm else ('CXX', 'clang++')
    compiler = shlex.split(os.environ.get(key, default))
    if not compiler or not shutil.which(compiler[0]):
        parser.error(f'{key} must name an available compiler')
    if args.rocm:
        common = ['-std=c++17', '-O3', '-x', 'hip', '--offload-arch=gfx1151', '-DTEST_NATIVE']
    else:
        common = ['-std=c++17', '-O2', '-g', '-fsanitize=address,undefined',
                  '-fno-sanitize-recover=all', '-Wall', '-Wextra', '-Werror']
    with tempfile.TemporaryDirectory(prefix='ds4-indexer-registers-') as tmp:
        binary = str(Path(tmp) / 'test')
        for mode in (['-ffp-contract=off'], ['-ffast-math', '-fno-finite-math-only']):
            subprocess.run(compiler + common + mode + ['-I', str(ROOT),
                           str(ROOT / 'tests/test_rocm_indexer_registers.cpp'), '-o', binary], check=True)
            subprocess.run([binary], check=True)


if __name__ == '__main__':
    main()
