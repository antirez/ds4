#!/usr/bin/env python3
"""Compare old and current fused Metal F16 decode shaders on identical hot inputs.

This is a direct shader microbenchmark, not a model/server throughput test. It
bypasses production dispatch gates, SSD I/O, and model scheduling. The baseline
must already exist in the local Git object database; this tool never fetches.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from kernel_source import extract_function
from test_metal_decode_fusions import shader_source

DEFAULT_BASELINE = '6289c516273979173abbc062209a81dd3706b804'
SOURCE_FILES = ('metal/dense.metal', 'metal/dsv4_hc.metal')


def git(*arguments):
    return subprocess.run(['git', '-C', str(ROOT), *arguments], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()


def positive(value):
    parsed = int(value)
    if not 1 <= parsed <= 100000:
        raise argparse.ArgumentTypeError('expected an integer from 1 to 100000')
    return parsed


def nonnegative(value):
    parsed = int(value)
    if not 0 <= parsed <= 100000:
        raise argparse.ArgumentTypeError('expected an integer from 0 to 100000')
    return parsed


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reduction_planes(dense):
    """Recognize the two supported scratch contracts; fail on an unknown layout."""
    entry = extract_function(dense,
        'kernel void kernel_mul_mv_f16_f32_pair_compressor_store_4(')
    if 'kernel_mul_mv_f16_f32_pair_compressor_store_4_impl<' in entry:
        helper = extract_function(dense,
            'void kernel_mul_mv_f16_f32_pair_compressor_store_4_impl(')
        if 'shared + NW * (NR0 + row)' in helper:
            return 2
    elif ('kernel_mul_mv_f16_f32_pair_4_disp<' in entry and
          'threadgroup_barrier(mem_flags::mem_device)' in entry):
        return 1
    raise ValueError('unsupported F16 compressor scratch layout; update the benchmark ABI')


def paired_summaries(csv_path):
    grouped = {}
    with csv_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['phase'] != 'sample':
                continue
            key = row['math'], row['case']
            grouped.setdefault(key, {}).setdefault(int(row['round']), {})[row['variant']] = float(row['gpu_us'])
    summaries = []
    for (math, case), rounds in grouped.items():
        if any(set(pair) != {'baseline', 'current'} for pair in rounds.values()):
            raise ValueError('incomplete paired measurements in the CSV')
        ratios = [pair['current'] / pair['baseline'] for pair in rounds.values()]
        summaries.append({
            'math': math, 'case': case, 'pairs': len(ratios),
            'baseline_median_us': statistics.median(pair['baseline'] for pair in rounds.values()),
            'current_median_us': statistics.median(pair['current'] for pair in rounds.values()),
            'median_paired_current_over_baseline': statistics.median(ratios),
            'min_paired_current_over_baseline': min(ratios),
            'max_paired_current_over_baseline': max(ratios),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default=DEFAULT_BASELINE,
                        help='local Git commit/ref used for the actual baseline shaders')
    parser.add_argument('--samples', type=positive, default=8,
                        help='measured paired rounds per case and math mode (default: 8)')
    parser.add_argument('--repeats', type=positive, default=512,
                        help='fused dispatches in each timed GPU command (default: 512)')
    parser.add_argument('--warmup', type=nonnegative, default=2,
                        help='untimed paired warmup rounds per case (default: 2)')
    parser.add_argument('--math', choices=('fast', 'strict', 'both'), default='fast')
    parser.add_argument('--output', type=Path, required=True,
                        help='new output prefix; writes PREFIX.csv and PREFIX.json')
    args = parser.parse_args()
    csv_path = Path(str(args.output) + '.csv').resolve()
    json_path = Path(str(args.output) + '.json').resolve()
    if csv_path.exists() or json_path.exists():
        parser.error('output already exists; choose a new --output prefix')
    try:
        baseline_sha = git('rev-parse', '--verify', '--end-of-options', args.baseline + '^{commit}').strip()
    except subprocess.CalledProcessError:
        parser.error(f'baseline {args.baseline!r} is not available locally. In a shallow clone, '
                     'fetch that history explicitly and rerun; the benchmark performs no network requests.')
    try:
        baseline = {path: git('show', baseline_sha + ':' + path) for path in SOURCE_FILES}
        current = {path: (ROOT / path).read_text() for path in SOURCE_FILES}
        source = [shader_source(version[SOURCE_FILES[0]], version[SOURCE_FILES[1]])
                  for version in (baseline, current)]
        planes = [reduction_planes(version[SOURCE_FILES[0]]) for version in (baseline, current)]
    except (subprocess.CalledProcessError, ValueError, AssertionError) as error:
        parser.error(f'cannot extract compatible baseline/current shader kernels: {error}')
    if platform.system() != 'Darwin':
        parser.error('this native benchmark requires macOS, the Metal GPU, and Apple Clang')
    status = git('status', '--porcelain=v1', '--untracked-files=normal').splitlines()
    metadata = {
        'schema_version': 1,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'baseline': {'ref': args.baseline, 'commit': baseline_sha,
                     'source_sha256': {path: sha(body) for path, body in baseline.items()},
                     'compiled_source_sha256': sha(source[0]), 'f16_reduction_planes': planes[0]},
        'current': {'commit': git('rev-parse', 'HEAD').strip(), 'dirty': bool(status),
                    'git_status': status,
                    'source_sha256': {path: sha(body) for path, body in current.items()},
                    'compiled_source_sha256': sha(source[1]), 'f16_reduction_planes': planes[1]},
        'parameters': {'samples': args.samples, 'repeats': args.repeats,
                       'warmup': args.warmup, 'math': args.math, 'seed': 0x19ae548d},
        'method': {
            'scope': 'direct fused shader dispatches with identical reused inputs and hot resident weights',
            'timing': 'GPU command end-start divided by dispatch count; compilation, allocation, references and comparisons excluded',
            'order': 'baseline/current and current/baseline alternate each round; starting order also alternates by case',
            'validation': 'full buffers, state, untouched slots and guards checked bitwise against baseline ordinary matvec plus a separate state store, after every arm',
            'limitations': [
                'No model is loaded. Production dispatch eligibility and runtime kernel incidence are not measured.',
                'No SSD streaming, full-model attention, inter-layer scheduling or token throughput is measured.',
                'Repeated inputs and weights differ from a complete decoding workload; GPU clocks and temperature can still drift.',
                'NR4 is invoked directly even where runtime dispatch would select NR2.',
                'Paired ratio ranges are descriptive sample ranges, not confidence intervals.',
            ],
        },
        'harness_sha256': {path: sha((ROOT / path).read_text()) for path in (
            'speed-bench/metal_f16_decode_bench.py', 'speed-bench/metal_f16_decode_bench.m',
            'tests/metal_decode_test_support.h', 'tests/test_metal_decode_fusions.py',
            'tests/kernel_source.py')},
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ds4-metal-f16-bench-') as directory:
        directory = Path(directory)
        shaders = [directory / 'baseline.metal', directory / 'current.metal']
        for path, body in zip(shaders, source):
            path.write_text(body)
        binary = directory / 'metal_f16_decode_bench'
        native_json = directory / 'native.json'
        command = ['clang', '-O2', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                   '-framework', 'Foundation', '-framework', 'Metal',
                   str(ROOT / 'speed-bench/metal_f16_decode_bench.m'), '-o', str(binary)]
        metadata['compiler'] = subprocess.check_output(['clang', '--version'], text=True).strip()
        subprocess.run(command, check=True)
        native = subprocess.run([str(binary), *map(str, shaders), str(csv_path), str(native_json),
                                 str(args.samples), str(args.repeats), str(args.warmup), args.math,
                                 *map(str, planes)])
        metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
        metadata['exit_code'] = native.returncode
        if native_json.exists():
            metadata['native'] = json.loads(native_json.read_text())
        if native.returncode == 0:
            metadata['paired_summaries'] = paired_summaries(csv_path)
        with json_path.open('x') as stream:
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
        if native.returncode:
            print(f'Benchmark failed; partial measurements: {csv_path}; metadata: {json_path}', file=sys.stderr)
            return native.returncode
    print(f'Raw measurements: {csv_path}\nMetadata and paired summaries: {json_path}')
    for item in metadata['paired_summaries']:
        print(f"{item['math']} {item['case']}: "
              f"{item['baseline_median_us']:.3f} -> {item['current_median_us']:.3f} us; "
              f"paired current/baseline {item['median_paired_current_over_baseline']:.4f} "
              f"[{item['min_paired_current_over_baseline']:.4f}, {item['max_paired_current_over_baseline']:.4f}]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
