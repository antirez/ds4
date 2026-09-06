#!/usr/bin/env python3
"""Check llguidance option changes and no-op builds in an isolated directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llguidance-dir', default='.deps/llguidance')
    args = parser.parse_args()
    dependency = Path(args.llguidance_dir).resolve()
    if not (dependency / 'target/release/libllguidance.a').is_file():
        parser.error('build libllguidance.a first (make LLGUIDANCE=1 test-llguidance)')
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='ds4-build-options-') as tmp:
        root = Path(tmp)
        for name in ('Makefile', 'ds4.h', 'ds4_ssd.h', 'ds4_llguidance.h', 'ds4_llguidance.c',
                     'tests/test_llguidance.c'):
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, dest)
        binary = root / 'tests/test_llguidance'
        objects = [root / 'ds4_llguidance.o', root / 'tests/test_llguidance.o']

        def build(enabled, directory):
            result = subprocess.run(
                ['make', '-j4', f'LLGUIDANCE={enabled}', f'LLGUIDANCE_DIR={directory}',
                 'tests/test_llguidance'], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(result.stdout)
            output = subprocess.check_output([str(binary)], cwd=root, text=True)
            assert ('enabled,' if enabled else 'disabled,') in output, output
            return [p.stat().st_mtime_ns for p in [*objects, binary]]

        previous = None
        for enabled in (0, 1, 0, 1):
            current = build(enabled, dependency)
            if previous is not None:
                assert all(a != b for a, b in zip(previous, current)), 'stale build after option switch'
            assert build(enabled, dependency) == current, 'unchanged configuration rebuilt'
            previous = current
        alias = root / 'another-llguidance-checkout'
        alias.symlink_to(dependency, target_is_directory=True)
        current = build(1, alias)
        assert all(a != b for a, b in zip(previous, current)), 'stale build after checkout switch'
        assert build(1, alias) == current, 'unchanged checkout rebuilt'
        subprocess.run(['make', 'clean'], cwd=root, check=True, stdout=subprocess.DEVNULL)
        assert not (root / '.llguidance-config').exists()
    print('llguidance build tests: option switches, checkout switches, no-op builds and clean OK')


if __name__ == '__main__':
    main()
