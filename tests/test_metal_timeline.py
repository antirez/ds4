#!/usr/bin/env python3
"""Run the real Metal backend's timeline regression without a GGUF model."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "kernel_mul_mv_f16_f32_quad_compressor_store_4"


def verify_native_trace(path):
    contexts, batches, encoders = {}, {}, {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        seq = int(fields[1])
        if fields[0] == "C":
            assert seq not in contexts, line
            contexts[seq] = (fields[2], int(fields[3]), int(fields[4]))
        elif fields[0] == "B":
            assert seq not in batches, line
            count, start, end = map(int, fields[2:5])
            assert count == 1 and 0 < start < end, line
            batches[seq] = count
        elif fields[0] == "E":
            assert seq not in encoders, line
            assert int(fields[2]) == 0 and 0 < int(fields[3]) < int(fields[4]), line
            assert int(fields[8]) == 1 and fields[9] == "384x1x1" and fields[10] == "32x8x1", line
            assert fields[11] == KERNEL, line
            encoders[seq] = fields
        else:
            raise AssertionError(f"Unexpected native record: {line}")
    assert len(batches) == 6 and contexts.keys() == batches.keys() == encoders.keys(), contexts
    assert [contexts[seq] for seq in sorted(contexts)] == [
        ("prefill", 11, 0), *(('decode', 12, 0) for _ in range(4)),
        ("after_owned", 2**32 - 1, 0),
    ], contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.build_only and args.binary is None:
        parser.error("--build-only requires --binary to retain the executable")
    with tempfile.TemporaryDirectory(prefix="ds4-metal-timeline-") as tmp:
        directory = Path(tmp)
        binary = args.binary.resolve() if args.binary else directory / "test_metal_timeline"
        if not args.binary or args.build_only:
            subprocess.run(["make", "ds4_image.o"], cwd=ROOT, check=True)
            subprocess.run(["cc", "-O2", "-fobjc-arc", "-I", str(ROOT),
                            str(ROOT / "tests/test_metal_timeline.m"), str(ROOT / "ds4_image.o"),
                            "-framework", "Foundation", "-framework", "Metal", "-lm", "-pthread", "-o", str(binary)],
                           cwd=ROOT, check=True)
        if args.build_only:
            print(f"Built {binary}")
            return
        trace = directory / "timeline.log"
        env = {key: value for key, value in os.environ.items() if not key.startswith("DS4_")}
        env["DS4_METAL_ENCODER_TIMELINE"] = str(trace)
        result = subprocess.run([str(binary)], cwd=ROOT, env=env)
        if result.returncode == 77:
            raise SystemExit("Metal timestamp counters unavailable; native timeline coverage was not verified")
        result.check_returncode()
        verify_native_trace(trace)
        print("PASS native timeline records: all 6 GPU intervals, immutable phases, single-dispatch names and restart sequence")


if __name__ == "__main__":
    main()
