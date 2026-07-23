#!/usr/bin/env python3
"""Compile all .comp GLSL shaders to .spv SPIR-V binaries.
Called from Makefile during build. Outputs to shaders/spv/."""

import subprocess, sys, os, pathlib

GLSLANG = os.environ.get("GLSLANG", "../glslangValidator")
SRC_DIR = pathlib.Path(__file__).parent
OUT_DIR = SRC_DIR / "spv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

shaders = sorted(SRC_DIR.glob("*.comp"))
if not shaders:
    print("No .comp shaders found, skipping.")
    sys.exit(0)

compiled = 0
failed = 0
for src in shaders:
    spv = OUT_DIR / f"{src.stem}.spv"
    result = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.2", str(src), "-o", str(spv)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        compiled += 1
    else:
        failed += 1
        print(f"FAIL {src.name}: {result.stderr.strip()}", file=sys.stderr)

print(f"Compiled {compiled} shaders{' (with {failed} failures)' if failed else ''} to {OUT_DIR}")
sys.exit(1 if failed else 0)
