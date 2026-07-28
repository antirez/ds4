#!/usr/bin/env python3
"""Backend-seam check (see the contract atop src/pulsar_gpu.h).

Nothing outside src/cuda/ may touch CUDA APIs directly. Comments and string
literals are stripped before matching, so documentation may freely explain
what the backend does underneath; code may not do it.
"""
import re
import sys
from pathlib import Path

ROOTS = ["src/engine", "src/server", "src/cli", "src/agent", "src/lib"]
FILES = ["src/pulsar.h", "src/pulsar_gpu.h"]

INCLUDE = re.compile(r'#\s*include\s*[<"](cuda|cublas|cooperative_groups|curand|cutlass)')
API = re.compile(
    r"\b(cudaMalloc\w*|cudaFree\w*|cudaMemcpy\w*|cudaStream\w*|cudaEvent\w*|"
    r"cudaError\w*|cudaDeviceSynchronize|cudaGetLastError|cudaLaunch\w*|"
    r"cublas\w+|cuMem\w+|curand\w+)\b|<<<")

def strip_comments_strings(txt):
    out = []
    i, n = 0, len(txt)
    while i < n:
        c = txt[i]
        if txt.startswith("/*", i):
            j = txt.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * txt.count("\n", i, j))
            i = j
        elif txt.startswith("//", i):
            j = txt.find("\n", i)
            j = n if j < 0 else j
            i = j
        elif c in "\"'":
            q = c
            j = i + 1
            while j < n:
                if txt[j] == "\\":
                    j += 2
                    continue
                if txt[j] == q:
                    j += 1
                    break
                j += 1
            out.append(q + q)
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)

def main():
    bad = []
    paths = []
    for root in ROOTS:
        paths += [p for p in Path(root).rglob("*") if p.suffix in (".h", ".hpp", ".cpp", ".inc")]
    paths += [Path(f) for f in FILES]
    for p in paths:
        code = strip_comments_strings(p.read_text(errors="replace"))
        for ln_no, line in enumerate(code.split("\n"), 1):
            if INCLUDE.search(line) or API.search(line):
                bad.append(f"{p}:{ln_no}: {line.strip()}")
    if bad:
        print("SEAM VIOLATION:")
        print("\n".join(bad))
        return 1
    print(f"seam-check: {len(paths)} host files are CUDA-API clean")
    return 0

sys.exit(main())
