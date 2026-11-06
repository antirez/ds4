#!/usr/bin/env python3
"""Compare Metal RoPE with the released model.py functions on CPU Torch."""
import argparse
import ast
from functools import lru_cache
import math
from pathlib import Path
import struct
import subprocess
import tempfile

import torch


def check(reference, executable):
    source = ast.parse(Path(reference).read_text())
    names = {"precompute_freqs_cis", "apply_rotary_emb"}
    source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(source.body) == 2
    namespace = dict(torch=torch, math=math, lru_cache=lru_cache)
    exec(compile(source, reference, "exec"), namespace)
    frequencies = namespace["precompute_freqs_cis"]
    rotate = namespace["apply_rotary_emb"]
    positions = [0, 1, 127, 128, 511, 4095, 16383, 65535, 100000, 262143, 1048575]
    torch.manual_seed(4141)
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(struct.pack("<I", len(positions) * 2 * 2))
        for layer in (0, 2):
            freq = frequencies(64, max(positions) + 1, 65536 if layer else 0,
                               160000 if layer else 10000, 16, 32, 1)
            for pos in positions:
                for inverse in (False, True):
                    x = torch.randn(1, 1, 4, 512).to(torch.bfloat16)
                    y = x.clone()
                    rotate(y[..., -64:], freq[pos:pos + 1], inverse)
                    fp.write(struct.pack("<III", layer, pos, inverse))
                    fp.write(x.float().numpy().tobytes())
                    fp.write(y.float().numpy().tobytes())
        fp.flush()
        subprocess.run([executable, "--rope-reference", fp.name], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="released inference/model.py")
    parser.add_argument("--executable", default="./tests/test_deepseek41_graph")
    args = parser.parse_args()
    check(args.reference, args.executable)
