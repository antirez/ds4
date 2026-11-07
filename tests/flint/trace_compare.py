#!/usr/bin/env python3
"""Decode trace test: ds4's Flint megakernel against rec, bit for bit.

Takes a dump from tests/flint/rec_logits.py made with --dump-kv (prefill KV
of every layer, the greedy tokens and, per decode step, rec's logits and
50 x 2048 bf16 residual trace), writes it in the flat format
`ds4 --flint-trace IN OUT` reads, runs ds4 from the same KV state over the
same tokens, and compares every step's trace and logits.  The gate of step 1
in docs/RECURSIVE_FLINT.md is zero differing elements.

    rec stop
    tests/flint/rec_logits.py --text "..." --steps 16 --dump-kv --out case.npz
    tests/flint/trace_compare.py case.npz [--steps 16] [--model GGUF]

Diagnostics read by ds4 (environment): DS4_FLINT_TRACE_WARM=N runs the
steps N extra times first and restores the imported KV (steady-state
timing); DS4_FLINT_LAYER_COUNT=L decodes the first L layers only;
DS4_FLINT_WORKSPACE_DUMP=FILE appends the kernel workspace after each step
(q, k, v, attention, o, gate, up, act, down of the last layer run);
DS4_FLINT_FAST_MATH=1 compiles the kernel with fast math (it then differs
from rec already in the embedding row).
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
N_LAYER = 49
N_VOCAB = 150144
TRACE = 50 * 2048


def write_input(z, path: Path, steps: int):
    pos = int(z["prefill_position"][0])
    tokens = z["tokens"][:steps].astype(np.int32)
    with open(path, "wb") as f:
        f.write(b"FLTR")
        f.write(struct.pack("<IIII", 1, pos, len(tokens), N_LAYER))
        f.write(tokens.tobytes())
        for il in range(N_LAYER):
            kv = z["prefill_kv_layer%d" % il]   # (2, 4, rows, 128) uint16, K then V
            rows = kv.shape[2]
            f.write(struct.pack("<I", rows))
            f.write(np.ascontiguousarray(kv[0]).tobytes())
            f.write(np.ascontiguousarray(kv[1]).tobytes())
    return pos, tokens


def read_output(path: Path):
    with open(path, "rb") as f:
        assert f.read(4) == b"FLTO"
        n_steps, n_vocab, trace = struct.unpack("<III", f.read(12))
        assert n_vocab == N_VOCAB and trace == TRACE
        logits = np.empty((n_steps, n_vocab), np.float32)
        traces = np.empty((n_steps, 50, 2048), np.uint16)
        for i in range(n_steps):
            logits[i] = np.frombuffer(f.read(n_vocab * 4), np.float32)
            traces[i] = np.frombuffer(f.read(trace * 2), np.uint16).reshape(50, 2048)
    return logits, traces


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--keep", action="store_true", help="keep the IN/OUT files next to the npz")
    args = ap.parse_args()

    z = np.load(args.npz)
    steps = len(z["tokens"]) - 1 if args.steps is None else min(args.steps, len(z["tokens"]) - 1)
    if "prefill_kv_layer0" not in z.files:
        sys.exit("dump has no KV; rerun rec_logits.py with --dump-kv")
    tmpdir = args.npz.parent if args.keep else Path(tempfile.mkdtemp(prefix="flint-trace-"))
    inp = tmpdir / (args.npz.stem + ".in.bin")
    out = tmpdir / (args.npz.stem + ".out.bin")
    pos, tokens = write_input(z, inp, steps)
    print("prefill position %d, %d decode steps; input %.1f MB" % (pos, steps, inp.stat().st_size / 1e6), file=sys.stderr)
    r = subprocess.run([str(args.ds4), "-m", str(args.model), "--flint-trace", str(inp), str(out)],
                       capture_output=True, text=True)
    sys.stderr.write(r.stderr[-3000:])
    if r.returncode != 0:
        sys.exit("ds4 exited with %d" % r.returncode)
    logits, traces = read_output(out)

    rec_trace = z["trace_bits"][:steps]            # (steps, 50, 2048) uint16
    rec_logits = z["logits"][:steps]               # (steps, V) float32 (bf16 values)
    bad = 0
    for i in range(steps):
        diff_layers = []
        for layer in range(50):
            d = int(np.count_nonzero(traces[i, layer] != rec_trace[i, layer]))
            if d:
                diff_layers.append((layer, d))
        ld = int(np.count_nonzero(logits[i] != rec_logits[i]))
        ds4_tok = int(np.argmax(logits[i]))
        rec_tok = int(np.argmax(rec_logits[i]))
        if diff_layers or ld:
            bad += 1
            first = diff_layers[0] if diff_layers else None
            if first is not None:
                a = bf16_bits_to_f32(traces[i, first[0]])
                b = bf16_bits_to_f32(rec_trace[i, first[0]])
                mx = float(np.max(np.abs(a - b)))
                where = "first in trace row %d (%s): %d elements differ, max |diff| %.3g" % (
                    first[0], "embedding" if first[0] == 0 else "layer %d output" % (first[0] - 1), first[1], mx)
            else:
                where = "trace identical"
            print("step %2d token %6d pos %6d: MISMATCH; %s; logits differ in %d of %d (argmax ds4 %d, rec %d)" % (
                i, tokens[i], pos + i, where, ld, N_VOCAB, ds4_tok, rec_tok))
            if len(diff_layers) > 1:
                print("        differing rows: %s" % diff_layers[:12])
        else:
            print("step %2d token %6d pos %6d: identical (argmax %d)" % (i, tokens[i], pos + i, ds4_tok))
    print("%d steps, %d with differences" % (steps, bad))
    if not args.keep:
        inp.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
