#!/usr/bin/env python3
"""Streaming exactness: ds4's Flint output with the routed experts streamed
from disk through a capped cache must equal the resident output bit for bit
(docs/RECURSIVE_FLINT.md step 4).

For each golden case runs `ds4 --flint-generate` teacher-forced twice, resident
and with `--ssd-streaming --ssd-streaming-cache-experts CAP`, and compares
every step's argmax and log-probabilities exactly.  The cache cap is kept
small so decode misses, evictions and every prefill policy (cache for short
appends, staged selections, whole layers with read-ahead on the long case)
are exercised; the streaming run prints its statistics.

    tests/flint/stream_check.py [--case ID ...] [--cache 1GB] [--max-tokens 32]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def run(ds4, model, prompt_ids, expected, tmpdir, tag, extra, env):
    inp = tmpdir / ("%s.in" % tag)
    out = tmpdir / ("%s.out" % tag)
    with open(inp, "wb") as f:
        f.write(b"FLGN")
        f.write(struct.pack("<IIIIII", 2, len(prompt_ids), len(expected), len(expected), 0, 0))
        f.write(np.asarray(prompt_ids, np.int32).tobytes())
        f.write(np.asarray(expected, np.int32).tobytes())
    r = subprocess.run([str(ds4), "-m", str(model)] + extra + ["--flint-generate", str(inp), str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("ds4 exited with %d" % r.returncode)
    with open(out, "rb") as f:
        assert f.read(4) == b"FLGO"
        (n,) = struct.unpack("<I", f.read(4))
        rows = np.frombuffer(f.read(n * 16), dtype=np.dtype([("fed", "<i4"), ("argmax", "<i4"), ("lp_fed", "<f4"), ("lp_arg", "<f4")]))
        prefill_ms, decode_ms = struct.unpack("<dd", f.read(16))
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    lines = [l for l in r.stderr.splitlines() if "Flint streaming:" in l or "Flint SSD streaming:" in l]
    return rows, prefill_ms, decode_ms, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=HERE / "golden.json")
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--case", action="append")
    ap.add_argument("--cache", default="1GB", help="--ssd-streaming-cache-experts value for the streaming run")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    g = json.loads(args.golden.read_text())
    tmpdir = Path(tempfile.mkdtemp(prefix="flint-stream-"))
    env = dict(os.environ, DS4_FLINT_STREAM_STATS="1")
    failed = 0
    for case in g["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        ids = case["prompt_ids"]
        expected = case["generated_ids"][: args.max_tokens]
        res, rpf, rdec, _ = run(args.ds4, args.model, ids, expected, tmpdir, case["id"] + "-res", [], env)
        st, spf, sdec, lines = run(args.ds4, args.model, ids, expected, tmpdir, case["id"] + "-stream",
                                   ["--ssd-streaming", "--ssd-streaming-cache-experts", args.cache], env)
        same = len(res) == len(st) and np.array_equal(res.view(np.uint8), st.view(np.uint8))
        if not same:
            failed += 1
            n = min(len(res), len(st))
            diff = np.nonzero((res["argmax"][:n] != st["argmax"][:n]) | (res["lp_fed"][:n] != st["lp_fed"][:n]))[0]
            where = "first difference at step %d" % diff[0] if len(diff) else "different lengths %d vs %d" % (len(res), len(st))
        else:
            where = ""
        print("%-20s prompt %6d steps %3d  %s  resident prefill %.0f ms decode %.1f ms/tok | streaming prefill %.0f ms decode %.1f ms/tok  %s" % (
            case["id"], len(ids), len(res), "IDENTICAL" if same else "DIFFERENT", rpf, rdec / max(1, len(res) - 1),
            spf, sdec / max(1, len(st) - 1), where))
        for l in lines:
            print("    " + l.split("ds4: ", 1)[-1])
    try:
        tmpdir.rmdir()
    except OSError:
        pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
