#!/usr/bin/env python3
"""Golden reference check: ds4's Flint sessions against rec serve's outputs.

For every case in tests/flint/golden.json (needs `prompt_ids`, see
golden_prompt_ids.py) runs `ds4 -m GGUF --flint-generate IN OUT` twice:

  * teacher-forced: rec's generated ids are fed; ds4 reports, per step, its
    argmax and the log-probability of rec's token.  Compared with rec's own
    logprobs this measures how far ds4's distribution is from rec's along
    rec's path (mean and max |delta logprob|, and how often ds4's argmax
    equals rec's token);
  * greedy: ds4 generates on its own; the first index where its tokens
    leave rec's is reported.

Prefill is not bit-exact (docs/RECURSIVE_FLINT.md), so greedy paths can
part at a near-tie; the teacher-forced numbers are the gate and must stay
within rec's own spread (`--max-mean-dlp`, `--max-max-dlp`).

    tests/flint/check_golden.py [--case ID ...] [--model GGUF] [--max-tokens N]
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


def run_generate(ds4, model, prompt_ids, expected, max_new, tmpdir, tag):
    inp = tmpdir / ("%s.in" % tag)
    out = tmpdir / ("%s.out" % tag)
    with open(inp, "wb") as f:
        f.write(b"FLGN")
        f.write(struct.pack("<IIIII", 1, len(prompt_ids), len(expected), max_new, 0))
        f.write(np.asarray(prompt_ids, np.int32).tobytes())
        f.write(np.asarray(expected, np.int32).tobytes())
    r = subprocess.run([str(ds4), "-m", str(model), "--flint-generate", str(inp), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        sys.exit("ds4 exited with %d" % r.returncode)
    timing = [l for l in r.stderr.splitlines() if "Flint generate:" in l]
    with open(out, "rb") as f:
        assert f.read(4) == b"FLGO"
        (n_steps,) = struct.unpack("<I", f.read(4))
        rows = np.frombuffer(f.read(n_steps * 16), dtype=np.dtype([("fed", "<i4"), ("argmax", "<i4"), ("lp_fed", "<f4"), ("lp_arg", "<f4")]))
        prefill_ms, decode_ms = struct.unpack("<dd", f.read(16))
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return rows, prefill_ms, decode_ms, timing[-1] if timing else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=HERE / "golden.json")
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--case", action="append")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-mean-dlp", type=float, default=0.05, help="gate: mean |logprob delta| per case")
    ap.add_argument("--max-max-dlp", type=float, default=1.0, help="gate: max |logprob delta| per case")
    ap.add_argument("--no-greedy", action="store_true")
    args = ap.parse_args()

    g = json.loads(args.golden.read_text())
    tmpdir = Path(tempfile.mkdtemp(prefix="flint-golden-"))
    failed = 0
    for case in g["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        if "prompt_ids" not in case:
            sys.exit("%s has no prompt_ids; run tests/flint/golden_prompt_ids.py" % case["id"])
        ids = case["prompt_ids"]
        expected = case["generated_ids"][: args.max_tokens]
        rec_lp = np.asarray(case["logprobs"][: len(expected)], np.float64)
        rows, pf_ms, dec_ms, _ = run_generate(args.ds4, args.model, ids, expected, len(expected), tmpdir, case["id"] + "-tf")
        n = len(rows)
        dlp = rows["lp_fed"][:n] - rec_lp[:n]
        agree = np.mean(rows["argmax"][:n] == np.asarray(expected[:n]))
        first_tok_ok = rows["argmax"][0] == expected[0]
        line = "%-20s prompt %6d  tf: argmax=rec %5.1f%%  |dlp| mean %.4f max %.3f (step %d)  first %s  prefill %.0f ms (%.0f tok/s) decode %.1f ms/tok" % (
            case["id"], len(ids), 100 * agree, np.abs(dlp).mean(), np.abs(dlp).max(), int(np.abs(dlp).argmax()),
            "ok" if first_tok_ok else "DIFF", pf_ms, 1000 * len(ids) / pf_ms if pf_ms else 0, dec_ms / max(1, n - 1))
        ok = np.abs(dlp).mean() <= args.max_mean_dlp and np.abs(dlp).max() <= args.max_max_dlp
        if not args.no_greedy:
            grows, _, _, _ = run_generate(args.ds4, args.model, ids, [], len(expected), tmpdir, case["id"] + "-greedy")
            gen = grows["fed"].tolist()
            m = 0
            while m < min(len(gen), len(expected)) and gen[m] == expected[m]:
                m += 1
            line += "  greedy: %d/%d tokens match%s" % (m, len(expected), "" if m == len(expected) else " (diverges at %d: ds4 %d rec %d)" % (m, gen[m] if m < len(gen) else -1, expected[m]))
        print(line + ("" if ok else "  FAIL"))
        if not ok:
            failed += 1
    try:
        tmpdir.rmdir()
    except OSError:
        pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
