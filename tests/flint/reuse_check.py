#!/usr/bin/env python3
"""Gate 2 of docs/RECURSIVE_FLINT.md: does reusing decode KV as a prefix
change Flint's probabilities enough to matter?

rec never reuses decode KV: after a reply it re-prefills prompt + reply in
the background so the next turn starts from batched-prefill state.  ds4's
sessions reuse decode KV everywhere.  This measures the difference on the
golden cases: a session prefills the prompt and decodes rec's reply token
by token (teacher-forced), then syncs prompt + reply + a follow-up user
turn, reusing the decode KV for the reply; a fresh session prefills the
whole sequence.  The two logits vectors after the follow-up are compared
(KL both ways, |delta logprob| of the argmax, top-1 agreement).  For scale,
the same table prints rec's own decode-vs-prefill spread figures measured
in the step-2 notes.

    tests/flint/reuse_check.py [--case ID ...] [--model GGUF]
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
V = 150144
FOLLOW = "<|user|>Thanks. Now state your final answer in one short sentence.<|assistant|>"


def log_softmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=HERE / "golden.json")
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json (default: rec's checkpoint dir)")
    ap.add_argument("--case", action="append")
    ap.add_argument("--reply-tokens", type=int, default=128)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = args.tokenizer
    if tok_path is None:
        models = Path.home() / ".local/share/rec/models"
        cands = sorted(models.glob("*/tokenizer.json"), key=lambda p: p.stat().st_mtime) if models.is_dir() else []
        if not cands:
            sys.exit("no tokenizer.json found; pass --tokenizer")
        tok_path = cands[-1]
    tok = Tokenizer.from_file(str(tok_path))
    follow = tok.encode(FOLLOW, add_special_tokens=False).ids

    g = json.loads(args.golden.read_text())
    tmpdir = Path(tempfile.mkdtemp(prefix="flint-reuse-"))
    print("%-20s %7s %6s  %-10s %-10s %-10s %s" % ("case", "prompt", "reply", "KL@follow", "|dlp| top1", "top1 same", "reused / fresh prefill ms"))
    for case in g["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        ids = case["prompt_ids"]
        expected = case["generated_ids"][: args.reply_tokens]
        inp = tmpdir / (case["id"] + ".in")
        out = tmpdir / (case["id"] + ".out")
        with open(inp, "wb") as f:
            f.write(b"FLGN")
            f.write(struct.pack("<IIIIII", 2, len(ids), len(expected), len(expected), 0, len(follow)))
            f.write(np.asarray(ids, np.int32).tobytes())
            f.write(np.asarray(expected, np.int32).tobytes())
            f.write(np.asarray(follow, np.int32).tobytes())
        r = subprocess.run([str(args.ds4), "-m", str(args.model), "--flint-generate", str(inp), str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
            sys.exit("ds4 exited with %d" % r.returncode)
        with open(out, "rb") as f:
            assert f.read(4) == b"FLGO"
            (n_steps,) = struct.unpack("<I", f.read(4))
            f.read(n_steps * 16)
            f.read(16)
            (has_follow,) = struct.unpack("<I", f.read(4))
            assert has_follow == 1
            reused = np.frombuffer(f.read(V * 4), np.float32).astype(np.float64)
            fresh = np.frombuffer(f.read(V * 4), np.float32).astype(np.float64)
            reuse_ms, fresh_ms = struct.unpack("<dd", f.read(16))
            (n2,) = struct.unpack("<I", f.read(4))
            steps = np.frombuffer(f.read(n2 * 12), dtype=np.dtype([("kl", "<f4"), ("dlp", "<f4"), ("same", "<u4")]))
        inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
        if n2 == 0:
            sys.exit("%s: no continuation steps were recorded" % case["id"])
        la, lb = log_softmax(reused), log_softmax(fresh)
        kl_fr = float(np.sum(np.exp(lb) * (lb - la)))
        top = int(lb.argmax())
        print("%-20s %7d %6d  %-10.3g %-10.4f %-10s %.0f / %.0f   along %d greedy steps: KL mean %.3g max %.3g, |dlp| mean %.4f max %.3f, top1 same %.1f%%" % (
            case["id"], len(ids), n_steps, kl_fr, abs(la[top] - lb[top]), "yes" if la.argmax() == top else "NO", reuse_ms, fresh_ms,
            n2, steps["kl"].mean(), steps["kl"].max(), steps["dlp"].mean(), steps["dlp"].max(), 100 * steps["same"].mean()))
    try:
        tmpdir.rmdir()
    except OSError:
        pass
    print("\nscale: rec decode vs rec prefill at identical positions (step-2 notes): KL mean 0.03-0.26, max 4.0; |dlp| mean 0.05-0.32")


if __name__ == "__main__":
    main()
