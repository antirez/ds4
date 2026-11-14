#!/usr/bin/env python3
"""Compare ds4's Flint prefill with rec's, layer by layer.

Takes a dump from tests/flint/rec_prefill_dump.py, runs
`ds4 -m GGUF --flint-prefill-trace IN OUT` on the same token ids and
reports, per layer, how far ds4's residual is from rec's (max and mean
absolute difference against the residual's own scale, and the fraction of
bit-identical elements), then the logits: argmax agreement per row, mean
KL divergence rec || ds4 per row, and the last row's top-5.  Prefill is not
expected to be bit-exact (docs/RECURSIVE_FLINT.md, Design: Prefill); what
the table shows is whether the error grows like rounding noise or jumps at
one layer, which names the kernel to look at.

    tests/flint/prefill_compare.py case.npz [--model GGUF]
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
V = 150144


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--ds4", type=Path, default=ROOT / "ds4")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    z = np.load(args.npz)
    ids = z["prompt_ids"].astype(np.int32)
    n = len(ids)
    tmpdir = args.npz.parent if args.keep else Path(tempfile.mkdtemp(prefix="flint-prefill-"))
    inp = tmpdir / (args.npz.stem + ".pt.in")
    out = tmpdir / (args.npz.stem + ".pt.out")
    with open(inp, "wb") as f:
        f.write(b"FLPT")
        f.write(struct.pack("<II", 1, n))
        f.write(ids.tobytes())
    r = subprocess.run([str(args.ds4), "-m", str(args.model), "--flint-prefill-trace", str(inp), str(out)],
                       capture_output=True, text=True)
    sys.stderr.write(r.stderr[-2000:])
    if r.returncode != 0:
        sys.exit("ds4 exited with %d" % r.returncode)
    with open(out, "rb") as f:
        assert f.read(4) == b"FLPO"
        nn_, layers, width, vocab = struct.unpack("<IIII", f.read(16))
        assert (nn_, layers, width, vocab) == (n, 50, 2048, V)
        hidden = np.frombuffer(f.read(50 * n * 2048 * 2), np.uint16).reshape(50, n, 2048)
        logits_all = np.frombuffer(f.read(n * V * 2), np.uint16).reshape(n, V)
        last = np.frombuffer(f.read(V * 2), np.uint16)
    if not args.keep:
        inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
        try: tmpdir.rmdir()
        except OSError: pass

    rec_hidden = z["hidden"]
    print("%-6s %-10s %-10s %-10s %-10s %-8s" % ("layer", "scale", "max|d|", "mean|d|", "rel mean", "exact"))
    for layer in range(50):
        a = bf16_to_f32(hidden[layer]); b = bf16_to_f32(rec_hidden[layer])
        d = np.abs(a - b)
        scale = float(np.sqrt(np.mean(b * b)))
        name = "embed" if layer == 0 else "L%d" % (layer - 1)
        worst = np.unravel_index(int(d.argmax()), d.shape)
        rowd = d.max(axis=1)
        print("%-6s %-10.4g %-10.4g %-10.4g %-10.3g %6.1f%%  max at row %d col %d (rec %.4g ds4 %.4g); rows with max|d|>8*scale: %d" % (
            name, scale, d.max(), d.mean(), d.mean() / scale, 100 * np.mean(hidden[layer] == rec_hidden[layer]),
            worst[0], worst[1], b[worst], a[worst], int(np.sum(rowd > 8 * scale))))

    la = bf16_to_f32(logits_all).astype(np.float64); lb = bf16_to_f32(z["logits_all"]).astype(np.float64)
    agree = np.mean(la.argmax(-1) == lb.argmax(-1))
    lpa = log_softmax(la); lpb = log_softmax(lb)
    kld = np.sum(np.exp(lpb) * (lpb - lpa), axis=-1)
    print("\nall-row logits: argmax agreement %.2f%% over %d rows; KL(rec||ds4) mean %.3g max %.3g" % (100 * agree, n, kld.mean(), kld.max()))
    order = np.argsort(-kld)[:8]
    print("  worst rows by KL: %s" % ", ".join("row %d KL %.3g (argmax ds4 %d rec %d)" % (r, kld[r], la[r].argmax(), lb[r].argmax()) for r in order))
    dis = np.nonzero(la.argmax(-1) != lb.argmax(-1))[0]
    print("  argmax disagreements at rows: %s%s" % (dis[:40].tolist(), " ..." if len(dis) > 40 else ""))
    ra = bf16_to_f32(last).astype(np.float64); rb = bf16_to_f32(z["last_logits"]).astype(np.float64)
    ta = np.argsort(-ra)[:5]; tb = np.argsort(-rb)[:5]
    lra = log_softmax(ra); lrb = log_softmax(rb)
    print("last row (vector head): ds4 top5 %s rec top5 %s; KL %.3g; exact %.1f%%" % (
        ta.tolist(), tb.tolist(), float(np.sum(np.exp(lrb) * (lrb - lra))), 100 * np.mean(last == z["last_logits"])))
    print("  ds4 logprobs of rec's top5: %s" % np.round(lra[tb], 4).tolist())
    print("  rec logprobs of rec's top5: %s" % np.round(lrb[tb], 4).tolist())
    sys.exit(0 if ta[0] == tb[0] else 1)


if __name__ == "__main__":
    main()
