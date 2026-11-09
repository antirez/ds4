#!/usr/bin/env python3
"""Dump rec's prefill activations for a prompt: the residual after the
embedding and after every layer, for every token, plus logits.

Runs inside rec's runtime (see rec_logits.py) and drives rec's own prefill
(QATSession.prefill: the native MLX graph in chunks of 2048) with the
layers wrapped so each chunk's outputs are captured.  Logits for every row
come from lm_head over the final norm (batched GEMM, as ds4's all-row
logits); `last_logits` is rec's actual sampling input.  This is the
reference for ds4 --flint-prefill-trace (tests/flint/prefill_compare.py).

    rec stop
    tests/flint/rec_prefill_dump.py --text "..." --out case.npz
    tests/flint/rec_prefill_dump.py --messages-json case.json --out case.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNTIMES = Path.home() / ".local/share/rec/runtimes"
MODELS = Path.home() / ".local/share/rec/models"


def newest(path: Path, marker: str) -> Path | None:
    cands = [d for d in path.iterdir() if (d / marker).exists()] if path.is_dir() else []
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def reexec_in_runtime(runtime: Path):
    py = runtime / "python/bin/python3"
    if Path(sys.executable).resolve() == py.resolve():
        return
    if not py.is_file():
        sys.exit("rec runtime python not found at %s; run `rec serve` once to extract it" % py)
    os.execv(str(py), [str(py), "-I", os.path.abspath(__file__), *sys.argv[1:]])


class Capture:
    """Wraps a callable module so FlintTransformer's `layer(h, mask, cache)` records its output."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        self.sink.append(out)
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", type=Path, default=None)
    ap.add_argument("--src", type=Path, default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--messages-json", type=Path)
    g.add_argument("--ids")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runtime = args.runtime or newest(RUNTIMES, "python/bin/python3")
    if runtime is None:
        sys.exit("no rec runtime extracted yet; run `rec serve` once")
    reexec_in_runtime(runtime)

    import numpy as np
    import mlx.core as mx
    from rec_mlx.engine import Engine

    src = args.src or newest(MODELS, "flint-int4.json")
    if src is None:
        sys.exit("no flint-int4 checkpoint; run `rec pull`")
    status = subprocess.run([str(Path.home() / ".local/bin/rec"), "status"], capture_output=True, text=True)
    if "running" in status.stdout.lower() and "no managed" not in status.stdout.lower():
        sys.exit("rec serve is running; stop it first (rec stop)")

    t0 = time.time()
    engine = Engine(src, src / "flint-int4.json", context_limit=args.context)
    print("engine loaded in %.1fs" % (time.time() - t0), file=sys.stderr)

    if args.text is not None:
        ids = engine.tokenizer.encode(args.text, add_special_tokens=False)
    elif args.ids is not None:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        spec = json.loads(args.messages_json.read_text())
        ids = engine.prompt(spec["messages"], tools=spec.get("tools"))
    ids = list(map(int, ids))
    n = len(ids)
    print("prompt: %d tokens" % n, file=sys.stderr)

    native = engine.runtime.model.native
    embed_out, layer_out, norm_out = [], [[] for _ in range(49)], []
    native.model.embed_tokens = Capture(native.model.embed_tokens, embed_out)
    native.model.layers = [Capture(layer, layer_out[i]) for i, layer in enumerate(native.model.layers)]
    native.model.norm = Capture(native.model.norm, norm_out)

    session = engine.runtime.make_session(max_new_tokens=1)
    t0 = time.time()
    logits = session.prefill(ids)
    mx.eval(logits)
    dt = time.time() - t0
    print("prefill: %.2fs (%.0f tok/s)" % (dt, n / dt), file=sys.stderr)

    def cat(chunks):
        arr = mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        return np.array(arr.reshape(-1, 2048).view(mx.uint16))

    hidden = np.empty((50, n, 2048), np.uint16)
    hidden[0] = cat(embed_out)
    for i in range(49):
        hidden[i + 1] = cat(layer_out[i])
    normed = mx.concatenate(norm_out, axis=1) if len(norm_out) > 1 else norm_out[0]
    all_logits = native.lm_head(normed)   # (1, n, V) bf16
    mx.eval(all_logits)
    out = {
        "prompt_ids": np.array(ids, dtype=np.int32),
        "hidden": hidden,
        "logits_all": np.array(all_logits.reshape(n, -1).view(mx.uint16)),
        "last_logits": np.array(logits.reshape(-1).view(mx.uint16)),
        "argmax": np.array([int(mx.argmax(logits.reshape(-1).astype(mx.float32)).item())], np.int32),
        "prefill_seconds": np.array([dt]),
    }
    np.savez(args.out, **out)
    print("wrote %s (hidden %s, argmax %d)" % (args.out, hidden.shape, out["argmax"][0]), file=sys.stderr)


if __name__ == "__main__":
    main()
