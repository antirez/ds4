#!/usr/bin/env python3
"""Dump rec's full logits and per-layer residual traces for a prompt.

Runs inside rec's extracted runtime (its own Python with mlx and rec_mlx),
loads the engine exactly as `rec serve` does, prefills the prompt with the
native MLX model and then decodes `--steps` tokens greedily with the
megakernel, calling `FullModel.forward(state, token)` directly so each step
yields the 150144 logits and the 50 x 2048 bf16 residual trace (embedding
output and the output of every layer).  This is the reference for the
decode trace test of docs/RECURSIVE_FLINT.md (step 1): ds4's decode must
reproduce the traces bit for bit.

Output: an .npz with prompt_ids, tokens (the greedy continuation, where
tokens[0] is the argmax of the prefill logits), prefill_logits (float32),
logits (float32 [steps, V]: the logits that follow feeding tokens[i]),
trace_bits (uint16 [steps, 50, 2048]: bf16 bit patterns) and, with
--dump-kv, the KV banks and metadata after prefill plus each decode step.

Never run this while `rec serve` holds the model: the box has 48 GB.

    rec stop
    tests/flint/rec_logits.py --text "Hello" --steps 8 --out /tmp/hello.npz
    tests/flint/rec_logits.py --messages-json case.json --steps 32 --out case.npz
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", type=Path, default=None, help="rec runtime dir (default: newest extracted)")
    ap.add_argument("--src", type=Path, default=None, help="checkpoint dir (default: newest flint-int4)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="raw text, tokenized without the chat template")
    g.add_argument("--messages-json", type=Path, help="JSON file with {messages: [...], tools?: [...]}")
    g.add_argument("--ids", help="comma-separated prompt token ids")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--dump-kv", action="store_true")
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
    print("engine loaded in %.1fs (device profile %s)" % (time.time() - t0, engine.startup["device_profile"]),
          file=sys.stderr)

    if args.text is not None:
        ids = engine.tokenizer.encode(args.text, add_special_tokens=False)
    elif args.ids is not None:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        spec = json.loads(args.messages_json.read_text())
        ids = engine.prompt(spec["messages"], tools=spec.get("tools"))
    ids = list(map(int, ids))
    print("prompt: %d tokens" % len(ids), file=sys.stderr)

    session = engine.runtime.make_session(max_new_tokens=args.steps + 1)
    t0 = time.time()
    logits = session.prefill(ids)
    mx.eval(logits)
    print("prefill: %.2fs" % (time.time() - t0), file=sys.stderr)
    state = session._state
    prefill_logits = np.array(logits.reshape(-1).astype(mx.float32))
    out = {"prompt_ids": np.array(ids, dtype=np.int32), "prefill_logits": prefill_logits}

    def kv_snapshot(tag):
        # Per layer: (2, 4, rows, 128) bf16 bits, K then V, through rec's own
        # view of its banks (global layers hold `position` rows; sliding layers
        # a 513-row ring, dumped whole once the position passes it).
        storage = state["kv"]
        pos = state["position"]
        for il in range(49):
            row = storage.rows[il]
            cap = int(row[62])
            count = pos if row[1] else min(pos, cap)
            view = storage.layer(il, count)
            arr = view.view(mx.uint16) if view.dtype == mx.bfloat16 else view
            out["%s_kv_layer%d" % (tag, il)] = np.array(arr)
        out["%s_metadata" % tag] = np.array(state["metadata"])
        out["%s_position" % tag] = np.array([pos], dtype=np.int64)

    if args.dump_kv:
        kv_snapshot("prefill")

    tokens = [int(mx.argmax(logits.reshape(-1).astype(mx.float32)).item())]
    step_logits = []
    traces = []
    t0 = time.time()
    for i in range(args.steps):
        lg, trace = engine.runtime.forward(state, tokens[-1])
        mx.eval(lg, trace)
        step_logits.append(np.array(lg.reshape(-1).astype(mx.float32)))
        tr = trace.view(mx.uint16) if trace.dtype == mx.bfloat16 else trace
        traces.append(np.array(tr).reshape(-1))
        tokens.append(int(mx.argmax(lg.reshape(-1).astype(mx.float32)).item()))
        if args.dump_kv and i == 0:
            kv_snapshot("step0")
    dt = time.time() - t0
    print("decode: %d steps in %.2fs (%.1f tok/s)" % (args.steps, dt, args.steps / dt if dt else 0), file=sys.stderr)
    print("tokens: %s" % tokens, file=sys.stderr)
    print("text: %r" % engine.tokenizer.decode(tokens, skip_special_tokens=False), file=sys.stderr)

    out["tokens"] = np.array(tokens, dtype=np.int32)
    out["logits"] = np.stack(step_logits) if step_logits else np.zeros((0, prefill_logits.shape[0]), np.float32)
    tr = np.stack(traces) if traces else np.zeros((0, 50 * 2048), np.uint16)
    out["trace_bits"] = tr.reshape(len(traces), -1, 2048) if traces else tr
    out["trace_dtype"] = np.array(["bf16"])
    np.savez(args.out, **out)
    print("wrote %s (trace shape %s)" % (args.out, out["trace_bits"].shape), file=sys.stderr)


if __name__ == "__main__":
    main()
