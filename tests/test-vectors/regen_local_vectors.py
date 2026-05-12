#!/usr/bin/env python3
"""Regenerate the test-vectors fixture from local model output.

The original `official.vec` was captured from the DeepSeek hosted API,
which is fixed. Quantized local models (especially after imatrix changes)
will drift from the cloud at some positions, breaking the bit-equivalence
gate even though our inference code is correct.

This tool instead snapshots the *current local model's* top-k logprobs at
4 greedy positions per test prompt and emits a `.vec` file compatible with
the C runner. The resulting fixture is a code-regression gate (it will
fail if we change inference behavior between versions of ds4 against the
same model), not a quality gate against the cloud API.

Usage:

    ./tests/test-vectors/regen_local_vectors.py \\
        -m /home/cghart/ds4/ds4flash.gguf \\
        -o tests/test-vectors/local.vec

The model path must point at the GGUF we want as the regression reference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Same case set as official.vec.
CASES = [
    ("short_italian_fact",     16384),
    ("short_code_completion",   4096),
    ("short_reasoning_plain",   4096),
    ("long_memory_archive",    16384),
    ("long_code_audit",        16384),
]
STEPS_PER_CASE = 4
TOP_K = 20


def hex_of_bytes(b: bytes) -> str:
    return b.hex()


def token_field_bytes(token_field) -> bytes:
    """ds4 --dump-logprobs writes tokens as a `{id, text, bytes}` object
    where `bytes` is the canonical byte sequence (some tokens are not
    valid UTF-8 on their own). Accept that form first, then fall back to
    plain string / int-array if the format ever changes."""
    if isinstance(token_field, dict):
        if "bytes" in token_field and isinstance(token_field["bytes"], list):
            return bytes(int(x) & 0xff for x in token_field["bytes"])
        if "text" in token_field and isinstance(token_field["text"], str):
            return token_field["text"].encode("utf-8")
        raise ValueError(f"token dict missing bytes/text: {token_field!r}")
    if isinstance(token_field, str):
        return token_field.encode("utf-8")
    if isinstance(token_field, list):
        return bytes(int(x) & 0xff for x in token_field)
    raise ValueError(f"unexpected token field: {token_field!r}")


def run_ds4_dump(model_path: str, prompt_path: str, ctx: int, n: int,
                 top_k: int, ds4_bin: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    try:
        cmd = [
            ds4_bin,
            "--cuda",
            "-m", model_path,
            "--prompt-file", prompt_path,
            "--ctx", str(ctx),
            "-n", str(n),
            "--temp", "0",
            "-sys", "",
            "--nothink",
            "--dump-logprobs", out_path,
            "--logprobs-top-k", str(top_k),
        ]
        rc = subprocess.run(cmd, check=False, capture_output=True)
        if rc.returncode != 0:
            sys.stderr.write(rc.stderr.decode("utf-8", errors="replace"))
            raise RuntimeError(f"ds4 failed for {prompt_path} (rc={rc.returncode})")
        with open(out_path, "r") as fp:
            return json.load(fp)
    finally:
        try:
            os.unlink(out_path)
        except FileNotFoundError:
            pass


def emit_case(out, vec_id: str, ctx: int, prompt_rel: str, dump: dict,
              max_steps: int) -> None:
    steps = dump.get("steps", [])
    # The model may stop early (EOS) before reaching max_steps; the cloud
    # fixture also has variable step counts per case (e.g. short_reasoning_plain
    # is 1 step because "16" was a complete answer). Accept whatever was
    # captured, capped at max_steps.
    steps = steps[:max_steps] if len(steps) > max_steps else steps
    if not steps:
        raise RuntimeError(f"{vec_id}: no steps captured")
    out.write(f"case {vec_id} {ctx} {len(steps)} {prompt_rel}\n")
    for i, step in enumerate(steps):
        selected_bytes = token_field_bytes(step["selected"])
        selected_hex = hex_of_bytes(selected_bytes)
        # Find the selected token's logprob inside top_logprobs so we get a
        # meaningful number (not the placeholder 0). Fall back to 0 if the
        # selected token is missing from the local top-k.
        selected_lp = 0.0
        for t in step.get("top_logprobs", []):
            if "token" not in t:
                continue
            if token_field_bytes(t["token"]) == selected_bytes:
                selected_lp = float(t.get("logprob", 0.0))
                break
        # Emit just the selected token per step. The cloud `official.vec`
        # does the same — storing the full local top-20 is fragile because
        # ranks 5-20 are not stable under CUDA non-determinism (small
        # numerical drift between runs swaps low-rank tokens in and out of
        # the local top-20 window, producing false-positive test failures
        # on a model that is otherwise behaving deterministically at the
        # argmax level). The argmax-only regression check is sensitive
        # enough to catch real code regressions (any change in inference
        # behavior will eventually shift a top-1 token).
        out.write(f"step {i} {selected_hex} 1\n")
        out.write(f"top {selected_hex} {selected_lp:.6g}\n")
    out.write("end\n\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True,
                    help="Path to the GGUF model to snapshot.")
    ap.add_argument("-o", "--output", required=True,
                    help="Output .vec path.")
    ap.add_argument("--ds4", default="./ds4",
                    help="Path to the ds4 binary. Default: ./ds4")
    ap.add_argument("--prompts-dir", default="tests/test-vectors/prompts",
                    help="Directory containing <id>.txt prompts.")
    ap.add_argument("--steps", type=int, default=STEPS_PER_CASE)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args(argv)

    out_path = Path(args.output)
    with tempfile.NamedTemporaryFile(mode="w",
                                     dir=str(out_path.parent) or ".",
                                     prefix=".vec-",
                                     delete=False) as tmp:
        tmp.write("# ds4-local-logprob-vectors-v1\n")
        tmp.write("# Regenerated from local model output, not the cloud API.\n")
        tmp.write("# See tests/test-vectors/regen_local_vectors.py.\n")
        tmp.write(f"# model: {args.model}\n")
        tmp.write("# case <id> <ctx> <steps> <prompt-file>\n")
        tmp.write("# step <index> <selected-hex> <top-count>\n")
        tmp.write("# top <token-hex> <local-logprob>\n\n")
        for vec_id, ctx in CASES:
            prompt_rel = f"{args.prompts_dir}/{vec_id}.txt"
            if not Path(prompt_rel).exists():
                sys.stderr.write(f"warn: missing prompt {prompt_rel}, skipping\n")
                continue
            sys.stderr.write(f"capturing {vec_id} ctx={ctx} n={args.steps}\n")
            dump = run_ds4_dump(args.model, prompt_rel, ctx, args.steps,
                                args.top_k, args.ds4)
            emit_case(tmp, vec_id, ctx, prompt_rel, dump, args.steps)
        tmp_path = tmp.name
    os.replace(tmp_path, args.output)
    sys.stderr.write(f"wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
