#!/usr/bin/env python3
"""Regenerate tests/test-vectors/official.vec from the local ds4flash.gguf.

Runs ./ds4 --dump-logprobs with the same strict configuration that
test_local_logprob_vectors() uses in the C runner (MPP off, prefill chunk 2048),
then emits the compact v2 vec format.

Per-case ctx and step count come from the prompts table below, matching the
existing official.vec layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CASES = [
    ("short_italian_fact",     16384, 4),
    ("short_code_completion",   4096, 4),
    ("short_reasoning_plain",   4096, 2),
    ("long_memory_archive",    16384, 4),
    ("long_code_audit",        16384, 4),
]


def hex_bytes(values):
    return "".join(f"{int(b):02x}" for b in values)


def capture_case(ds4_bin: Path, root: Path, prompt_id: str, ctx: int, steps: int,
                 lock_file: str) -> dict:
    prompt_path = root / "prompts" / f"{prompt_id}.txt"
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ds4-vec-{prompt_id}-"))
    out_path = tmp_dir / "logprobs.json"
    env = os.environ.copy()
    env["DS4_METAL_PREFILL_CHUNK"] = "2048"
    env["DS4_METAL_DISABLE_METAL4"] = "1"
    env["DS4_LOCK_FILE"] = lock_file
    cmd = [
        str(ds4_bin),
        "--metal",
        "--system", "",
        "--prompt-file", str(prompt_path),
        "--ctx", str(ctx),
        "-n", str(steps),
        "--temp", "0",
        "--nothink",
        "--logprobs-top-k", "20",
        "--dump-logprobs", str(out_path),
    ]
    print(f"-> {prompt_id} ctx={ctx} steps={steps}", file=sys.stderr)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"ds4 failed for {prompt_id} (exit {proc.returncode})")
    with out_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return data


def build_vec(records, root: Path) -> str:
    lines = [
        "# ds4-local-cyberneurova-abliterated-logprob-vectors-v2",
        "# case <id> <ctx> <steps> <prompt-file>",
        "# step <index> <selected-hex> <top-count>",
        "# top <token-hex> <local-logprob>",
        "",
    ]
    for prompt_id, ctx, steps, dump in records:
        prompt_rel = f"tests/test-vectors/prompts/{prompt_id}.txt"
        actual_steps = len(dump["steps"])
        if actual_steps < steps:
            raise SystemExit(
                f"{prompt_id}: expected {steps} steps, ds4 produced {actual_steps}"
            )
        lines.append(f"case {prompt_id} {ctx} {steps} {prompt_rel}")
        for i in range(steps):
            step = dump["steps"][i]
            selected_hex = hex_bytes(step["selected"]["bytes"])
            top = [
                (hex_bytes(t["token"]["bytes"]), float(t["logprob"]))
                for t in step["top_logprobs"]
                if t["token"]["bytes"]
            ]
            lines.append(f"step {i} {selected_hex} {len(top)}")
            for token_hex, lp in top:
                lines.append(f"top {token_hex} {lp:.9g}")
        lines.append("end")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds4", default=str(here.parent.parent / "ds4"),
                        help="path to ds4 binary")
    parser.add_argument("--out", default=str(here / "official.vec"),
                        help="output vec file path")
    parser.add_argument("--only", action="append",
                        help="capture only the named prompt id (repeatable)")
    parser.add_argument("--lock-file", default="/tmp/ds4-regen-vectors.lock",
                        help="DS4_LOCK_FILE override so a running ds4-server does not block")
    args = parser.parse_args()

    ds4_bin = Path(args.ds4)
    if not ds4_bin.exists():
        raise SystemExit(f"missing ds4 binary at {ds4_bin}")

    selected = set(args.only) if args.only else None
    records = []
    for prompt_id, ctx, steps in CASES:
        if selected and prompt_id not in selected:
            continue
        dump = capture_case(ds4_bin, here, prompt_id, ctx, steps, args.lock_file)
        records.append((prompt_id, ctx, steps, dump))

    if not records:
        raise SystemExit("no cases captured")

    vec_text = build_vec(records, here)
    Path(args.out).write_text(vec_text, encoding="ascii")
    print(f"wrote {args.out} ({len(records)} cases)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
