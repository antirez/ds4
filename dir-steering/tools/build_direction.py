#!/usr/bin/env python3
"""Build a DS4 directional-steering vector from paired prompt sets.

The extractor asks ds4 to dump one 4096-wide activation row per layer, averages
the target and control rows, and writes a flat f32 file with 43 layer vectors.
At runtime ds4 applies:

    y = y - scale * direction[layer] * dot(direction[layer], y)

Positive scale suppresses the target direction.  Negative scale amplifies it.
"""

import argparse
import array
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


N_LAYER = 43
N_EMBD = 4096

SPECIALS = {
    "bos": "<｜begin▁of▁sentence｜>",
    "user": "<｜User｜>",
    "assistant": "<｜Assistant｜>",
    "think": "<think>",
    "nothink": "</think>",
}


def read_prompt_file(path: Path) -> list[str]:
    """Read one prompt per non-empty line, ignoring shell-style comments."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def render_ds4_prompt(system: str, user: str, think: bool) -> str:
    """Render the minimal DS4 chat prefix used for activation capture."""
    pieces = [SPECIALS["bos"]]
    if system:
        pieces.append(system)
    pieces += [
        SPECIALS["user"],
        user,
        SPECIALS["assistant"],
        SPECIALS["think"] if think else SPECIALS["nothink"],
    ]
    return "".join(pieces)


def normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def run_capture(
    ds4: Path,
    model: Path,
    prompt: str,
    system: str,
    think: bool,
    ctx: int,
    component: str,
    work: Path,
) -> list[list[float]]:
    """Run ds4 once and return the last prompt-row dump for every layer."""
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(render_ds4_prompt(system, prompt, think), encoding="utf-8")
    dump_prefix = work / "dump"

    env = os.environ.copy()
    env["PULSAR_CUDA_GRAPH_DUMP_PREFIX"] = str(dump_prefix)
    env["PULSAR_CUDA_GRAPH_DUMP_NAME"] = component
    env["PULSAR_CUDA_GRAPH_DUMP_POS"] = "0"

    cmd = [
        str(ds4),
        "-m", str(model),
        "--ctx", str(ctx),
        "--prompt-file", str(prompt_path),
        "-n", "1",
    ]
    subprocess.run(cmd, cwd=ds4.parent, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    rows: list[list[float]] = []
    for layer in range(N_LAYER):
        path = work / f"dump_{component}-{layer}_pos0.bin"
        data = array.array("f")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // 4)
        if len(data) < N_EMBD or len(data) % N_EMBD != 0:
            raise RuntimeError(f"bad dump shape for {path}: {len(data)} floats")
        rows.append(list(data[-N_EMBD:]))
    return rows


def add_rows(total: list[list[float]], rows: list[list[float]]) -> None:
    for layer in range(N_LAYER):
        dst = total[layer]
        src = rows[layer]
        for i, value in enumerate(src):
            dst[i] += value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 CLI")
    ap.add_argument("--model", default="ds4flash.gguf", help="GGUF model path")
    ap.add_argument("--good-file", required=True,
                    help="desired/target prompts, one per line")
    ap.add_argument("--bad-file", required=True,
                    help="contrast/control prompts, one per line")
    ap.add_argument("--out", default="dir-steering/out/direction.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="ffn_out",
                    choices=("ffn_out", "attn_out"),
                    help="runtime-editable 4096-wide activation stream")
    ap.add_argument("--think", action="store_true",
                    help="capture after <think>; default captures direct answers")
    ap.add_argument("--pair-normalize", action="store_true",
                    help="average normalized per-pair differences")
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="do not remove the component parallel to the control mean")
    args = ap.parse_args()

    ds4 = Path(args.ds4).resolve()
    model = Path(args.model).resolve()
    good_prompts = read_prompt_file(Path(args.good_file))
    bad_prompts = read_prompt_file(Path(args.bad_file))
    n = min(len(good_prompts), len(bad_prompts))
    good_prompts = good_prompts[:n]
    bad_prompts = bad_prompts[:n]

    good_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    bad_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]
    pair_sum = [[0.0] * N_EMBD for _ in range(N_LAYER)]

    with tempfile.TemporaryDirectory(prefix="ds4-dir-steer-") as td:
        root = Path(td)
        for i, (good, bad) in enumerate(zip(good_prompts, bad_prompts), 1):
            print(f"pair {i}/{n}", flush=True)
            gw = root / f"good-{i}"
            bw = root / f"bad-{i}"
            gw.mkdir()
            bw.mkdir()
            good_rows = run_capture(ds4, model, good, args.system, args.think,
                                    args.ctx, args.component, gw)
            bad_rows = run_capture(ds4, model, bad, args.system, args.think,
                                   args.ctx, args.component, bw)
            add_rows(good_sum, good_rows)
            add_rows(bad_sum, bad_rows)
            if args.pair_normalize:
                for layer in range(N_LAYER):
                    diff = normalize([
                        good_rows[layer][j] - bad_rows[layer][j]
                        for j in range(N_EMBD)
                    ])
                    for j, value in enumerate(diff):
                        pair_sum[layer][j] += value

    layers = []
    for layer in range(N_LAYER):
        good_mean = [x / n for x in good_sum[layer]]
        bad_mean = [x / n for x in bad_sum[layer]]
        if args.pair_normalize:
            direction = normalize([x / n for x in pair_sum[layer]])
        else:
            direction = normalize([
                good_mean[i] - bad_mean[i]
                for i in range(N_EMBD)
            ])
        if not args.no_orthogonalize:
            base = normalize(bad_mean)
            projection = dot(direction, base)
            direction = normalize([
                direction[i] - projection * base[i]
                for i in range(N_EMBD)
            ])
        layers.append(direction)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-directional-steering-v1",
        "shape": [N_LAYER, N_EMBD],
        "component": args.component,
        "thinking": bool(args.think),
        "pair_normalize": bool(args.pair_normalize),
        "orthogonalize_control_mean": not args.no_orthogonalize,
        "good_file": str(Path(args.good_file)),
        "bad_file": str(Path(args.bad_file)),
        "model": str(model),
        "note": "runtime positive scale suppresses this direction; negative scale amplifies it",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for direction in layers:
        flat.extend(direction)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()
