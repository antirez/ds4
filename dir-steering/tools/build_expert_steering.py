#!/usr/bin/env python3
"""Build a DS4 expert-routing steering map from paired prompt sets.

The extractor captures routed expert ids with DS4's existing Metal graph dump
hook, compares activation rates between paired prompt files, and writes a flat
43 x 256 f32 map. Positive entries promote experts associated with the first
file; negative entries suppress experts associated with the second file.
"""

import argparse
import array
import json
import os
import subprocess
import tempfile
from pathlib import Path


N_LAYER = 43
N_EXPERT = 256
N_EXPERT_USED = 6
N_HASH_LAYER = 3

SPECIALS = {
    "bos": "<｜begin▁of▁sentence｜>",
    "user": "<｜User｜>",
    "assistant": "<｜Assistant｜>",
    "think": "<think>",
    "nothink": "</think>",
}


def read_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def render_ds4_prompt(system: str, user: str, think: bool, raw: bool) -> str:
    if raw:
        return user
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


def parse_layers(text: str) -> set[int]:
    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    bad = [layer for layer in layers if layer < 0 or layer >= N_LAYER]
    if bad:
        raise SystemExit(f"invalid layer indexes: {bad}")
    return layers


def run_capture(
    ds4: Path,
    model: Path,
    prompt: str,
    system: str,
    think: bool,
    raw: bool,
    ctx: int,
    focus_last_tokens: int,
    work: Path,
) -> tuple[list[list[int]], int]:
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(render_ds4_prompt(system, prompt, think, raw), encoding="utf-8")
    dump_prefix = work / "dump"

    env = os.environ.copy()
    env["DS4_METAL_GRAPH_DUMP_PREFIX"] = str(dump_prefix)
    env["DS4_METAL_GRAPH_DUMP_NAME"] = "ffn_moe_topk"
    env["DS4_METAL_GRAPH_DUMP_POS"] = "0"

    cmd = [
        str(ds4),
        "-m", str(model),
        "--ctx", str(ctx),
        "--prompt-file", str(prompt_path),
        "-n", "1",
    ]
    proc = subprocess.run(cmd, cwd=ds4.parent, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-40:])
        raise RuntimeError(f"ds4 capture failed for {prompt_path}:\n{tail}")

    counts = [[0] * N_EXPERT for _ in range(N_LAYER)]
    token_count = 0
    for layer in range(N_LAYER):
        path = work / f"dump_ffn_moe_topk-{layer}_pos0.i32"
        data = array.array("i")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // data.itemsize)
        if len(data) == 0 or len(data) % N_EXPERT_USED != 0:
            raise RuntimeError(f"bad router dump shape for {path}: {len(data)} ints")
        n_rows = len(data) // N_EXPERT_USED
        start_row = max(0, n_rows - focus_last_tokens) if focus_last_tokens > 0 else 0
        if layer == 0:
            token_count = n_rows - start_row
        for row in range(start_row, n_rows):
            base = row * N_EXPERT_USED
            for slot in range(N_EXPERT_USED):
                expert = data[base + slot]
                if 0 <= expert < N_EXPERT:
                    counts[layer][expert] += 1
    return counts, token_count


def add_counts(dst: list[list[int]], src: list[list[int]]) -> None:
    for layer in range(N_LAYER):
        for expert, value in enumerate(src[layer]):
            dst[layer][expert] += value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 CLI")
    ap.add_argument("--model", default="ds4flash.gguf", help="GGUF model path")
    ap.add_argument("--good-file", required=True,
                    help="target behavior prompts, one per line")
    ap.add_argument("--bad-file", required=True,
                    help="contrast behavior prompts, one per line")
    ap.add_argument("--out", default="dir-steering/out/expert-steering.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--think", action="store_true",
                    help="capture after <think>; default captures direct answers")
    ap.add_argument("--raw", action="store_true",
                    help="prompt files are already rendered DS4 text")
    ap.add_argument("--focus-last-tokens", type=int, default=0,
                    help="count only the last N dumped prompt-token rows")
    ap.add_argument("--activate-per-layer", type=int, default=2,
                    help="most positive RD experts to promote per layer")
    ap.add_argument("--deactivate-per-layer", type=int, default=2,
                    help="most negative RD experts to suppress per layer")
    ap.add_argument("--min-rd", type=float, default=0.0,
                    help="minimum absolute risk difference required")
    ap.add_argument("--layers", default="3-42",
                    help="comma/range layer filter; default skips DS4 hash-routed layers 0-2")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    ds4 = Path(args.ds4).resolve()
    model = Path(args.model).resolve()
    good_prompts = read_prompt_file(Path(args.good_file))
    bad_prompts = read_prompt_file(Path(args.bad_file))
    n = min(len(good_prompts), len(bad_prompts))
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)
    good_prompts = good_prompts[:n]
    bad_prompts = bad_prompts[:n]
    layers = parse_layers(args.layers)

    good_counts = [[0] * N_EXPERT for _ in range(N_LAYER)]
    bad_counts = [[0] * N_EXPERT for _ in range(N_LAYER)]
    good_tokens = 0
    bad_tokens = 0

    with tempfile.TemporaryDirectory(prefix="ds4-expert-steer-") as td:
        root = Path(td)
        for i, (good, bad) in enumerate(zip(good_prompts, bad_prompts), 1):
            print(f"pair {i}/{n}", flush=True)
            gw = root / f"good-{i}"
            bw = root / f"bad-{i}"
            gw.mkdir()
            bw.mkdir()
            g_counts, g_tokens = run_capture(ds4, model, good, args.system, args.think,
                                             args.raw, args.ctx, args.focus_last_tokens, gw)
            b_counts, b_tokens = run_capture(ds4, model, bad, args.system, args.think,
                                             args.raw, args.ctx, args.focus_last_tokens, bw)
            add_counts(good_counts, g_counts)
            add_counts(bad_counts, b_counts)
            good_tokens += g_tokens
            bad_tokens += b_tokens

    if good_tokens <= 0 or bad_tokens <= 0:
        raise SystemExit("no router rows were captured")

    steer = [[0.0] * N_EXPERT for _ in range(N_LAYER)]
    layer_report = []
    for layer in range(N_LAYER):
        rd = []
        for expert in range(N_EXPERT):
            p_good = good_counts[layer][expert] / good_tokens
            p_bad = bad_counts[layer][expert] / bad_tokens
            rd.append((p_good - p_bad, expert, p_good, p_bad))

        activated = []
        deactivated = []
        if layer in layers:
            for delta, expert, p_good, p_bad in sorted(rd, reverse=True):
                if len(activated) >= args.activate_per_layer:
                    break
                if delta > 0.0 and abs(delta) >= args.min_rd:
                    steer[layer][expert] = 1.0
                    activated.append({"expert": expert, "rd": delta,
                                      "target_rate": p_good, "contrast_rate": p_bad})
            for delta, expert, p_good, p_bad in sorted(rd):
                if len(deactivated) >= args.deactivate_per_layer:
                    break
                if delta < 0.0 and abs(delta) >= args.min_rd:
                    steer[layer][expert] = -1.0
                    deactivated.append({"expert": expert, "rd": delta,
                                        "target_rate": p_good, "contrast_rate": p_bad})

        layer_report.append({
            "layer": layer,
            "enabled": layer in layers,
            "activate": activated,
            "deactivate": deactivated,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-expert-steering-v1",
        "shape": [N_LAYER, N_EXPERT],
        "method": "risk_difference_topk_router_activation",
        "runtime": "positive scale promotes +1 entries and suppresses -1 entries; negative scale reverses",
        "default_runtime_scale": 0.01,
        "target_file": str(Path(args.good_file)),
        "contrast_file": str(Path(args.bad_file)),
        "model": str(model),
        "pairs": n,
        "target_rows": good_tokens,
        "contrast_rows": bad_tokens,
        "layers": sorted(layers),
        "hash_layers": N_HASH_LAYER,
        "activate_per_layer": args.activate_per_layer,
        "deactivate_per_layer": args.deactivate_per_layer,
        "min_rd": args.min_rd,
        "focus_last_tokens": args.focus_last_tokens,
        "layer_report": layer_report,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for layer in steer:
        flat.extend(layer)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()