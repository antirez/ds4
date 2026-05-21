#!/usr/bin/env python3
"""Run a resumed-prefill frontier logit drift gate.

The normal five-fixture quality gate captures logits after a cold prompt
prefill.  Candidates that route only nonzero prefill positions need another
check: grow one long prompt through the same frontiers as ds4-bench, dump logits
after each resumed frontier, and compare:

  standard_vs_quality
  tensor_vs_quality
  tensor_vs_standard

When tensor-mode environment overrides are supplied, the gate also captures the
plain no-env Tensor baseline as default_tensor and compares:

  default_tensor_vs_quality
  default_tensor_vs_standard
  tensor_vs_default_tensor
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from compare_logit_drift import compare, load_dump
from metal_tensor_presets import CANDIDATE_PRESETS, preset_help


MODES: dict[str, list[str]] = {
    "quality": ["--quality"],
    "standard": ["-mt", "off"],
    "default_tensor": ["-mt", "auto"],
    "tensor": ["-mt", "auto"],
}

BASE_PAIRS = (
    ("standard_vs_quality", "quality", "standard"),
    ("tensor_vs_quality", "quality", "tensor"),
    ("tensor_vs_standard", "standard", "tensor"),
)

DEFAULT_TENSOR_PAIRS = (
    ("default_tensor_vs_quality", "quality", "default_tensor"),
    ("default_tensor_vs_standard", "standard", "default_tensor"),
    ("tensor_vs_default_tensor", "default_tensor", "tensor"),
)

DS4_BENCH_FRESHNESS_SOURCES = (
    "ds4.c",
    "ds4.h",
    "ds4_gpu.h",
    "ds4_bench.c",
    "ds4_metal.m",
    "metal/*.metal",
)


def assert_fresh_binary(
    binary: Path,
    *,
    repo_root: Path,
    source_patterns: tuple[str, ...],
    allow_stale: bool,
) -> None:
    if allow_stale:
        return
    if not binary.exists():
        raise SystemExit(f"{binary}: binary does not exist; run the relevant make target first")
    binary_mtime = binary.stat().st_mtime
    stale_sources: list[Path] = []
    for pattern in source_patterns:
        matches = sorted(repo_root.glob(pattern))
        if not matches:
            continue
        stale_sources.extend(path for path in matches if path.stat().st_mtime > binary_mtime)
    if stale_sources:
        newest = max(stale_sources, key=lambda path: path.stat().st_mtime)
        rel = newest.relative_to(repo_root)
        raise SystemExit(
            f"{binary}: stale binary; {rel} is newer. "
            "Rebuild before running the chunked drift gate, or pass "
            "--allow-stale-binary only when intentionally summarizing old artifacts."
        )


def shell_join(argv: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--set-env expects NAME=VALUE, got: {value}")
        name, env_value = value.split("=", 1)
        if not name:
            raise SystemExit(f"--set-env expects NAME=VALUE, got: {value}")
        env[name] = env_value
    return env


def candidate_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.preset:
        env.update(CANDIDATE_PRESETS[args.preset].env)
    env.update(parse_env_overrides(args.set_env))
    return env


def active_modes(capture_default_tensor: bool) -> list[str]:
    if capture_default_tensor:
        return ["quality", "standard", "default_tensor", "tensor"]
    return ["quality", "standard", "tensor"]


def active_pairs(capture_default_tensor: bool) -> list[tuple[str, str, str]]:
    pairs = list(BASE_PAIRS)
    if capture_default_tensor:
        pairs.extend(DEFAULT_TENSOR_PAIRS)
    return pairs


def mode_dir(out_dir: Path, mode: str) -> Path:
    return out_dir / f"{mode}-frontier-logits"


def mode_csv(out_dir: Path, mode: str) -> Path:
    return out_dir / f"{mode}.csv"


def frontier_logits_path(out_dir: Path, mode: str, frontier: int) -> Path:
    return mode_dir(out_dir, mode) / f"frontier_{frontier:06d}.logits.json"


def run_command(
    cmd: list[object],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
    dry_run: bool,
) -> None:
    printable = [str(part) for part in cmd]
    if env_overrides:
        env_text = " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(env_overrides.items()))
        print("+", env_text, shell_join(printable), flush=True)
    else:
        print("+", shell_join(printable), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(printable, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed with exit {proc.returncode}: {shell_join(printable)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-8000:]}"
        )


def capture_mode(
    args: argparse.Namespace,
    mode: str,
    *,
    tensor_env: dict[str, str],
) -> None:
    dump_dir = mode_dir(args.out_dir, mode)
    dump_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse and all(frontier_logits_path(args.out_dir, mode, f).exists() for f in args.frontiers):
        print(f"Reusing {mode} frontier dumps in {dump_dir}", flush=True)
        return

    mode_env = tensor_env if mode == "tensor" else {}
    cmd: list[object] = [
        args.ds4_bench,
        "--prompt-file",
        args.prompt_file,
        "--ctx-start",
        args.ctx_start,
        "--ctx-max",
        args.ctx_max,
        "--step-mul",
        args.step_mul,
        "--gen-tokens",
        args.gen_tokens,
        "--dump-frontier-logits-dir",
        dump_dir,
        "--csv",
        mode_csv(args.out_dir, mode),
    ]
    if args.model:
        cmd[1:1] = ["-m", args.model]
    cmd.extend(MODES[mode])
    run_command(cmd, cwd=args.repo_root, env_overrides=mode_env, dry_run=args.dry_run)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frontiers": len(rows),
        "top1_mismatches": sum(0 if row["same_top1"] else 1 for row in rows),
        "min_top5_overlap": min(row["top5_overlap"] for row in rows),
        "min_top20_overlap": min(row["top20_overlap"] for row in rows),
        "worst_rank_delta": max(row["max_rank_delta"] for row in rows),
        "worst_rms": max(row["rms"] for row in rows),
        "worst_max_abs": max(row["max_abs"] for row in rows),
        "worst_top20_max_abs": max(row["top20_max_abs"] for row in rows),
    }


def extrema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    worst_rms = max(rows, key=lambda row: row["rms"])
    worst_top20 = max(rows, key=lambda row: row["top20_max_abs"])
    min_top20 = min(rows, key=lambda row: row["top20_overlap"])
    return {
        "worst_rms_frontier": worst_rms["frontier"],
        "worst_rms": worst_rms["rms"],
        "worst_top20_max_abs_frontier": worst_top20["frontier"],
        "worst_top20_max_abs": worst_top20["top20_max_abs"],
        "min_top20_overlap_frontier": min_top20["frontier"],
        "min_top20_overlap": min_top20["top20_overlap"],
        "top1_mismatch_frontiers": [row["frontier"] for row in rows if not row["same_top1"]],
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for pair_name, ref_mode, cand_mode in args.pairs:
        rows: list[dict[str, Any]] = []
        for frontier in args.frontiers:
            ref_path = frontier_logits_path(args.out_dir, ref_mode, frontier)
            cand_path = frontier_logits_path(args.out_dir, cand_mode, frontier)
            metrics = compare(load_dump(ref_path), load_dump(cand_path), args.top_k)
            rows.append({"frontier": frontier, **metrics})
        pairs[pair_name] = {
            "rows": rows,
            "summary": aggregate(rows),
            "extrema": extrema(rows),
        }
        print_pair_table(pair_name, rows)
    return {
        "pairs": pairs,
        "modes": {mode: MODES[mode] for mode in args.modes},
        "pair_order": [pair_name for pair_name, _, _ in args.pairs],
        "frontiers": args.frontiers,
    }


def print_pair_table(pair_name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{pair_name}")
    print("frontier same_top1 top5 top20 rank rms max_abs top20_abs")
    for row in rows:
        print(
            f"{row['frontier']} "
            f"{'yes' if row['same_top1'] else 'no'} "
            f"{row['top5_overlap']}/5 "
            f"{row['top20_overlap']}/20 "
            f"{row['max_rank_delta']} "
            f"{row['rms']:.6g} "
            f"{row['max_abs']:.6g} "
            f"{row['top20_max_abs']:.6g}"
        )
    summary = aggregate(rows)
    print(
        "summary "
        f"top1_mismatches={summary['top1_mismatches']} "
        f"min_top20={summary['min_top20_overlap']}/20 "
        f"worst_rms={summary['worst_rms']:.6g} "
        f"worst_top20_max_abs={summary['worst_top20_max_abs']:.6g}"
    )


def check_gate(
    payload: dict[str, Any],
    *,
    max_tensor_standard_rms: float | None,
    max_tensor_standard_top20_abs: float | None,
    max_tensor_default_rms: float | None,
    max_tensor_default_top20_abs: float | None,
) -> list[str]:
    failures: list[str] = []
    for pair_name in payload.get("pair_order", ("standard_vs_quality", "tensor_vs_quality", "tensor_vs_standard")):
        summary = payload["pairs"][pair_name]["summary"]
        if summary["top1_mismatches"] != 0:
            failures.append(f"{pair_name}: top1_mismatches={summary['top1_mismatches']}")

    tensor_delta = payload["pairs"]["tensor_vs_standard"]["summary"]
    tensor_extrema = payload["pairs"]["tensor_vs_standard"]["extrema"]
    if max_tensor_standard_rms is not None and tensor_delta["worst_rms"] > max_tensor_standard_rms:
        failures.append(
            "tensor_vs_standard: worst_rms exceeds configured envelope "
            f"({tensor_delta['worst_rms']:.6g} > {max_tensor_standard_rms:.6g}, "
            f"frontier={tensor_extrema['worst_rms_frontier']})"
        )
    if (max_tensor_standard_top20_abs is not None and
            tensor_delta["worst_top20_max_abs"] > max_tensor_standard_top20_abs):
        failures.append(
            "tensor_vs_standard: worst_top20_max_abs exceeds configured envelope "
            f"({tensor_delta['worst_top20_max_abs']:.6g} > "
            f"{max_tensor_standard_top20_abs:.6g}, "
            f"frontier={tensor_extrema['worst_top20_max_abs_frontier']})"
        )

    if "tensor_vs_default_tensor" in payload["pairs"]:
        default_delta = payload["pairs"]["tensor_vs_default_tensor"]["summary"]
        default_extrema = payload["pairs"]["tensor_vs_default_tensor"]["extrema"]
        if max_tensor_default_rms is not None and default_delta["worst_rms"] > max_tensor_default_rms:
            failures.append(
                "tensor_vs_default_tensor: worst_rms exceeds configured envelope "
                f"({default_delta['worst_rms']:.6g} > {max_tensor_default_rms:.6g}, "
                f"frontier={default_extrema['worst_rms_frontier']})"
            )
        if (max_tensor_default_top20_abs is not None and
                default_delta["worst_top20_max_abs"] > max_tensor_default_top20_abs):
            failures.append(
                "tensor_vs_default_tensor: worst_top20_max_abs exceeds configured envelope "
                f"({default_delta['worst_top20_max_abs']:.6g} > "
                f"{max_tensor_default_top20_abs:.6g}, "
                f"frontier={default_extrema['worst_top20_max_abs_frontier']})"
            )

    standard = payload["pairs"]["standard_vs_quality"]["summary"]
    tensor = payload["pairs"]["tensor_vs_quality"]["summary"]
    if tensor["worst_rms"] > standard["worst_rms"] * 1.10:
        failures.append(
            "tensor_vs_quality: worst_rms materially worse than standard "
            f"({tensor['worst_rms']:.6g} > {standard['worst_rms']:.6g} * 1.10)"
        )
    if tensor["worst_top20_max_abs"] > standard["worst_top20_max_abs"] * 1.10:
        failures.append(
            "tensor_vs_quality: worst_top20_max_abs materially worse than standard "
            f"({tensor['worst_top20_max_abs']:.6g} > "
            f"{standard['worst_top20_max_abs']:.6g} * 1.10)"
        )
    return failures


def markdown_pair_table(pair_name: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"## {markdown_escape(pair_name)}",
        "",
        "| Frontier | Same top1 | Top5 | Top20 | Rank delta | RMS | Max abs | Top20 abs |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['frontier']} | "
            f"{'yes' if row['same_top1'] else 'no'} | "
            f"{row['top5_overlap']}/5 | "
            f"{row['top20_overlap']}/20 | "
            f"{row['max_rank_delta']} | "
            f"{row['rms']:.6g} | "
            f"{row['max_abs']:.6g} | "
            f"{row['top20_max_abs']:.6g} |"
        )
    summary = aggregate(rows)
    row_extrema = extrema(rows)
    lines.extend(
        [
            "",
            "| Summary | Value |",
            "| --- | ---: |",
            f"| Top1 mismatches | {summary['top1_mismatches']} |",
            f"| Min top5 overlap | {summary['min_top5_overlap']}/5 |",
            f"| Min top20 overlap | {summary['min_top20_overlap']}/20 |",
            f"| Worst rank delta | {summary['worst_rank_delta']} |",
            f"| Worst RMS | {summary['worst_rms']:.6g} |",
            f"| Worst max abs | {summary['worst_max_abs']:.6g} |",
            f"| Worst top20 max abs | {summary['worst_top20_max_abs']:.6g} |",
            "",
            "| Worst frontier | Value |",
            "| --- | --- |",
            f"| Worst RMS frontier | {row_extrema['worst_rms_frontier']} "
            f"({row_extrema['worst_rms']:.6g}) |",
            f"| Worst top20 abs frontier | {row_extrema['worst_top20_max_abs_frontier']} "
            f"({row_extrema['worst_top20_max_abs']:.6g}) |",
            f"| Min top20 overlap frontier | {row_extrema['min_top20_overlap_frontier']} "
            f"({row_extrema['min_top20_overlap']}/20) |",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_summary(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Chunked Prefill Drift Gate",
        "",
        "This gate dumps logits after resumed `ds4_session_sync()` frontiers from one long prompt.",
        "",
        "Modes:",
        "",
    ]
    for mode, mode_args in payload["modes"].items():
        lines.append(f"- `{markdown_escape(mode)}`: `{' '.join(mode_args)}`")
    if payload["candidate_env"]:
        lines.extend(["", "Tensor-mode environment overrides:", ""])
        for name, value in sorted(payload["candidate_env"].items()):
            lines.append(f"- `{markdown_escape(name)}={markdown_escape(value)}`")
    else:
        lines.extend(["", "Tensor-mode environment overrides: none"])

    config = payload["run_config"]
    lines.extend(["", "Run config:", "", "| Setting | Value |", "| --- | --- |"])
    for key in (
        "repo_root",
        "ds4_bench",
        "model",
        "prompt_file",
        "out_dir",
        "candidate_preset",
        "ctx_start",
        "ctx_max",
        "step_mul",
        "gen_tokens",
        "top_k",
        "reuse",
        "max_tensor_standard_rms",
        "max_tensor_standard_top20_abs",
        "max_tensor_default_rms",
        "max_tensor_default_top20_abs",
        "capture_default_tensor",
    ):
        lines.append(f"| `{markdown_escape(key)}` | `{markdown_escape(config.get(key))}` |")
    lines.extend(["", "Replay command:", "", "```sh", shell_join(["python3", *config["argv"]]), "```"])

    envelope = payload.get("drift_envelope") or {}
    lines.extend(["", "Tensor-vs-standard drift envelope:", ""])
    if envelope.get("max_rms") is not None:
        lines.append(f"- Worst RMS <= `{envelope['max_rms']:.6g}`")
    if envelope.get("max_top20_abs") is not None:
        lines.append(f"- Worst top20 abs <= `{envelope['max_top20_abs']:.6g}`")
    if not envelope:
        lines.append("- not configured")
    default_envelope = payload.get("tensor_default_envelope") or {}
    if default_envelope:
        lines.extend(["", "Candidate-vs-default-Tensor drift envelope:", ""])
        if default_envelope.get("max_rms") is not None:
            lines.append(f"- Worst RMS <= `{default_envelope['max_rms']:.6g}`")
        if default_envelope.get("max_top20_abs") is not None:
            lines.append(f"- Worst top20 abs <= `{default_envelope['max_top20_abs']:.6g}`")

    failures = payload["gate_failures"]
    lines.extend(["", f"Gate: {'FAIL' if failures else 'OK'}", ""])
    if failures:
        lines.append("Failures:")
        lines.append("")
        for failure in failures:
            lines.append(f"- {markdown_escape(failure)}")
        lines.append("")

    for pair_name in payload.get("pair_order", list(payload["pairs"])):
        lines.append(markdown_pair_table(pair_name, payload["pairs"][pair_name]["rows"]))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "repo_root": str(args.repo_root),
        "ds4_bench": str(args.ds4_bench),
        "model": str(args.model) if args.model else None,
        "prompt_file": str(args.prompt_file),
        "out_dir": str(args.out_dir),
        "candidate_preset": args.preset,
        "ctx_start": args.ctx_start,
        "ctx_max": args.ctx_max,
        "step_mul": args.step_mul,
        "gen_tokens": args.gen_tokens,
        "top_k": args.top_k,
        "reuse": args.reuse,
        "dry_run": args.dry_run,
        "max_tensor_standard_rms": args.max_tensor_standard_rms,
        "max_tensor_standard_top20_abs": args.max_tensor_standard_top20_abs,
        "max_tensor_default_rms": args.max_tensor_default_rms,
        "max_tensor_default_top20_abs": args.max_tensor_default_top20_abs,
        "capture_default_tensor": args.capture_default_tensor,
        "allow_stale_binary": args.allow_stale_binary,
        "no_fail": args.no_fail,
    }


def compute_frontiers(ctx_start: int, ctx_max: int, step_mul: float) -> list[int]:
    frontiers: list[int] = []
    cur = ctx_start
    while True:
        frontiers.append(cur)
        if cur >= ctx_max:
            break
        next_value = int((cur * step_mul) + 0.999999)
        if next_value <= cur:
            next_value = cur + 1
        if next_value > ctx_max:
            next_value = ctx_max
        cur = next_value
    return frontiers


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Candidate presets:\n{preset_help()}",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ds4-bench", type=Path, default=Path("./ds4-bench"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt-file", type=Path, default=Path("speed-bench/promessi_sposi.txt"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ctx-start", type=int, default=512)
    parser.add_argument("--ctx-max", type=int, default=8192)
    parser.add_argument("--step-mul", type=float, default=2.0)
    parser.add_argument("--gen-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing frontier dumps in --out-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--allow-stale-binary",
        action="store_true",
        help="Skip the source-vs-binary freshness check.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CANDIDATE_PRESETS),
        help="Use a named default-off candidate environment preset for the tensor mode.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set an environment variable only for the tensor-mode capture; repeatable.",
    )
    parser.add_argument(
        "--max-tensor-standard-rms",
        type=float,
        help="Optional maximum Tensor-vs-standard worst RMS allowed by this gate.",
    )
    parser.add_argument(
        "--max-tensor-standard-top20-abs",
        type=float,
        help="Optional maximum Tensor-vs-standard worst top-20 absolute drift allowed by this gate.",
    )
    parser.add_argument(
        "--max-tensor-default-rms",
        type=float,
        help="Optional maximum candidate Tensor-vs-default Tensor worst RMS allowed by this gate.",
    )
    parser.add_argument(
        "--max-tensor-default-top20-abs",
        type=float,
        help="Optional maximum candidate Tensor-vs-default Tensor worst top-20 absolute drift allowed by this gate.",
    )
    parser.add_argument(
        "--no-default-tensor-baseline",
        action="store_true",
        help="Do not capture the no-env -mt auto baseline when tensor-mode env overrides are set.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after reporting gate failures.",
    )
    args = parser.parse_args()

    if args.top_k < 20:
        raise SystemExit("--top-k must be at least 20")
    if args.ctx_start <= 0 or args.ctx_max < args.ctx_start:
        raise SystemExit("--ctx-start must be positive and <= --ctx-max")
    if args.step_mul < 1.0:
        raise SystemExit("--step-mul must be >= 1")
    if args.gen_tokens <= 0:
        raise SystemExit("--gen-tokens must be positive")

    label = args.preset or "chunked-prefill-drift-gate"
    if args.out_dir is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        args.out_dir = Path("speed-bench/local-runs") / f"{run_id}-{safe_label(label)}-chunked-drift-gate"

    args.repo_root = args.repo_root.resolve()
    if not args.ds4_bench.is_absolute():
        args.ds4_bench = args.repo_root / args.ds4_bench
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        assert_fresh_binary(
            args.ds4_bench,
            repo_root=args.repo_root,
            source_patterns=DS4_BENCH_FRESHNESS_SOURCES,
            allow_stale=args.allow_stale_binary,
        )
    args.frontiers = compute_frontiers(args.ctx_start, args.ctx_max, args.step_mul)
    tensor_env = candidate_env(args)
    args.capture_default_tensor = bool(tensor_env) and not args.no_default_tensor_baseline
    args.modes = active_modes(args.capture_default_tensor)
    args.pairs = active_pairs(args.capture_default_tensor)

    if tensor_env:
        print("Tensor-mode environment overrides:", flush=True)
        for name, value in sorted(tensor_env.items()):
            print(f"  {name}={value}", flush=True)

    for mode in args.modes:
        capture_mode(args, mode, tensor_env=tensor_env)

    if args.dry_run:
        return 0

    payload = summarize(args)
    payload["candidate_env"] = tensor_env
    payload["run_config"] = build_run_config(args)
    envelope = {
        "max_rms": args.max_tensor_standard_rms,
        "max_top20_abs": args.max_tensor_standard_top20_abs,
    }
    if envelope["max_rms"] is not None or envelope["max_top20_abs"] is not None:
        payload["drift_envelope"] = envelope
    default_envelope = {
        "max_rms": args.max_tensor_default_rms,
        "max_top20_abs": args.max_tensor_default_top20_abs,
    }
    if default_envelope["max_rms"] is not None or default_envelope["max_top20_abs"] is not None:
        payload["tensor_default_envelope"] = default_envelope
    payload["gate_failures"] = check_gate(
        payload,
        max_tensor_standard_rms=args.max_tensor_standard_rms,
        max_tensor_standard_top20_abs=args.max_tensor_standard_top20_abs,
        max_tensor_default_rms=args.max_tensor_default_rms,
        max_tensor_default_top20_abs=args.max_tensor_default_top20_abs,
    )

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {summary_path}")

    markdown_path = args.out_dir / "summary.md"
    write_markdown_summary(payload, markdown_path)
    print(f"Wrote {markdown_path}")

    if payload["gate_failures"]:
        print("\nGate failures:")
        for failure in payload["gate_failures"]:
            print(f"  {failure}")
        return 0 if args.no_fail else 1
    print("\nGate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
