#!/usr/bin/env python3
"""Benchmark a prefill candidate and optionally run the quality drift gate.

This is intended for default-off Metal Tensor experiments. It compares:

  standard  -> ./ds4-bench -mt off
  tensor    -> ./ds4-bench -mt auto
  candidate -> ./ds4-bench -mt <candidate-mode> with --set-env overrides

Use --run-drift-gate before promotion. The drift gate reuses the same
candidate env overrides, so its "tensor" row is the candidate route.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchRun:
    name: str
    label: str
    mode_args: list[str]
    env: dict[str, str]


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


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return label or "candidate"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
    dry_run: bool,
) -> None:
    env_prefix = [f"{name}={value}" for name, value in sorted(env_overrides.items())]
    print("+", " ".join(env_prefix + cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed with exit {proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-8000:]}"
        )


def read_bench_csv(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: empty CSV")
        required = {"ctx_tokens", "prefill_tps", "gen_tps"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows: dict[int, dict[str, float]] = {}
        for row in reader:
            ctx = int(row["ctx_tokens"])
            rows[ctx] = {
                "prefill_tps": float(row["prefill_tps"]),
                "gen_tps": float(row["gen_tps"]),
            }
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    return rows


def summarize_repeats(
    csv_paths: dict[str, list[Path]],
    *,
    baseline_name: str,
    tensor_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    raw: dict[str, list[dict[int, dict[str, float]]]] = {
        name: [read_bench_csv(path) for path in paths]
        for name, paths in csv_paths.items()
    }
    context_sets = [
        set().union(*(run.keys() for run in repeats))
        for repeats in raw.values()
    ]
    contexts = sorted(set.intersection(*context_sets))
    if not contexts:
        raise SystemExit("benchmark CSVs have no shared ctx_tokens values")

    runs: dict[str, dict[str, Any]] = {}
    for name, repeats in raw.items():
        by_context: dict[str, Any] = {}
        for ctx in contexts:
            prefill = [run[ctx]["prefill_tps"] for run in repeats if ctx in run]
            gen = [run[ctx]["gen_tps"] for run in repeats if ctx in run]
            by_context[str(ctx)] = {
                "prefill_tps_median": statistics.median(prefill),
                "gen_tps_median": statistics.median(gen),
                "prefill_tps_values": prefill,
                "gen_tps_values": gen,
            }
        runs[name] = {"contexts": by_context}

    gains: dict[str, dict[str, Any]] = {}
    for other_name, base_name in (
        (tensor_name, baseline_name),
        (candidate_name, baseline_name),
        (candidate_name, tensor_name),
    ):
        pair = f"{other_name}_vs_{base_name}"
        gains[pair] = {}
        for ctx in contexts:
            ctx_key = str(ctx)
            other = runs[other_name]["contexts"][ctx_key]
            base = runs[base_name]["contexts"][ctx_key]
            base_prefill = base["prefill_tps_median"]
            base_gen = base["gen_tps_median"]
            gains[pair][ctx_key] = {
                "prefill_gain_pct": ((other["prefill_tps_median"] / base_prefill) - 1.0) * 100.0
                if base_prefill
                else 0.0,
                "gen_gain_pct": ((other["gen_tps_median"] / base_gen) - 1.0) * 100.0
                if base_gen
                else 0.0,
            }

    return {
        "contexts": contexts,
        "runs": runs,
        "gains": gains,
    }


def print_summary(summary: dict[str, Any], *, candidate_name: str) -> None:
    print("\nMedian speed summary")
    print("ctx standard_prefill tensor_prefill candidate_prefill candidate_vs_tensor candidate_gen_vs_tensor")
    gains = summary["gains"][f"{candidate_name}_vs_tensor"]
    for ctx in summary["contexts"]:
        ctx_key = str(ctx)
        standard = summary["runs"]["standard"]["contexts"][ctx_key]
        tensor = summary["runs"]["tensor"]["contexts"][ctx_key]
        candidate = summary["runs"][candidate_name]["contexts"][ctx_key]
        gain = gains[ctx_key]
        print(
            f"{ctx} "
            f"{standard['prefill_tps_median']:.2f} "
            f"{tensor['prefill_tps_median']:.2f} "
            f"{candidate['prefill_tps_median']:.2f} "
            f"{gain['prefill_gain_pct']:+.1f}% "
            f"{gain['gen_gain_pct']:+.1f}%"
        )


def run_benchmarks(args: argparse.Namespace, candidate_env: dict[str, str]) -> dict[str, list[Path]]:
    candidate_name = safe_label(args.candidate_label)
    if candidate_name in {"standard", "tensor"}:
        raise SystemExit("--candidate-label must not resolve to 'standard' or 'tensor'")
    runs = (
        BenchRun("standard", "Standard Metal", ["-mt", "off"], {}),
        BenchRun("tensor", "Tensor Metal", ["-mt", "auto"], {}),
        BenchRun(candidate_name, args.candidate_label, ["-mt", args.candidate_mode], candidate_env),
    )
    common_args = [
        "--prompt-file",
        str(args.prompt_file),
        "--ctx-start",
        str(args.ctx_start),
        "--ctx-max",
        str(args.ctx_max),
        "--step-mul",
        str(args.step_mul),
        "--gen-tokens",
        str(args.gen_tokens),
    ]
    if args.model:
        common_args[:0] = ["-m", str(args.model)]

    csv_paths: dict[str, list[Path]] = {run.name: [] for run in runs}
    for repeat in range(1, args.repeat + 1):
        repeat_dir = args.out_dir / f"repeat-{repeat}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        chart_inputs: list[Path] = []
        chart_labels: list[str] = []
        for run in runs:
            csv_path = repeat_dir / f"{run.name}.csv"
            csv_paths[run.name].append(csv_path)
            cmd = [str(args.ds4_bench)] + run.mode_args + common_args + ["--csv", str(csv_path)]
            print(f"\nrepeat {repeat}/{args.repeat}: {run.label} -> {csv_path}")
            run_command(cmd, cwd=args.repo_root, env_overrides=run.env, dry_run=args.dry_run)
            chart_inputs.append(csv_path)
            chart_labels.append(run.label)

        chart_path = repeat_dir / "prefill-candidate.png"
        compare_cmd = [
            str(args.python),
            "speed-bench/compare_bench.py",
            *[str(path) for path in chart_inputs],
            "--labels",
            *chart_labels,
            "--title",
            f"Prefill candidate: {args.candidate_label} (repeat {repeat})",
            "-o",
            str(chart_path),
        ]
        run_command(compare_cmd, cwd=args.repo_root, env_overrides={}, dry_run=args.dry_run)

    return csv_paths


def run_drift_gate(args: argparse.Namespace, candidate_env: dict[str, str]) -> Path:
    gate_dir = args.out_dir / "quality-drift-gate"
    cmd = [
        str(args.python),
        "speed-bench/run_quality_drift_gate.py",
        "--repo-root",
        str(args.repo_root),
        "--ds4",
        str(args.ds4),
        "--out-dir",
        str(gate_dir),
    ]
    if args.model:
        cmd += ["--model", str(args.model)]
    if args.fail_on_quality_greedy:
        cmd.append("--fail-on-quality-greedy")
    for name, value in sorted(candidate_env.items()):
        cmd += ["--set-env", f"{name}={value}"]
    run_command(cmd, cwd=args.repo_root, env_overrides={}, dry_run=args.dry_run)
    return gate_dir / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ds4-bench", type=Path, default=Path("./ds4-bench"))
    parser.add_argument("--ds4", type=Path, default=Path("./ds4"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt-file", type=Path, default=Path("speed-bench/promessi_sposi.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ds4-prefill-candidate"))
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--candidate-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--ctx-start", type=int, default=512)
    parser.add_argument("--ctx-max", type=int, default=8192)
    parser.add_argument("--step-mul", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set an environment variable only for the candidate bench and drift gate.",
    )
    parser.add_argument("--run-drift-gate", action="store_true")
    parser.add_argument("--fail-on-quality-greedy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    args.repo_root = args.repo_root.resolve()
    if not args.ds4_bench.is_absolute():
        args.ds4_bench = args.repo_root / args.ds4_bench
    if not args.ds4.is_absolute():
        args.ds4 = args.repo_root / args.ds4
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate_env = parse_env_overrides(args.set_env)
    candidate_name = safe_label(args.candidate_label)
    if candidate_name in {"standard", "tensor"}:
        raise SystemExit("--candidate-label must not resolve to 'standard' or 'tensor'")
    csv_paths = run_benchmarks(args, candidate_env)

    payload: dict[str, Any] = {
        "candidate_label": args.candidate_label,
        "candidate_name": candidate_name,
        "candidate_mode": args.candidate_mode,
        "candidate_env": candidate_env,
        "csv_paths": {name: [str(path) for path in paths] for name, paths in csv_paths.items()},
    }
    if not args.dry_run:
        speed_summary = summarize_repeats(
            csv_paths,
            baseline_name="standard",
            tensor_name="tensor",
            candidate_name=candidate_name,
        )
        payload["speed_summary"] = speed_summary
        print_summary(speed_summary, candidate_name=candidate_name)

    if args.run_drift_gate:
        gate_summary = run_drift_gate(args, candidate_env)
        payload["quality_drift_gate_summary"] = str(gate_summary)

    summary_path = args.out_dir / "prefill-candidate-summary.json"
    if not args.dry_run:
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
            fp.write("\n")
        print(f"\nWrote {summary_path}")
    else:
        print(f"\nDry run only; would write {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
