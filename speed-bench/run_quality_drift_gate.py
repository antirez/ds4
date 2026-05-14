#!/usr/bin/env python3
"""Run the five-fixture Metal quality drift gate.

The gate captures first-token full logits and 16-token greedy continuations for
three modes:

  quality   -> --metal --quality
  standard  -> --metal -mt off
  tensor    -> --metal -mt auto

It reports:

  standard_vs_quality
  tensor_vs_quality
  tensor_vs_standard

The third comparison isolates the Tensor-route delta. The first two show
whether Tensor Metal is materially worse than the existing non-quality Metal
path when both are judged against --quality.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compare_logit_drift import compare, load_dump


@dataclass(frozen=True)
class Case:
    case_id: str
    ctx: int
    prompt_path: str


CASES = (
    Case("short_italian_fact", 16384, "tests/test-vectors/prompts/short_italian_fact.txt"),
    Case("short_code_completion", 4096, "tests/test-vectors/prompts/short_code_completion.txt"),
    Case("short_reasoning_plain", 4096, "tests/test-vectors/prompts/short_reasoning_plain.txt"),
    Case("long_memory_archive", 16384, "tests/test-vectors/prompts/long_memory_archive.txt"),
    Case("long_code_audit", 16384, "tests/test-vectors/prompts/long_code_audit.txt"),
)

MODES: dict[str, list[str]] = {
    "quality": ["--quality"],
    "standard": ["-mt", "off"],
    "tensor": ["-mt", "auto"],
}

PAIRS = (
    ("standard_vs_quality", "quality", "standard"),
    ("tensor_vs_quality", "quality", "tensor"),
    ("tensor_vs_standard", "standard", "tensor"),
)


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed with exit {proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-8000:]}"
        )


def dump_paths(out_dir: Path, case: Case, mode: str) -> tuple[Path, Path]:
    stem = f"{case.case_id}.{mode}"
    return out_dir / f"{stem}.logits.json", out_dir / f"{stem}.logprobs.json"


def ds4_base_cmd(args: argparse.Namespace, case: Case) -> list[str]:
    cmd = [
        str(args.ds4),
        "--metal",
        "--temp",
        "0",
        "--nothink",
        "--system",
        "",
        "-c",
        str(case.ctx),
        "--prompt-file",
        case.prompt_path,
    ]
    if args.model:
        cmd[1:1] = ["-m", str(args.model)]
    return cmd


def capture_case(args: argparse.Namespace, case: Case, mode: str) -> None:
    logits_path, logprobs_path = dump_paths(args.out_dir, case, mode)
    mode_args = MODES[mode]
    base = ds4_base_cmd(args, case)

    if not args.reuse or not logits_path.exists():
        run_command(
            base + mode_args + ["--dump-logits", str(logits_path)],
            cwd=args.repo_root,
            dry_run=args.dry_run,
        )

    if not args.reuse or not logprobs_path.exists():
        run_command(
            base
            + mode_args
            + [
                "-n",
                str(args.greedy_tokens),
                "--dump-logprobs",
                str(logprobs_path),
                "--logprobs-top-k",
                str(args.top_k),
            ],
            cwd=args.repo_root,
            dry_run=args.dry_run,
        )


def selected_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [int(step["selected"]["id"]) for step in data.get("steps", [])]


def greedy_diff(ref_path: Path, cand_path: Path) -> dict[str, Any]:
    ref = selected_ids(ref_path)
    cand = selected_ids(cand_path)
    first_diff = None
    for i, (ref_id, cand_id) in enumerate(zip(ref, cand)):
        if ref_id != cand_id:
            first_diff = i
            break
    if first_diff is None and len(ref) != len(cand):
        first_diff = min(len(ref), len(cand))
    return {
        "same": first_diff is None,
        "first_diff": first_diff,
        "ref_tokens": ref,
        "cand_tokens": cand,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "top1_mismatches": sum(0 if row["same_top1"] else 1 for row in rows),
        "greedy_mismatches": sum(0 if row["greedy_same"] else 1 for row in rows),
        "min_top5_overlap": min(row["top5_overlap"] for row in rows),
        "min_top20_overlap": min(row["top20_overlap"] for row in rows),
        "worst_rank_delta": max(row["max_rank_delta"] for row in rows),
        "worst_rms": max(row["rms"] for row in rows),
        "worst_max_abs": max(row["max_abs"] for row in rows),
        "worst_top20_max_abs": max(row["top20_max_abs"] for row in rows),
    }


def print_pair_table(pair_name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{pair_name}")
    print("case same_top1 top5 top20 rank rms max_abs top20_abs greedy")
    for row in rows:
        greedy = "same" if row["greedy_same"] else f"diff@{row['greedy_first_diff']}"
        print(
            f"{row['case']} "
            f"{'yes' if row['same_top1'] else 'no'} "
            f"{row['top5_overlap']}/5 "
            f"{row['top20_overlap']}/20 "
            f"{row['max_rank_delta']} "
            f"{row['rms']:.6g} "
            f"{row['max_abs']:.6g} "
            f"{row['top20_max_abs']:.6g} "
            f"{greedy}"
        )
    summary = aggregate(rows)
    print(
        "summary "
        f"top1_mismatches={summary['top1_mismatches']} "
        f"greedy_mismatches={summary['greedy_mismatches']} "
        f"min_top20={summary['min_top20_overlap']}/20 "
        f"worst_rms={summary['worst_rms']:.6g} "
        f"worst_top20_max_abs={summary['worst_top20_max_abs']:.6g}"
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for pair_name, ref_mode, cand_mode in PAIRS:
        rows: list[dict[str, Any]] = []
        for case in CASES:
            ref_logits, ref_logprobs = dump_paths(args.out_dir, case, ref_mode)
            cand_logits, cand_logprobs = dump_paths(args.out_dir, case, cand_mode)
            metrics = compare(load_dump(ref_logits), load_dump(cand_logits), args.top_k)
            greedy = greedy_diff(ref_logprobs, cand_logprobs)
            row = {
                "case": case.case_id,
                "ctx": case.ctx,
                **metrics,
                "greedy_same": greedy["same"],
                "greedy_first_diff": greedy["first_diff"],
                "greedy_ref_tokens": greedy["ref_tokens"],
                "greedy_cand_tokens": greedy["cand_tokens"],
            }
            rows.append(row)
        pairs[pair_name] = {
            "rows": rows,
            "summary": aggregate(rows),
        }
        print_pair_table(pair_name, rows)
    return {
        "cases": [case.__dict__ for case in CASES],
        "modes": MODES,
        "pairs": pairs,
    }


def check_gate(payload: dict[str, Any], *, fail_on_quality_greedy: bool) -> list[str]:
    failures: list[str] = []
    for pair_name in ("standard_vs_quality", "tensor_vs_quality"):
        summary = payload["pairs"][pair_name]["summary"]
        if summary["top1_mismatches"] != 0:
            failures.append(f"{pair_name}: top1_mismatches={summary['top1_mismatches']}")
        if fail_on_quality_greedy and summary["greedy_mismatches"] != 0:
            failures.append(f"{pair_name}: greedy_mismatches={summary['greedy_mismatches']}")

    tensor_delta = payload["pairs"]["tensor_vs_standard"]["summary"]
    if tensor_delta["top1_mismatches"] != 0:
        failures.append(
            f"tensor_vs_standard: top1_mismatches={tensor_delta['top1_mismatches']}"
        )
    if tensor_delta["greedy_mismatches"] != 0:
        failures.append(
            f"tensor_vs_standard: greedy_mismatches={tensor_delta['greedy_mismatches']}"
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


def apply_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--set-env expects NAME=VALUE, got: {value}")
        name, env_value = value.split("=", 1)
        if not name:
            raise SystemExit(f"--set-env expects NAME=VALUE, got: {value}")
        overrides[name] = env_value
    for name, value in overrides.items():
        os.environ[name] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ds4", type=Path, default=Path("./ds4"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ds4-quality-drift-gate"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--greedy-tokens", type=int, default=16)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing dumps in --out-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set an environment variable for all ds4 captures; repeatable.",
    )
    parser.add_argument(
        "--fail-on-quality-greedy",
        action="store_true",
        help="Fail when standard/tensor differs from --quality in greedy continuation.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after reporting gate failures.",
    )
    args = parser.parse_args()

    if args.top_k < 20:
        raise SystemExit("--top-k must be at least 20")

    args.repo_root = args.repo_root.resolve()
    if not args.ds4.is_absolute():
        args.ds4 = args.repo_root / args.ds4
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = apply_env_overrides(args.set_env)

    for case in CASES:
        for mode in MODES:
            capture_case(args, case, mode)

    if args.dry_run:
        return 0

    payload = summarize(args)
    payload["env"] = env_overrides
    payload["gate_failures"] = check_gate(
        payload,
        fail_on_quality_greedy=args.fail_on_quality_greedy,
    )
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"\nWrote {summary_path}")

    if payload["gate_failures"]:
        print("\nGate failures:")
        for failure in payload["gate_failures"]:
            print(f"  {failure}")
        return 0 if args.no_fail else 1
    print("\nGate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
