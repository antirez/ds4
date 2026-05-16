#!/usr/bin/env python3
"""Run a Metal Tensor local comparator probe and summarize the result.

This is a targeted diagnostic for default-off prefill candidates. It runs
`./ds4 --metal -mt auto` with DS4_METAL_MPP_COMPARE_* environment variables,
captures stderr/stdout under speed-bench/local-runs/, then writes a comparator
Markdown/JSON summary. It is not a replacement for the five-fixture drift gate;
use it to decide what to narrow before running run_quality_drift_gate.py.
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

from metal_tensor_presets import CANDIDATE_PRESETS, preset_help
from run_quality_drift_gate import CASES
from summarize_mpp_compare import as_json, merge_summaries, parse_log, render_markdown


CASE_BY_ID = {case.case_id: case for case in CASES}

DS4_FRESHNESS_SOURCES = (
    "ds4.c",
    "ds4.h",
    "ds4_gpu.h",
    "ds4_cli.c",
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
            "Rebuild before running the comparator probe, or pass "
            "--allow-stale-binary only when intentionally summarizing old artifacts."
        )


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
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "probe"


def shell_join(argv: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def normalize_routes(values: list[str]) -> list[str]:
    routes: list[str] = []
    for value in values or ["all"]:
        for route in value.replace("|", ",").split(","):
            route = route.strip()
            if route:
                routes.append(route)
    return routes or ["all"]


def probe_env(args: argparse.Namespace, route: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.preset:
        env.update(CANDIDATE_PRESETS[args.preset].env)
    env.update(parse_env_overrides(args.set_env))
    env["DS4_METAL_MPP_COMPARE_ROUTE"] = route
    env["DS4_METAL_MPP_COMPARE_MAX"] = str(args.compare_max)
    if route == "q8":
        env["DS4_METAL_Q8_COMPARE"] = "1"
        if args.q8_filter:
            env["DS4_METAL_Q8_COMPARE_FILTER"] = args.q8_filter
    if route == "flash_attn":
        env["DS4_METAL_FLASH_ATTN_COMPARE"] = "1"
        if args.flash_attn_filter:
            env["DS4_METAL_FLASH_ATTN_COMPARE_FILTER"] = args.flash_attn_filter
    if args.verbose:
        env["DS4_METAL_MPP_COMPARE_VERBOSE"] = "1"
    if args.continue_after_breach:
        env["DS4_METAL_MPP_COMPARE_CONTINUE_ON_BREACH"] = "1"
    return env


def ds4_command(args: argparse.Namespace, case_id: str) -> list[str]:
    case = CASE_BY_ID[case_id]
    cmd = [
        str(args.ds4),
        "--metal",
        "-mt",
        "auto",
        "--prompt-file",
        case.prompt_path,
        "-c",
        str(case.ctx),
        "-n",
        str(args.gen_tokens),
        "--system",
        "",
        "--nothink",
        "--temp",
        "0",
    ]
    if args.model:
        cmd[1:1] = ["-m", str(args.model)]
    return cmd


def run_probe(
    cmd: list[str],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    env_prefix = [f"{name}={value}" for name, value in sorted(env_overrides.items())]
    print("+", shell_join(["env", *env_prefix, *cmd]), f">{log_path} 2>&1", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(
            f"probe failed with exit {proc.returncode}: {' '.join(cmd)}\n"
            f"see {log_path}"
        )


def build_run_config(
    args: argparse.Namespace,
    *,
    env_overrides: dict[str, dict[str, str]],
    commands: dict[str, list[str]],
    logs: dict[str, str],
) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "repo_root": str(args.repo_root),
        "ds4": str(args.ds4),
        "model": str(args.model) if args.model else None,
        "out_dir": str(args.out_dir),
        "preset": args.preset,
        "cases": args.case,
        "routes": args.route,
        "q8_filter": args.q8_filter,
        "flash_attn_filter": args.flash_attn_filter,
        "compare_max": args.compare_max,
        "continue_after_breach": args.continue_after_breach,
        "verbose": args.verbose,
        "gen_tokens": args.gen_tokens,
        "max_abs_target": args.max_abs_target,
        "rms_target": args.rms_target,
        "env": env_overrides,
        "commands": commands,
        "logs": logs,
        "dry_run": args.dry_run,
        "allow_stale_binary": args.allow_stale_binary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Candidate presets:\n{preset_help()}",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ds4", type=Path, default=Path("./ds4"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(CANDIDATE_PRESETS),
        help="Use a named default-off candidate environment preset.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set or override an environment variable for the probe.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASE_BY_ID),
        help="Five-fixture case id to probe; repeatable. Defaults to long_memory_archive.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Probe all five drift-gate cases.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help=(
            "DS4_METAL_MPP_COMPARE_ROUTE value, e.g. all, moe_down, moe_gate, "
            "moe_up, attn_out, q8, flash_attn. Repeatable; comma or pipe "
            "separated values are split."
        ),
    )
    parser.add_argument(
        "--q8-filter",
        help="Set DS4_METAL_Q8_COMPARE_FILTER for dense Q8_0 probes with --route q8.",
    )
    parser.add_argument(
        "--flash-attn-filter",
        help="Set DS4_METAL_FLASH_ATTN_COMPARE_FILTER for FlashAttention probes with --route flash_attn.",
    )
    parser.add_argument("--compare-max", type=int, default=200)
    parser.add_argument(
        "--continue-after-breach",
        action="store_true",
        help="Continue local comparisons after a target breach instead of stopping at the first breach.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gen-tokens", type=int, default=1)
    parser.add_argument("--max-abs-target", type=float, default=1.0e-3)
    parser.add_argument("--rms-target", type=float, default=1.0e-4)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-stale-binary",
        action="store_true",
        help="Skip the source-vs-binary freshness check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare_max < 1:
        raise SystemExit("--compare-max must be >= 1")
    if args.gen_tokens < 1:
        raise SystemExit("--gen-tokens must be >= 1")
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    if args.all_cases:
        args.case = [case.case_id for case in CASES]
    elif not args.case:
        args.case = ["long_memory_archive"]
    args.route = normalize_routes(args.route)
    if args.q8_filter and "q8" not in args.route:
        raise SystemExit("--q8-filter requires --route q8")
    if args.flash_attn_filter and "flash_attn" not in args.route:
        raise SystemExit("--flash-attn-filter requires --route flash_attn")

    args.repo_root = args.repo_root.resolve()
    if not args.ds4.is_absolute():
        args.ds4 = args.repo_root / args.ds4
    if not args.dry_run:
        assert_fresh_binary(
            args.ds4,
            repo_root=args.repo_root,
            source_patterns=DS4_FRESHNESS_SOURCES,
            allow_stale=args.allow_stale_binary,
        )
    if args.out_dir is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        preset_label = args.preset or "manual"
        args.out_dir = Path("speed-bench/local-runs") / f"{run_id}-{safe_label(preset_label)}-mpp-compare-probe"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {}
    logs: dict[str, str] = {}
    env_for_config: dict[str, dict[str, str]] = {}
    for route in args.route:
        env_overrides = probe_env(args, route)
        env_for_config[route] = env_overrides
        for case_id in args.case:
            cmd = ds4_command(args, case_id)
            run_key = f"{case_id}:{route}"
            log_path = args.out_dir / f"{case_id}.{safe_label(route)}.log"
            commands[run_key] = cmd
            logs[run_key] = str(log_path)
            run_probe(
                cmd,
                cwd=args.repo_root,
                env_overrides=env_overrides,
                log_path=log_path,
                dry_run=args.dry_run,
            )

    run_config = build_run_config(
        args,
        env_overrides=env_for_config,
        commands=commands,
        logs=logs,
    )
    config_path = args.out_dir / "mpp-compare-run-config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}")

    if args.dry_run:
        print(f"Dry run only; would write {args.out_dir / 'mpp-compare-summary.md'}")
        print(f"Dry run only; would write {args.out_dir / 'mpp-compare-summary.json'}")
        return 0

    summaries = [parse_log(Path(path)) for path in logs.values()]
    summary = merge_summaries(summaries)
    markdown_path = args.out_dir / "mpp-compare-summary.md"
    json_path = args.out_dir / "mpp-compare-summary.json"
    markdown_path.write_text(
        render_markdown(
            summary,
            max_abs_target=args.max_abs_target,
            rms_target=args.rms_target,
            top=args.top,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "run_config": run_config,
                "summary": as_json(
                    summary,
                    max_abs_target=args.max_abs_target,
                    rms_target=args.rms_target,
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
