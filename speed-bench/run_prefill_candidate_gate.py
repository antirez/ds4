#!/usr/bin/env python3
"""Benchmark a prefill candidate and optionally run the quality drift gate.

This is intended for default-off Metal Tensor experiments. It compares:

  standard  -> ./ds4-bench -mt off
  tensor    -> ./ds4-bench -mt auto
  candidate -> ./ds4-bench -mt <candidate-mode> with --set-env overrides

Use --run-drift-gate before promotion. The helper only launches drift gates
after the speed screen passes, and the drift gates reuse the same candidate env
overrides so their "tensor" rows are the candidate route. Candidates that route
nonzero prefill positions also run the chunked frontier drift gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metal_tensor_presets import CANDIDATE_PRESETS, preset_help


@dataclass(frozen=True)
class BenchRun:
    name: str
    label: str
    mode_args: list[str]
    env: dict[str, str]


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
            "Rebuild before running the candidate gate, or pass --allow-stale-binary "
            "only when intentionally summarizing old artifacts."
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


def candidate_env_from_args(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.preset:
        preset = CANDIDATE_PRESETS[args.preset]
        env.update(preset.env)
        if args.candidate_label is None:
            args.candidate_label = preset.label
    if args.candidate_label is None:
        args.candidate_label = "candidate"
    env.update(parse_env_overrides(args.set_env))
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


def evaluate_prefill_speed(
    summary: dict[str, Any],
    *,
    candidate_name: str,
    min_prefill_gain_pct: float,
    min_repeat_prefill_gain_pct: float,
    min_generation_gain_pct: float,
) -> dict[str, Any]:
    gains = summary["gains"][f"{candidate_name}_vs_tensor"]
    rows: list[dict[str, Any]] = []
    for ctx in summary["contexts"]:
        ctx_key = str(ctx)
        gain = gains[ctx_key]
        tensor = summary["runs"]["tensor"]["contexts"][ctx_key]
        candidate = summary["runs"][candidate_name]["contexts"][ctx_key]
        repeat_prefill_gains = [
            ((candidate_prefill / tensor_prefill) - 1.0) * 100.0
            if tensor_prefill
            else 0.0
            for candidate_prefill, tensor_prefill in zip(
                candidate["prefill_tps_values"],
                tensor["prefill_tps_values"],
            )
        ]
        repeat_generation_gains = [
            ((candidate_gen / tensor_gen) - 1.0) * 100.0
            if tensor_gen
            else 0.0
            for candidate_gen, tensor_gen in zip(
                candidate["gen_tps_values"],
                tensor["gen_tps_values"],
            )
        ]
        min_repeat_prefill_gain = min(repeat_prefill_gains) if repeat_prefill_gains else gain["prefill_gain_pct"]
        min_repeat_generation_gain = min(repeat_generation_gains) if repeat_generation_gains else gain["gen_gain_pct"]
        rows.append({
            "ctx": ctx,
            "prefill_gain_pct": gain["prefill_gain_pct"],
            "gen_gain_pct": gain["gen_gain_pct"],
            "repeat_prefill_gain_pct_values": repeat_prefill_gains,
            "repeat_generation_gain_pct_values": repeat_generation_gains,
            "min_repeat_prefill_gain_pct": min_repeat_prefill_gain,
            "min_repeat_generation_gain_pct": min_repeat_generation_gain,
            "prefill_ok": gain["prefill_gain_pct"] >= min_prefill_gain_pct,
            "repeat_prefill_ok": min_repeat_prefill_gain >= min_repeat_prefill_gain_pct,
            "generation_ok": gain["gen_gain_pct"] >= min_generation_gain_pct,
        })
    return {
        "min_prefill_gain_pct_required": min_prefill_gain_pct,
        "min_repeat_prefill_gain_pct_required": min_repeat_prefill_gain_pct,
        "min_generation_gain_pct_required": min_generation_gain_pct,
        "min_prefill_gain_pct": min(row["prefill_gain_pct"] for row in rows),
        "min_repeat_prefill_gain_pct": min(row["min_repeat_prefill_gain_pct"] for row in rows),
        "min_repeat_generation_gain_pct": min(row["min_repeat_generation_gain_pct"] for row in rows),
        "min_generation_gain_pct": min(row["gen_gain_pct"] for row in rows),
        "all_prefill_contexts_ok": all(row["prefill_ok"] for row in rows),
        "all_repeat_prefill_contexts_ok": all(row["repeat_prefill_ok"] for row in rows),
        "all_generation_contexts_ok": all(row["generation_ok"] for row in rows),
        "contexts": rows,
    }


def speed_gate_is_ok(speed_gate: dict[str, Any] | None) -> bool:
    return bool(
        speed_gate and
        speed_gate["all_prefill_contexts_ok"] and
        speed_gate["all_repeat_prefill_contexts_ok"] and
        speed_gate["all_generation_contexts_ok"]
    )


def speed_gate_skip_reason(speed_gate: dict[str, Any] | None) -> str:
    if speed_gate is None:
        return "speed summary missing"
    reasons: list[str] = []
    if not speed_gate["all_prefill_contexts_ok"]:
        reasons.append(
            "candidate prefill is not above Tensor baseline at every measured context "
            f"(min={speed_gate['min_prefill_gain_pct']:.1f}%, "
            f"required={speed_gate['min_prefill_gain_pct_required']:.1f}%)"
        )
    if not speed_gate["all_repeat_prefill_contexts_ok"]:
        reasons.append(
            "candidate prefill is not above the repeat-level Tensor baseline floor "
            f"(min repeat={speed_gate['min_repeat_prefill_gain_pct']:.1f}%, "
            f"required={speed_gate['min_repeat_prefill_gain_pct_required']:.1f}%)"
        )
    if not speed_gate["all_generation_contexts_ok"]:
        reasons.append(
            "candidate generation is below the allowed Tensor-baseline floor "
            f"(min={speed_gate['min_generation_gain_pct']:.1f}%, "
            f"required={speed_gate['min_generation_gain_pct_required']:.1f}%)"
        )
    return "; ".join(reasons) if reasons else "speed screen failed"


def candidate_env_requires_chunked_drift(candidate_env: dict[str, str]) -> bool:
    for value in candidate_env.values():
        for match in re.finditer(r"\bpos\s*[:=]\s*(\d+)", value):
            if int(match.group(1)) != 0:
                return True
    return False


def load_drift_payload(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        with Path(path).open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def tensor_pair_summary_for_gate(
    gate_payload: dict[str, Any],
    *,
    pair_name: str,
    max_tensor_standard_rms: float,
    max_tensor_standard_top20_abs: float,
) -> dict[str, Any]:
    tensor_delta = gate_payload["pairs"][pair_name]["summary"]
    tensor_extrema = gate_payload["pairs"][pair_name].get("extrema", {})
    failures = list(gate_payload.get("gate_failures", []))
    result = {
        "pair": pair_name,
        "ok": len(failures) == 0,
        "failures": failures,
        "max_tensor_standard_rms": max_tensor_standard_rms,
        "max_tensor_standard_top20_abs": max_tensor_standard_top20_abs,
        "tensor_vs_standard_top1_mismatches": tensor_delta["top1_mismatches"],
        "tensor_vs_standard_greedy_mismatches": tensor_delta.get("greedy_mismatches"),
        "tensor_vs_standard_min_top20_overlap": tensor_delta["min_top20_overlap"],
        "tensor_vs_standard_worst_rms": tensor_delta["worst_rms"],
        "tensor_vs_standard_worst_top20_max_abs": tensor_delta["worst_top20_max_abs"],
        "tensor_vs_standard_worst_rms_case": (
            tensor_extrema.get("worst_rms_case") or
            tensor_extrema.get("worst_rms_frontier")
        ),
        "tensor_vs_standard_worst_top20_max_abs_case": (
            tensor_extrema.get("worst_top20_max_abs_case") or
            tensor_extrema.get("worst_top20_max_abs_frontier")
        ),
        "tensor_vs_standard_min_top20_overlap_case": (
            tensor_extrema.get("min_top20_overlap_case") or
            tensor_extrema.get("min_top20_overlap_frontier")
        ),
    }
    rms_failure_present = any("worst_rms exceeds configured envelope" in failure or
                              "worst RMS exceeds configured envelope" in failure
                              for failure in failures)
    top20_failure_present = any("worst_top20_max_abs exceeds configured envelope" in failure or
                                "worst top20 abs exceeds configured envelope" in failure
                                for failure in failures)
    if tensor_delta["worst_rms"] > max_tensor_standard_rms:
        result["ok"] = False
        if not rms_failure_present:
            failures.append(
                f"{pair_name} worst RMS exceeds configured envelope "
                f"({tensor_delta['worst_rms']:.6g} > {max_tensor_standard_rms:.6g})"
            )
    if tensor_delta["worst_top20_max_abs"] > max_tensor_standard_top20_abs:
        result["ok"] = False
        if not top20_failure_present:
            failures.append(
                f"{pair_name} worst top20 abs exceeds configured envelope "
                f"({tensor_delta['worst_top20_max_abs']:.6g} > "
                f"{max_tensor_standard_top20_abs:.6g})"
            )
    result["failures"] = failures
    return result


def evaluate_candidate(
    payload: dict[str, Any],
    *,
    min_prefill_gain_pct: float,
    min_repeat_prefill_gain_pct: float,
    min_generation_gain_pct: float,
    max_tensor_standard_rms: float,
    max_tensor_standard_top20_abs: float,
) -> dict[str, Any]:
    speed = payload.get("speed_summary")
    speed_gate = None
    if speed is not None:
        speed_gate = evaluate_prefill_speed(speed,
                                            candidate_name=payload["candidate_name"],
                                            min_prefill_gain_pct=min_prefill_gain_pct,
                                            min_repeat_prefill_gain_pct=min_repeat_prefill_gain_pct,
                                            min_generation_gain_pct=min_generation_gain_pct)

    drift_path = payload.get("quality_drift_gate_summary")
    drift_payload = load_drift_payload(drift_path)
    drift_gate = {
        "run": drift_payload is not None,
        "ok": False,
        "failures": ["drift gate was not run"] if drift_payload is None else
                    list(drift_payload.get("gate_failures", [])),
    }
    if drift_payload is not None:
        tensor_gate = tensor_pair_summary_for_gate(
            drift_payload,
            pair_name="tensor_vs_standard",
            max_tensor_standard_rms=max_tensor_standard_rms,
            max_tensor_standard_top20_abs=max_tensor_standard_top20_abs,
        )
        drift_gate.update({
            "ok": tensor_gate["ok"],
            "failures": tensor_gate["failures"],
            **{
                key: value
                for key, value in tensor_gate.items()
                if key not in {"ok", "failures"}
            },
        })

    failures: list[str] = []
    if speed_gate is None:
        failures.append("speed summary missing")
    elif not speed_gate["all_prefill_contexts_ok"]:
        failures.append(
            "candidate prefill is not above Tensor baseline at every measured context "
            f"(min={speed_gate['min_prefill_gain_pct']:.1f}%, "
            f"required={speed_gate['min_prefill_gain_pct_required']:.1f}%)"
        )
    if speed_gate is not None and not speed_gate["all_repeat_prefill_contexts_ok"]:
        failures.append(
            "candidate prefill is not above the repeat-level Tensor baseline floor "
            f"(min repeat={speed_gate['min_repeat_prefill_gain_pct']:.1f}%, "
            f"required={speed_gate['min_repeat_prefill_gain_pct_required']:.1f}%)"
        )
    if speed_gate is not None and not speed_gate["all_generation_contexts_ok"]:
        failures.append(
            "candidate generation is below the allowed Tensor-baseline floor "
            f"(min={speed_gate['min_generation_gain_pct']:.1f}%, "
            f"required={speed_gate['min_generation_gain_pct_required']:.1f}%)"
        )
    if not drift_gate["ok"]:
        failures.extend(drift_gate["failures"])

    chunked_required = candidate_env_requires_chunked_drift(payload.get("candidate_env", {}))
    chunked_payload = load_drift_payload(payload.get("chunked_drift_gate_summary"))
    coverage_gate: dict[str, Any] = {
        "required": chunked_required,
        "run": chunked_payload is not None,
        "ok": True,
        "failures": [],
    }
    if chunked_required and chunked_payload is None:
        coverage_gate["ok"] = False
        coverage_gate["failures"].append(
            "candidate uses nonzero pos= route filters; the five-fixture drift "
            "gate does not prove those continuation-prefill chunks, so run the "
            "chunked frontier drift gate before promotion"
        )
    elif chunked_payload is not None:
        coverage_pair = (
            "tensor_vs_default_tensor"
            if "tensor_vs_default_tensor" in chunked_payload.get("pairs", {})
            else "tensor_vs_standard"
        )
        chunked_gate = tensor_pair_summary_for_gate(
            chunked_payload,
            pair_name=coverage_pair,
            max_tensor_standard_rms=max_tensor_standard_rms,
            max_tensor_standard_top20_abs=max_tensor_standard_top20_abs,
        )
        coverage_gate.update({
            "ok": chunked_gate["ok"],
            **{
                key: value
                for key, value in chunked_gate.items()
                if key not in {"ok"}
            },
        })
        coverage_gate["failures"] = [
            f"chunked drift gate: {failure}"
            for failure in chunked_gate["failures"]
        ]
    coverage_failures = coverage_gate["failures"]
    failures.extend(coverage_failures)

    return {
        "promotion_safe": len(failures) == 0,
        "failures": failures,
        "speed_gate": speed_gate,
        "drift_gate": drift_gate,
        "coverage_gate": coverage_gate,
    }


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def shell_join(argv: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def fmt_pct_list(values: list[float]) -> str:
    return ", ".join(fmt_pct(value) for value in values)


def markdown_speed_summary(summary: dict[str, Any], *, candidate_name: str) -> str:
    lines = [
        "## Median Speed",
        "",
        "| Ctx | Standard prefill | Tensor prefill | Candidate prefill | Candidate vs Tensor prefill | Candidate vs Tensor generation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    gains = summary["gains"][f"{candidate_name}_vs_tensor"]
    for ctx in summary["contexts"]:
        ctx_key = str(ctx)
        standard = summary["runs"]["standard"]["contexts"][ctx_key]
        tensor = summary["runs"]["tensor"]["contexts"][ctx_key]
        candidate = summary["runs"][candidate_name]["contexts"][ctx_key]
        gain = gains[ctx_key]
        lines.append(
            "| "
            f"{ctx} | "
            f"{standard['prefill_tps_median']:.2f} | "
            f"{tensor['prefill_tps_median']:.2f} | "
            f"{candidate['prefill_tps_median']:.2f} | "
            f"{fmt_pct(gain['prefill_gain_pct'])} | "
            f"{fmt_pct(gain['gen_gain_pct'])} |"
        )
    return "\n".join(lines)


def markdown_drift_summary(payload: dict[str, Any]) -> str:
    summary_path = payload.get("quality_drift_gate_summary")
    markdown_path = payload.get("quality_drift_gate_markdown")
    if not summary_path:
        skip_reason = payload.get("quality_drift_gate_skipped_reason")
        if skip_reason:
            return "\n".join(
                [
                    "## Drift Gate",
                    "",
                    "Skipped because the speed screen failed.",
                    "",
                    f"Reason: {markdown_escape(skip_reason)}",
                ]
            )
        return "\n".join(
            [
                "## Drift Gate",
                "",
                "Not run. Use `--run-drift-gate` after the speed screen passes before promoting a prefill candidate.",
            ]
        )

    lines = ["## Drift Gate", ""]
    drift_payload: dict[str, Any] | None = None
    try:
        with Path(summary_path).open("r", encoding="utf-8") as fp:
            drift_payload = json.load(fp)
    except FileNotFoundError:
        lines.append(f"Summary JSON not found: `{markdown_escape(summary_path)}`")
    except json.JSONDecodeError as exc:
        lines.append(f"Could not parse `{markdown_escape(summary_path)}`: {exc}")

    if drift_payload is not None:
        failures = drift_payload.get("gate_failures", [])
        lines.append(f"Gate: {'FAIL' if failures else 'OK'}")
        lines.append("")
        if failures:
            lines.append("Failures:")
            lines.append("")
            for failure in failures:
                lines.append(f"- {markdown_escape(failure)}")
            lines.append("")

        lines.extend(
            [
                "| Pair | Top1 mismatches | Greedy mismatches | Min top20 | Worst RMS | Worst top20 abs |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair_name in drift_payload.get(
            "pair_order",
            ("standard_vs_quality", "tensor_vs_quality", "tensor_vs_standard"),
        ):
            pair_payload = drift_payload["pairs"][pair_name]
            pair_summary = pair_payload["summary"]
            lines.append(
                "| "
                f"{markdown_escape(pair_name)} | "
                f"{pair_summary['top1_mismatches']} | "
                f"{pair_summary['greedy_mismatches']} | "
                f"{pair_summary['min_top20_overlap']}/20 | "
                f"{pair_summary['worst_rms']:.6g} | "
                f"{pair_summary['worst_top20_max_abs']:.6g} |"
            )
        target_extrema = drift_payload["pairs"].get("tensor_vs_standard", {}).get("extrema")
        if target_extrema:
            lines.extend(
                [
                    "",
                    "| Tensor-vs-standard target | Fixture | Value |",
                    "| --- | --- | ---: |",
                    "| Worst RMS | "
                    f"{markdown_escape(target_extrema.get('worst_rms_case'))} | "
                    f"{target_extrema['worst_rms']:.6g} |",
                    "| Worst top20 abs | "
                    f"{markdown_escape(target_extrema.get('worst_top20_max_abs_case'))} | "
                    f"{target_extrema['worst_top20_max_abs']:.6g} |",
                    "| Min top20 overlap | "
                    f"{markdown_escape(target_extrema.get('min_top20_overlap_case'))} | "
                    f"{target_extrema['min_top20_overlap']}/20 |",
                ]
            )
    lines.extend(["", "Artifacts:", ""])
    lines.append(f"- JSON: `{markdown_escape(summary_path)}`")
    if markdown_path:
        lines.append(f"- Markdown: `{markdown_escape(markdown_path)}`")
    return "\n".join(lines)


def markdown_chunked_drift_summary(payload: dict[str, Any]) -> str:
    required = candidate_env_requires_chunked_drift(payload.get("candidate_env", {}))
    summary_path = payload.get("chunked_drift_gate_summary")
    markdown_path = payload.get("chunked_drift_gate_markdown")
    skip_reason = payload.get("chunked_drift_gate_skipped_reason")
    if not required and not summary_path and not skip_reason:
        return ""

    if not summary_path:
        lines = ["## Chunked Drift Gate", ""]
        if skip_reason:
            lines.extend([
                "Skipped because the speed screen failed.",
                "",
                f"Reason: {markdown_escape(skip_reason)}",
            ])
        elif required:
            lines.append(
                "Not run. This candidate uses nonzero `pos=` filters, so run "
                "`--run-drift-gate` to capture resumed-prefill frontier drift before promotion."
            )
        else:
            lines.append("Not run.")
        return "\n".join(lines)

    lines = ["## Chunked Drift Gate", ""]
    drift_payload: dict[str, Any] | None = None
    try:
        with Path(summary_path).open("r", encoding="utf-8") as fp:
            drift_payload = json.load(fp)
    except FileNotFoundError:
        lines.append(f"Summary JSON not found: `{markdown_escape(summary_path)}`")
    except json.JSONDecodeError as exc:
        lines.append(f"Could not parse `{markdown_escape(summary_path)}`: {exc}")

    if drift_payload is not None:
        failures = drift_payload.get("gate_failures", [])
        lines.append(f"Gate: {'FAIL' if failures else 'OK'}")
        lines.append("")
        if failures:
            lines.append("Failures:")
            lines.append("")
            for failure in failures:
                lines.append(f"- {markdown_escape(failure)}")
            lines.append("")
        lines.extend(
            [
                "| Pair | Top1 mismatches | Min top20 | Worst RMS | Worst RMS frontier | Worst top20 abs | Worst top20 abs frontier |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair_name in drift_payload.get(
            "pair_order",
            ("standard_vs_quality", "tensor_vs_quality", "tensor_vs_standard"),
        ):
            pair_payload = drift_payload["pairs"][pair_name]
            pair_summary = pair_payload["summary"]
            pair_extrema = pair_payload.get("extrema", {})
            lines.append(
                "| "
                f"{markdown_escape(pair_name)} | "
                f"{pair_summary['top1_mismatches']} | "
                f"{pair_summary['min_top20_overlap']}/20 | "
                f"{pair_summary['worst_rms']:.6g} | "
                f"{markdown_escape(pair_extrema.get('worst_rms_frontier', 'n/a'))} | "
                f"{pair_summary['worst_top20_max_abs']:.6g} | "
                f"{markdown_escape(pair_extrema.get('worst_top20_max_abs_frontier', 'n/a'))} |"
            )
    lines.extend(["", "Artifacts:", ""])
    lines.append(f"- JSON: `{markdown_escape(summary_path)}`")
    if markdown_path:
        lines.append(f"- Markdown: `{markdown_escape(markdown_path)}`")
    return "\n".join(lines)


def markdown_promotion_summary(payload: dict[str, Any]) -> str:
    decision = payload.get("promotion_decision")
    if not decision:
        return "\n".join(["## Promotion Decision", "", "Not evaluated."])

    lines = [
        "## Promotion Decision",
        "",
        f"Promotion-safe: {'yes' if decision['promotion_safe'] else 'no'}",
        "",
    ]
    if decision["failures"]:
        lines.append("Reasons:")
        lines.append("")
        for failure in decision["failures"]:
            lines.append(f"- {markdown_escape(failure)}")
        lines.append("")

    speed_gate = decision.get("speed_gate")
    if speed_gate:
        lines.extend(
            [
                "| Speed gate | Value |",
                "| --- | ---: |",
                f"| Required min prefill gain | {fmt_pct(speed_gate['min_prefill_gain_pct_required'])} |",
                f"| Required min repeat prefill gain | {fmt_pct(speed_gate['min_repeat_prefill_gain_pct_required'])} |",
                f"| Required min generation gain | {fmt_pct(speed_gate['min_generation_gain_pct_required'])} |",
                f"| Observed min prefill gain | {fmt_pct(speed_gate['min_prefill_gain_pct'])} |",
                f"| Observed min repeat prefill gain | {fmt_pct(speed_gate['min_repeat_prefill_gain_pct'])} |",
                f"| Observed min generation gain | {fmt_pct(speed_gate['min_generation_gain_pct'])} |",
                f"| Observed min repeat generation gain | {fmt_pct(speed_gate['min_repeat_generation_gain_pct'])} |",
                f"| All prefill contexts pass | {'yes' if speed_gate['all_prefill_contexts_ok'] else 'no'} |",
                f"| All repeat prefill contexts pass | {'yes' if speed_gate['all_repeat_prefill_contexts_ok'] else 'no'} |",
                f"| All generation contexts pass | {'yes' if speed_gate['all_generation_contexts_ok'] else 'no'} |",
                "",
            ]
        )
        lines.extend(
            [
                "| Ctx | Median prefill | Repeat prefill | Median generation | Repeat generation |",
                "| ---: | ---: | --- | ---: | --- |",
            ]
        )
        for row in speed_gate["contexts"]:
            lines.append(
                "| "
                f"{row['ctx']} | "
                f"{fmt_pct(row['prefill_gain_pct'])} | "
                f"{markdown_escape(fmt_pct_list(row['repeat_prefill_gain_pct_values']))} | "
                f"{fmt_pct(row['gen_gain_pct'])} | "
                f"{markdown_escape(fmt_pct_list(row['repeat_generation_gain_pct_values']))} |"
            )
        lines.append("")

    drift_gate = decision.get("drift_gate")
    if drift_gate:
        lines.extend(
            [
                "| Drift gate | Value |",
                "| --- | ---: |",
                f"| Run | {'yes' if drift_gate['run'] else 'no'} |",
                f"| OK | {'yes' if drift_gate['ok'] else 'no'} |",
            ]
        )
        if drift_gate.get("run"):
            lines.extend(
                [
                    f"| Max Tensor-vs-standard RMS | {drift_gate['max_tensor_standard_rms']:.6g} |",
                    f"| Max Tensor-vs-standard top20 abs | {drift_gate['max_tensor_standard_top20_abs']:.6g} |",
                    f"| Tensor-vs-standard top1 mismatches | {drift_gate['tensor_vs_standard_top1_mismatches']} |",
                    f"| Tensor-vs-standard greedy mismatches | {drift_gate['tensor_vs_standard_greedy_mismatches']} |",
                    f"| Tensor-vs-standard min top20 | {drift_gate['tensor_vs_standard_min_top20_overlap']}/20 |",
                    f"| Tensor-vs-standard worst RMS | {drift_gate['tensor_vs_standard_worst_rms']:.6g} |",
                    f"| Tensor-vs-standard worst RMS case | {markdown_escape(drift_gate.get('tensor_vs_standard_worst_rms_case') or 'n/a')} |",
                    f"| Tensor-vs-standard worst top20 abs | {drift_gate['tensor_vs_standard_worst_top20_max_abs']:.6g} |",
                    f"| Tensor-vs-standard worst top20 abs case | {markdown_escape(drift_gate.get('tensor_vs_standard_worst_top20_max_abs_case') or 'n/a')} |",
                ]
            )
        lines.append("")
    coverage_gate = decision.get("coverage_gate")
    if coverage_gate:
        lines.extend(
            [
                "",
                "| Coverage gate | Value |",
                "| --- | ---: |",
                f"| Requires chunked drift coverage | {'yes' if coverage_gate.get('required') else 'no'} |",
                f"| Chunked drift run | {'yes' if coverage_gate.get('run') else 'no'} |",
                f"| OK | {'yes' if coverage_gate['ok'] else 'no'} |",
            ]
        )
        if coverage_gate.get("run") and "tensor_vs_standard_worst_rms" in coverage_gate:
            lines.extend(
                [
                    f"| Coverage pair | {markdown_escape(coverage_gate.get('pair') or 'n/a')} |",
                    f"| Max coverage RMS | {coverage_gate['max_tensor_standard_rms']:.6g} |",
                    f"| Max coverage top20 abs | {coverage_gate['max_tensor_standard_top20_abs']:.6g} |",
                    f"| Coverage top1 mismatches | {coverage_gate['tensor_vs_standard_top1_mismatches']} |",
                    f"| Coverage min top20 | {coverage_gate['tensor_vs_standard_min_top20_overlap']}/20 |",
                    f"| Coverage worst RMS | {coverage_gate['tensor_vs_standard_worst_rms']:.6g} |",
                    f"| Coverage worst RMS frontier | {markdown_escape(coverage_gate.get('tensor_vs_standard_worst_rms_case') or 'n/a')} |",
                    f"| Coverage worst top20 abs | {coverage_gate['tensor_vs_standard_worst_top20_max_abs']:.6g} |",
                    f"| Coverage worst top20 abs frontier | {markdown_escape(coverage_gate.get('tensor_vs_standard_worst_top20_max_abs_case') or 'n/a')} |",
                ]
            )
    return "\n".join(lines)


def markdown_run_config(payload: dict[str, Any]) -> str:
    config = payload.get("run_config")
    if not config:
        return ""
    lines = [
        "## Run Config",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key in (
        "repo_root",
        "ds4_bench",
        "ds4",
        "model",
        "prompt_file",
        "out_dir",
        "ctx_start",
        "ctx_max",
        "step_mul",
        "gen_tokens",
        "repeat",
        "candidate_preset",
        "candidate_mode",
        "reuse",
        "run_drift_gate",
        "min_prefill_gain_pct",
        "min_repeat_prefill_gain_pct",
        "min_generation_gain_pct",
        "max_tensor_standard_rms",
        "max_tensor_standard_top20_abs",
    ):
        if key in config:
            lines.append(f"| `{markdown_escape(key)}` | `{markdown_escape(config[key])}` |")
    if config.get("argv"):
        lines.extend(
            [
                "",
                "Replay command:",
                "",
                "```sh",
                shell_join(["python3", *config["argv"]]),
                "```",
            ]
        )
    return "\n".join(lines)


def write_candidate_markdown_summary(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prefill Candidate Gate",
        "",
        f"Candidate: `{markdown_escape(payload['candidate_label'])}`",
        f"Mode: `-mt {markdown_escape(payload['candidate_mode'])}`",
        "",
    ]
    if payload.get("candidate_preset"):
        lines.append(f"Preset: `{markdown_escape(payload['candidate_preset'])}`")
        lines.append("")
    candidate_env = payload["candidate_env"]
    if candidate_env:
        lines.append("Environment overrides:")
        lines.append("")
        for name, value in sorted(candidate_env.items()):
            lines.append(f"- `{markdown_escape(name)}={markdown_escape(value)}`")
    else:
        lines.append("Environment overrides: none")
    lines.append("")
    run_config = markdown_run_config(payload)
    if run_config:
        lines.append(run_config)
        lines.append("")
    lines.append(markdown_promotion_summary(payload))
    lines.append("")

    if "speed_summary" in payload:
        lines.append(markdown_speed_summary(payload["speed_summary"],
                                            candidate_name=payload["candidate_name"]))
    else:
        lines.append("## Median Speed")
        lines.append("")
        lines.append("Not available in dry-run mode.")
    lines.append("")
    lines.append(markdown_drift_summary(payload))
    chunked_drift_summary = markdown_chunked_drift_summary(payload)
    if chunked_drift_summary:
        lines.append("")
        lines.append(chunked_drift_summary)
    lines.append("")
    lines.append("## CSV Inputs")
    lines.append("")
    for name, paths in payload["csv_paths"].items():
        for csv_path in paths:
            lines.append(f"- `{markdown_escape(name)}`: `{markdown_escape(csv_path)}`")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "repo_root": str(args.repo_root),
        "ds4_bench": str(args.ds4_bench),
        "ds4": str(args.ds4),
        "python": str(args.python),
        "model": str(args.model) if args.model else None,
        "prompt_file": str(args.prompt_file),
        "out_dir": str(args.out_dir),
        "candidate_preset": args.preset,
        "candidate_label": args.candidate_label,
        "candidate_mode": args.candidate_mode,
        "ctx_start": args.ctx_start,
        "ctx_max": args.ctx_max,
        "step_mul": args.step_mul,
        "gen_tokens": args.gen_tokens,
        "repeat": args.repeat,
        "min_prefill_gain_pct": args.min_prefill_gain_pct,
        "min_repeat_prefill_gain_pct": args.min_repeat_prefill_gain_pct,
        "min_generation_gain_pct": args.min_generation_gain_pct,
        "max_tensor_standard_rms": args.max_tensor_standard_rms,
        "max_tensor_standard_top20_abs": args.max_tensor_standard_top20_abs,
        "run_drift_gate": args.run_drift_gate,
        "fail_on_quality_greedy": args.fail_on_quality_greedy,
        "allow_stale_binary": args.allow_stale_binary,
        "reuse": args.reuse,
        "no_fail": args.no_fail,
        "dry_run": args.dry_run,
    }


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
            if args.reuse and csv_path.exists():
                print(f"reuse {csv_path}", flush=True)
            else:
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
        if args.reuse and chart_path.exists():
            print(f"reuse {chart_path}", flush=True)
        else:
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
    cmd.append("--no-fail")
    if args.reuse:
        cmd.append("--reuse")
    if args.allow_stale_binary:
        cmd.append("--allow-stale-binary")
    cmd += ["--max-tensor-standard-rms", str(args.max_tensor_standard_rms)]
    cmd += ["--max-tensor-standard-top20-abs", str(args.max_tensor_standard_top20_abs)]
    for name, value in sorted(candidate_env.items()):
        cmd += ["--set-env", f"{name}={value}"]
    run_command(cmd, cwd=args.repo_root, env_overrides={}, dry_run=args.dry_run)
    return gate_dir


def run_chunked_drift_gate(args: argparse.Namespace, candidate_env: dict[str, str]) -> Path:
    gate_dir = args.out_dir / "chunked-drift-gate"
    cmd = [
        str(args.python),
        "speed-bench/run_chunked_prefill_drift_gate.py",
        "--repo-root",
        str(args.repo_root),
        "--ds4-bench",
        str(args.ds4_bench),
        "--prompt-file",
        str(args.prompt_file),
        "--out-dir",
        str(gate_dir),
        "--ctx-start",
        str(args.ctx_start),
        "--ctx-max",
        str(args.ctx_max),
        "--step-mul",
        str(args.step_mul),
        "--gen-tokens",
        "1",
        "--max-tensor-default-rms",
        str(args.max_tensor_standard_rms),
        "--max-tensor-default-top20-abs",
        str(args.max_tensor_standard_top20_abs),
        "--no-fail",
    ]
    if args.model:
        cmd += ["--model", str(args.model)]
    if args.reuse:
        cmd.append("--reuse")
    for name, value in sorted(candidate_env.items()):
        cmd += ["--set-env", f"{name}={value}"]
    run_command(cmd, cwd=args.repo_root, env_overrides={}, dry_run=args.dry_run)
    return gate_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Candidate presets:\n{preset_help()}",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ds4-bench", type=Path, default=Path("./ds4-bench"))
    parser.add_argument("--ds4", type=Path, default=Path("./ds4"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt-file", type=Path, default=Path("speed-bench/promessi_sposi.txt"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(CANDIDATE_PRESETS),
        help="Use a named default-off candidate environment preset.",
    )
    parser.add_argument("--candidate-label")
    parser.add_argument("--candidate-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--ctx-start", type=int, default=512)
    parser.add_argument("--ctx-max", type=int, default=8192)
    parser.add_argument("--step-mul", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--min-prefill-gain-pct",
        type=float,
        default=0.0,
        help="Minimum candidate-vs-Tensor prefill gain required at every measured context for promotion.",
    )
    parser.add_argument(
        "--min-repeat-prefill-gain-pct",
        type=float,
        default=0.0,
        help="Minimum candidate-vs-Tensor prefill gain required for every repeat/context pair.",
    )
    parser.add_argument(
        "--min-generation-gain-pct",
        type=float,
        default=-5.0,
        help="Minimum candidate-vs-Tensor generation gain allowed at every measured context for promotion.",
    )
    parser.add_argument(
        "--max-tensor-standard-rms",
        type=float,
        default=0.30,
        help="Maximum Tensor-vs-standard worst RMS allowed for production promotion.",
    )
    parser.add_argument(
        "--max-tensor-standard-top20-abs",
        type=float,
        default=0.60,
        help="Maximum Tensor-vs-standard worst top-20 absolute drift allowed for production promotion.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set an environment variable only for the candidate bench and drift gate.",
    )
    parser.add_argument("--run-drift-gate", action="store_true")
    parser.add_argument("--fail-on-quality-greedy", action="store_true")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing benchmark CSVs/charts and drift-gate dumps in --out-dir when present.",
    )
    parser.add_argument(
        "--allow-stale-binary",
        action="store_true",
        help="Skip source-vs-binary freshness checks.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after writing the promotion decision.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    candidate_env = candidate_env_from_args(args)
    if args.out_dir is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        args.out_dir = Path("speed-bench/local-runs") / f"{run_id}-{safe_label(args.candidate_label)}"
    args.repo_root = args.repo_root.resolve()
    if not args.ds4_bench.is_absolute():
        args.ds4_bench = args.repo_root / args.ds4_bench
    if not args.ds4.is_absolute():
        args.ds4 = args.repo_root / args.ds4
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        assert_fresh_binary(
            args.ds4_bench,
            repo_root=args.repo_root,
            source_patterns=DS4_BENCH_FRESHNESS_SOURCES,
            allow_stale=args.allow_stale_binary,
        )

    candidate_name = safe_label(args.candidate_label)
    if candidate_name in {"standard", "tensor"}:
        raise SystemExit("--candidate-label must not resolve to 'standard' or 'tensor'")
    csv_paths = run_benchmarks(args, candidate_env)

    payload: dict[str, Any] = {
        "candidate_label": args.candidate_label,
        "candidate_name": candidate_name,
        "candidate_preset": args.preset,
        "candidate_mode": args.candidate_mode,
        "candidate_env": candidate_env,
        "run_config": build_run_config(args),
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
        payload["speed_screen"] = evaluate_prefill_speed(
            speed_summary,
            candidate_name=candidate_name,
            min_prefill_gain_pct=args.min_prefill_gain_pct,
            min_repeat_prefill_gain_pct=args.min_repeat_prefill_gain_pct,
            min_generation_gain_pct=args.min_generation_gain_pct,
        )

    if args.run_drift_gate:
        speed_screen = payload.get("speed_screen")
        if args.dry_run or speed_gate_is_ok(speed_screen):
            gate_dir = run_drift_gate(args, candidate_env)
            payload["quality_drift_gate_summary"] = str(gate_dir / "summary.json")
            payload["quality_drift_gate_markdown"] = str(gate_dir / "summary.md")
            if candidate_env_requires_chunked_drift(candidate_env):
                chunked_gate_dir = run_chunked_drift_gate(args, candidate_env)
                payload["chunked_drift_gate_summary"] = str(chunked_gate_dir / "summary.json")
                payload["chunked_drift_gate_markdown"] = str(chunked_gate_dir / "summary.md")
        else:
            skip_reason = speed_gate_skip_reason(speed_screen)
            payload["quality_drift_gate_skipped_reason"] = skip_reason
            if candidate_env_requires_chunked_drift(candidate_env):
                payload["chunked_drift_gate_skipped_reason"] = skip_reason
            print(f"\nSkipping drift gate because the speed screen failed: {skip_reason}")
    elif args.reuse:
        gate_dir = args.out_dir / "quality-drift-gate"
        if (gate_dir / "summary.json").exists():
            payload["quality_drift_gate_summary"] = str(gate_dir / "summary.json")
        if (gate_dir / "summary.md").exists():
            payload["quality_drift_gate_markdown"] = str(gate_dir / "summary.md")
        chunked_gate_dir = args.out_dir / "chunked-drift-gate"
        if (chunked_gate_dir / "summary.json").exists():
            payload["chunked_drift_gate_summary"] = str(chunked_gate_dir / "summary.json")
        if (chunked_gate_dir / "summary.md").exists():
            payload["chunked_drift_gate_markdown"] = str(chunked_gate_dir / "summary.md")

    if not args.dry_run:
        payload["promotion_decision"] = evaluate_candidate(
            payload,
            min_prefill_gain_pct=args.min_prefill_gain_pct,
            min_repeat_prefill_gain_pct=args.min_repeat_prefill_gain_pct,
            min_generation_gain_pct=args.min_generation_gain_pct,
            max_tensor_standard_rms=args.max_tensor_standard_rms,
            max_tensor_standard_top20_abs=args.max_tensor_standard_top20_abs,
        )

    summary_path = args.out_dir / "prefill-candidate-summary.json"
    markdown_path = args.out_dir / "prefill-candidate-summary.md"
    if not args.dry_run:
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
            fp.write("\n")
        write_candidate_markdown_summary(payload, markdown_path)
        print(f"\nWrote {summary_path}")
        print(f"Wrote {markdown_path}")
    else:
        print(f"\nDry run only; would write {summary_path}")
        print(f"Dry run only; would write {markdown_path}")
    if (not args.dry_run and
            args.run_drift_gate and
            not args.no_fail and
            not payload["promotion_decision"]["promotion_safe"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
