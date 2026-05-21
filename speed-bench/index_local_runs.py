#!/usr/bin/env python3
"""Index saved speed-bench/local-runs artifacts.

This scans ignored local run artifacts and builds a compact Markdown/JSON
evidence index across candidate gates, drift gates, comparator probes, and stage
profiles. It never runs the model; it only reads existing JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_label(path: Path, root: Path) -> str:
    parent = path.parent
    if parent.name in {"quality-drift-gate", "chunked-drift-gate"} and parent.parent != root:
        return f"{parent.parent.name}/{parent.name}"
    return parent.name


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def bool_label(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def coverage_label(item: dict[str, Any]) -> str:
    if not item.get("coverage_required") and not item.get("coverage_run"):
        return "n/a"
    return bool_label(item.get("coverage_ok"))


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def env_label(env: dict[str, str] | None, max_items: int = 3) -> str:
    if not env:
        return "none"
    items = [f"{name}={value}" for name, value in sorted(env.items())]
    if len(items) > max_items:
        items = items[:max_items] + [f"...(+{len(env) - max_items})"]
    return ", ".join(items)


def candidate_speed_from_gains(data: dict[str, Any]) -> tuple[float | None, float | None]:
    speed = data.get("speed_summary") or {}
    name = data.get("candidate_name")
    gains = speed.get("gains") or {}
    pair = gains.get(f"{name}_vs_tensor") if name else None
    if not isinstance(pair, dict) or not pair:
        return None, None
    prefill = [
        row.get("prefill_gain_pct")
        for row in pair.values()
        if isinstance(row, dict) and row.get("prefill_gain_pct") is not None
    ]
    gen = [
        row.get("gen_gain_pct")
        for row in pair.values()
        if isinstance(row, dict) and row.get("gen_gain_pct") is not None
    ]
    return (min(prefill) if prefill else None, min(gen) if gen else None)


def read_bench_csv(path: Path) -> dict[int, dict[str, float]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                return None
            required = {"ctx_tokens", "prefill_tps", "gen_tps"}
            if not required.issubset(reader.fieldnames):
                return None
            rows: dict[int, dict[str, float]] = {}
            for row in reader:
                ctx = int(row["ctx_tokens"])
                rows[ctx] = {
                    "prefill_tps": float(row["prefill_tps"]),
                    "gen_tps": float(row["gen_tps"]),
                }
            return rows or None
    except (OSError, ValueError):
        return None


def gain_pct(other: float | None, base: float | None) -> float | None:
    if other is None or base is None or base == 0.0:
        return None
    return ((other / base) - 1.0) * 100.0


def min_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def max_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def prefixed_files(run_dir: Path, suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(run_dir.glob(f"*{suffix}")):
        name = path.name
        if name.endswith(suffix):
            files[name[:-len(suffix)]] = path
    return files


def collect_candidate(path: Path, root: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not isinstance(data, dict) or "candidate_label" not in data:
        return None
    decision = data.get("promotion_decision") or {}
    speed_gate = decision.get("speed_gate") or {}
    drift_gate = decision.get("drift_gate") or {}
    coverage_gate = decision.get("coverage_gate") or {}
    min_prefill = speed_gate.get("min_prefill_gain_pct")
    min_gen = speed_gate.get("min_generation_gain_pct")
    if min_prefill is None or min_gen is None:
        fallback_prefill, fallback_gen = candidate_speed_from_gains(data)
        min_prefill = fallback_prefill if min_prefill is None else min_prefill
        min_gen = fallback_gen if min_gen is None else min_gen
    return {
        "path": rel(path, root),
        "run": run_label(path, root),
        "candidate": data.get("candidate_label"),
        "preset": data.get("candidate_preset"),
        "env": data.get("candidate_env") or {},
        "promotion_safe": decision.get("promotion_safe"),
        "min_prefill_gain_pct": min_prefill,
        "min_generation_gain_pct": min_gen,
        "min_repeat_prefill_gain_pct": speed_gate.get("min_repeat_prefill_gain_pct"),
        "drift_run": drift_gate.get("run"),
        "drift_ok": drift_gate.get("ok"),
        "coverage_required": coverage_gate.get("required"),
        "coverage_run": coverage_gate.get("run"),
        "coverage_ok": coverage_gate.get("ok"),
        "coverage_pair": coverage_gate.get("pair"),
        "coverage_tensor_standard_worst_rms": coverage_gate.get("tensor_vs_standard_worst_rms"),
        "coverage_tensor_standard_worst_rms_case": coverage_gate.get("tensor_vs_standard_worst_rms_case"),
        "coverage_tensor_standard_worst_top20_abs": coverage_gate.get("tensor_vs_standard_worst_top20_max_abs"),
        "coverage_tensor_standard_worst_top20_abs_case": coverage_gate.get("tensor_vs_standard_worst_top20_max_abs_case"),
        "tensor_standard_worst_rms": drift_gate.get("tensor_vs_standard_worst_rms"),
        "tensor_standard_worst_rms_case": drift_gate.get("tensor_vs_standard_worst_rms_case"),
        "tensor_standard_worst_top20_abs": drift_gate.get("tensor_vs_standard_worst_top20_max_abs"),
        "tensor_standard_worst_top20_abs_case": drift_gate.get("tensor_vs_standard_worst_top20_abs_case"),
        "failures": decision.get("failures") or [],
    }


def collect_drift(path: Path, root: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not isinstance(data, dict) or "pairs" not in data or "modes" not in data:
        return None
    pairs = data.get("pairs") or {}
    tensor_standard = pairs.get("tensor_vs_standard", {})
    ts_summary = tensor_standard.get("summary", {})
    ts_extrema = tensor_standard.get("extrema", {})
    is_chunked = isinstance(data.get("frontiers"), list)
    return {
        "path": rel(path, root),
        "run": run_label(path, root),
        "kind": "chunked" if is_chunked else "five-fixture",
        "env": data.get("env") or data.get("candidate_env") or {},
        "preset": (data.get("run_config") or {}).get("candidate_preset"),
        "gate_ok": not bool(data.get("gate_failures")),
        "failures": data.get("gate_failures") or [],
        "tensor_standard_top1": ts_summary.get("top1_mismatches"),
        "tensor_standard_greedy": ts_summary.get("greedy_mismatches"),
        "tensor_standard_min_top20": ts_summary.get("min_top20_overlap"),
        "tensor_standard_worst_rms": ts_summary.get("worst_rms"),
        "tensor_standard_worst_rms_case": (
            ts_extrema.get("worst_rms_case") or ts_extrema.get("worst_rms_frontier")
        ),
        "tensor_standard_worst_top20_abs": ts_summary.get("worst_top20_max_abs"),
        "tensor_standard_worst_top20_abs_case": (
            ts_extrema.get("worst_top20_max_abs_case") or
            ts_extrema.get("worst_top20_max_abs_frontier")
        ),
    }


def unwrap_compare_summary(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    if isinstance(summary, dict) and "count" in summary:
        return summary
    return data


def collect_compare(path: Path, root: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    summary = unwrap_compare_summary(data)
    if "top_max_abs" not in summary:
        return None
    top_max = (summary.get("top_max_abs") or [{}])[0] if summary.get("top_max_abs") else {}
    top_rms = (summary.get("top_rms") or [{}])[0] if summary.get("top_rms") else {}
    return {
        "path": rel(path, root),
        "run": run_label(path, root),
        "count": summary.get("count"),
        "routes": summary.get("route_counts") or {},
        "threshold_breaches": len(summary.get("threshold_breaches") or []),
        "explicit_breaches": len(summary.get("breaches") or []),
        "worst_max_abs": top_max.get("max_abs"),
        "worst_max_abs_route": top_max.get("route"),
        "worst_max_abs_module": top_max.get("module"),
        "worst_rms": top_rms.get("rms"),
        "worst_rms_route": top_rms.get("route"),
        "worst_rms_module": top_rms.get("module"),
    }


def collect_stage(path: Path, root: Path) -> dict[str, Any] | None:
    data = load_json(path)
    summaries = data if isinstance(data, list) else [data]
    if not summaries or not isinstance(summaries[0], dict) or "stages" not in summaries[0]:
        return None
    first = summaries[0]
    stages = first.get("stages") or {}
    q8_shapes = first.get("q8_shapes") or {}
    flash_shapes = first.get("flash_shapes") or {}
    top_stage_name, top_stage = max(
        stages.items(),
        key=lambda item: item[1].get("total_ms", 0.0),
        default=("n/a", {}),
    )
    top_q8_name, top_q8 = max(
        q8_shapes.items(),
        key=lambda item: item[1].get("total_ms", 0.0),
        default=("n/a", {}),
    )
    top_flash_name, top_flash = max(
        flash_shapes.items(),
        key=lambda item: item[1].get("total_ms", 0.0),
        default=("n/a", {}),
    )
    throughput = first.get("throughput") or []
    last_throughput = throughput[-1] if throughput else {}
    return {
        "path": rel(path, root),
        "run": run_label(path, root),
        "events": first.get("events"),
        "prefill_tps": last_throughput.get("prefill_tps"),
        "generation_tps": last_throughput.get("generation_tps"),
        "top_stage": top_stage_name,
        "top_stage_ms": top_stage.get("total_ms"),
        "top_q8_shape": top_q8_name,
        "top_q8_ms": top_q8.get("total_ms"),
        "top_flash_shape": top_flash_name,
        "top_flash_ms": top_flash.get("total_ms"),
    }


def collect_metal_tensor_bench(run_dir: Path, root: Path) -> list[dict[str, Any]]:
    standards = prefixed_files(run_dir, "_ds4_bench_standard_metal.csv")
    qualities = prefixed_files(run_dir, "_ds4_bench_quality.csv")
    tensors = prefixed_files(run_dir, "_ds4_bench_tensor_metal.csv")
    prefixes = sorted(set(standards) & set(qualities) & set(tensors))
    if not prefixes:
        return []

    items: list[dict[str, Any]] = []
    for prefix in prefixes:
        standard_csv = standards[prefix]
        quality_csv = qualities[prefix]
        tensor_csv = tensors[prefix]
        standard = read_bench_csv(standard_csv)
        quality = read_bench_csv(quality_csv)
        tensor = read_bench_csv(tensor_csv)
        if not standard or not quality or not tensor:
            continue

        contexts = sorted(set(standard) & set(quality) & set(tensor))
        if not contexts:
            continue

        tensor_vs_standard_prefill = [
            gain_pct(tensor[ctx]["prefill_tps"], standard[ctx]["prefill_tps"])
            for ctx in contexts
        ]
        tensor_vs_standard_gen = [
            gain_pct(tensor[ctx]["gen_tps"], standard[ctx]["gen_tps"])
            for ctx in contexts
        ]
        quality_vs_standard_prefill = [
            gain_pct(quality[ctx]["prefill_tps"], standard[ctx]["prefill_tps"])
            for ctx in contexts
        ]
        chart_path = run_dir / f"{prefix}_ds4_bench_standard_quality_tensor.png"
        run_name = run_dir.name if len(prefixes) == 1 else f"{run_dir.name}/{prefix}"
        items.append({
            "path": rel(run_dir, root),
            "run": run_name,
            "prefix": prefix,
            "chart": rel(chart_path, root) if chart_path.exists() else None,
            "standard_csv": rel(standard_csv, root),
            "quality_csv": rel(quality_csv, root),
            "tensor_csv": rel(tensor_csv, root),
            "contexts": contexts,
            "min_tensor_prefill_vs_standard_pct": min_present(tensor_vs_standard_prefill),
            "max_tensor_prefill_vs_standard_pct": max_present(tensor_vs_standard_prefill),
            "min_tensor_gen_vs_standard_pct": min_present(tensor_vs_standard_gen),
            "max_tensor_gen_vs_standard_pct": max_present(tensor_vs_standard_gen),
            "min_quality_prefill_vs_standard_pct": min_present(quality_vs_standard_prefill),
            "max_quality_prefill_vs_standard_pct": max_present(quality_vs_standard_prefill),
        })
    return items


def collect(root: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    drifts: list[dict[str, Any]] = []
    compares: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    metal_benches: list[dict[str, Any]] = []
    if root.exists():
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            metal_benches.extend(collect_metal_tensor_bench(run_dir, root))
    for path in sorted(root.rglob("*.json")):
        name = path.name
        if name == "prefill-candidate-summary.json":
            item = collect_candidate(path, root)
            if item:
                candidates.append(item)
        elif name == "summary.json" and path.parent.name == "quality-drift-gate":
            item = collect_drift(path, root)
            if item:
                drifts.append(item)
        elif name == "summary.json":
            item = collect_drift(path, root)
            if item:
                drifts.append(item)
        elif name == "mpp-compare-summary.json":
            item = collect_compare(path, root)
            if item:
                compares.append(item)
        elif name == "stage-profile-summary.json":
            item = collect_stage(path, root)
            if item:
                stages.append(item)
    return {
        "candidates": candidates,
        "drift_gates": drifts,
        "mpp_compares": compares,
        "stage_profiles": stages,
        "metal_tensor_benches": metal_benches,
    }


def top_items(items: list[dict[str, Any]], key: str, top: int, reverse: bool = True) -> list[dict[str, Any]]:
    sortable = [item for item in items if item.get(key) is not None]
    return sorted(sortable, key=lambda item: item[key], reverse=reverse)[:top]


def render_markdown(index: dict[str, list[dict[str, Any]]], top: int) -> str:
    lines: list[str] = [
        "# DS4 Local Run Index",
        "",
        "| Artifact type | Count |",
        "| --- | ---: |",
        f"| Prefill candidates | {len(index['candidates'])} |",
        f"| Metal Tensor bench charts | {len(index['metal_tensor_benches'])} |",
        f"| Drift gates | {len(index['drift_gates'])} |",
        f"| Comparator summaries | {len(index['mpp_compares'])} |",
        f"| Stage profiles | {len(index['stage_profiles'])} |",
        "",
    ]

    if index["candidates"]:
        lines.extend(
            [
                "## Prefill Candidates By Speed",
                "",
                "| Run | Candidate | Promotion-safe | 5-fixture OK | Coverage OK | Coverage pair | Min prefill vs Tensor | Min repeat prefill | Min gen vs Tensor | 5-fixture RMS | 5-fixture top20 | Coverage RMS | Coverage top20 |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in top_items(index["candidates"], "min_prefill_gain_pct", top):
            lines.append(
                "| "
                f"`{markdown_escape(item['run'])}` | "
                f"`{markdown_escape(item['candidate'])}` | "
                f"{bool_label(item.get('promotion_safe'))} | "
                f"{bool_label(item.get('drift_ok'))} | "
                f"{coverage_label(item)} | "
                f"`{markdown_escape(item.get('coverage_pair') or 'n/a')}` | "
                f"{fmt_pct(item.get('min_prefill_gain_pct'))} | "
                f"{fmt_pct(item.get('min_repeat_prefill_gain_pct'))} | "
                f"{fmt_pct(item.get('min_generation_gain_pct'))} | "
                f"{fmt_num(item.get('tensor_standard_worst_rms'))} | "
                f"{fmt_num(item.get('tensor_standard_worst_top20_abs'))} | "
                f"{fmt_num(item.get('coverage_tensor_standard_worst_rms'))} | "
                f"{fmt_num(item.get('coverage_tensor_standard_worst_top20_abs'))} |"
            )
        lines.append("")

        lines.extend(
            [
                "## Candidate Promotion Failures",
                "",
                "| Run | Candidate | Env | First failure |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in index["candidates"]:
            failures = item.get("failures") or []
            if failures:
                lines.append(
                    "| "
                    f"`{markdown_escape(item['run'])}` | "
                    f"`{markdown_escape(item['candidate'])}` | "
                    f"`{markdown_escape(env_label(item.get('env')))}` | "
                    f"{markdown_escape(failures[0])} |"
                )
        lines.append("")

    if index["metal_tensor_benches"]:
        lines.extend(
            [
                "## Metal Tensor Bench Charts",
                "",
                "| Run | Contexts | Tensor prefill vs Standard | Tensor gen vs Standard | Quality prefill vs Standard | Chart |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in sorted(index["metal_tensor_benches"], key=lambda row: row["run"], reverse=True)[:top]:
            lines.append(
                "| "
                f"`{markdown_escape(item['run'])}` | "
                f"{len(item.get('contexts') or [])} | "
                f"{fmt_pct(item.get('min_tensor_prefill_vs_standard_pct'))}.."
                f"{fmt_pct(item.get('max_tensor_prefill_vs_standard_pct'))} | "
                f"{fmt_pct(item.get('min_tensor_gen_vs_standard_pct'))}.."
                f"{fmt_pct(item.get('max_tensor_gen_vs_standard_pct'))} | "
                f"{fmt_pct(item.get('min_quality_prefill_vs_standard_pct'))}.."
                f"{fmt_pct(item.get('max_quality_prefill_vs_standard_pct'))} | "
                f"`{markdown_escape(item.get('chart') or 'n/a')}` |"
            )
        lines.append("")

    if index["drift_gates"]:
        lines.extend(
            [
                "## Drift Gates",
                "",
                "| Run | Kind | Gate OK | Env | Top1 | Greedy | Min top20 | Worst RMS | RMS case/frontier | Worst top20 abs | Top20 case/frontier |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for item in sorted(index["drift_gates"], key=lambda row: row["run"], reverse=True)[:top]:
            lines.append(
                "| "
                f"`{markdown_escape(item['run'])}` | "
                f"{markdown_escape(item.get('kind') or 'n/a')} | "
                f"{bool_label(item.get('gate_ok'))} | "
                f"`{markdown_escape(env_label(item.get('env')))}` | "
                f"{fmt_num(item.get('tensor_standard_top1'))} | "
                f"{fmt_num(item.get('tensor_standard_greedy'))} | "
                f"{fmt_num(item.get('tensor_standard_min_top20'))}/20 | "
                f"{fmt_num(item.get('tensor_standard_worst_rms'))} | "
                f"{markdown_escape(item.get('tensor_standard_worst_rms_case') or 'n/a')} | "
                f"{fmt_num(item.get('tensor_standard_worst_top20_abs'))} | "
                f"{markdown_escape(item.get('tensor_standard_worst_top20_abs_case') or 'n/a')} |"
            )
        lines.append("")

    if index["mpp_compares"]:
        lines.extend(
            [
                "## Comparator Summaries",
                "",
                "| Run | Comparisons | Breaches | Worst max abs | Route | Module | Worst RMS |",
                "| --- | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for item in top_items(index["mpp_compares"], "worst_max_abs", top):
            lines.append(
                "| "
                f"`{markdown_escape(item['run'])}` | "
                f"{fmt_num(item.get('count'))} | "
                f"{fmt_num(item.get('threshold_breaches'))} | "
                f"{fmt_num(item.get('worst_max_abs'))} | "
                f"`{markdown_escape(item.get('worst_max_abs_route') or 'n/a')}` | "
                f"`{markdown_escape(item.get('worst_max_abs_module') or 'n/a')}` | "
                f"{fmt_num(item.get('worst_rms'))} |"
            )
        lines.append("")

    if index["stage_profiles"]:
        lines.extend(
            [
                "## Stage Profiles",
                "",
                "| Run | Prefill t/s | Top stage | Stage ms | Top Q8 shape | Q8 ms | Top Flash shape | Flash ms |",
                "| --- | ---: | --- | ---: | --- | ---: | --- | ---: |",
            ]
        )
        for item in sorted(index["stage_profiles"], key=lambda row: row["run"], reverse=True)[:top]:
            lines.append(
                "| "
                f"`{markdown_escape(item['run'])}` | "
                f"{fmt_num(item.get('prefill_tps'))} | "
                f"`{markdown_escape(item.get('top_stage') or 'n/a')}` | "
                f"{fmt_num(item.get('top_stage_ms'))} | "
                f"`{markdown_escape(item.get('top_q8_shape') or 'n/a')}` | "
                f"{fmt_num(item.get('top_q8_ms'))} | "
                f"`{markdown_escape(item.get('top_flash_shape') or 'n/a')}` | "
                f"{fmt_num(item.get('top_flash_ms'))} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("speed-bench/local-runs"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, help="write Markdown index here")
    parser.add_argument("--json-output", type=Path, help="write JSON index here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    root = args.root
    index = collect(root)
    markdown = render_markdown(index, args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
