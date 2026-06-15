#!/usr/bin/env python3
"""Update the generated benchmark summary in speed-bench/README.md."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


BEGIN_MARKER = "<!-- BEGIN GENERATED BENCHMARK SUMMARY -->"
END_MARKER = "<!-- END GENERATED BENCHMARK SUMMARY -->"
README = Path(__file__).with_name("README.md")
BENCH_DIR = Path(__file__).resolve().parent
METADATA = BENCH_DIR / "benchmarks.json"
REQUIRED_COLUMNS = {"ctx_tokens", "prefill_tps", "gen_tps"}
TARGET_CTX = 32768


@dataclass(frozen=True)
class BenchmarkMetadata:
    csv: str
    hardware: str
    backend: str
    model: str
    quant: str
    model_label: str
    prompt_file: str
    ctx_start: int
    ctx_max: int
    step_incr: int
    gen_tokens: int


@dataclass
class BenchSummary:
    hardware: str
    model: str
    best_gen: float
    gen_at_target_ctx: float | None
    avg_gen: float
    best_prefill: float
    prefill_at_target_ctx: float | None
    avg_prefill: float


def read_metadata() -> dict[str, BenchmarkMetadata]:
    try:
        data = json.loads(METADATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"{METADATA}: metadata file is required") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{METADATA}: invalid JSON: {exc}") from None

    try:
        benchmarks = data["benchmarks"]
    except (KeyError, TypeError):
        raise SystemExit(f"{METADATA}: expected a benchmarks list") from None
    if not isinstance(benchmarks, list):
        raise SystemExit(f"{METADATA}: expected a benchmarks list")

    by_csv: dict[str, BenchmarkMetadata] = {}
    for item in benchmarks:
        try:
            metadata = BenchmarkMetadata(**item)
        except TypeError as exc:
            raise SystemExit(f"{METADATA}: invalid benchmark metadata: {exc}") from None
        if metadata.csv in by_csv:
            raise SystemExit(f"{METADATA}: duplicate benchmark metadata for {metadata.csv}")
        by_csv[metadata.csv] = metadata
    return by_csv


def fmt_tps(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def read_summary(path: Path, metadata: BenchmarkMetadata) -> BenchSummary:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise SystemExit(f"{path}: missing CSV column(s): {missing_list}")

        for row in reader:
            rows.append(
                {
                    "ctx_tokens": int(row["ctx_tokens"]),
                    "prefill_tps": float(row["prefill_tps"]),
                    "gen_tps": float(row["gen_tps"]),
                }
            )

    if not rows:
        raise SystemExit(f"{path}: no benchmark rows")

    target_row = next((row for row in rows if row["ctx_tokens"] == TARGET_CTX), None)
    return BenchSummary(
        hardware=metadata.hardware,
        model=metadata.model_label,
        best_gen=max(row["gen_tps"] for row in rows),
        gen_at_target_ctx=target_row["gen_tps"] if target_row else None,
        avg_gen=sum(row["gen_tps"] for row in rows) / len(rows),
        best_prefill=max(row["prefill_tps"] for row in rows),
        prefill_at_target_ctx=target_row["prefill_tps"] if target_row else None,
        avg_prefill=sum(row["prefill_tps"] for row in rows) / len(rows),
    )


def render_summary(summaries: list[BenchSummary]) -> str:
    by_model = {}
    for summary in summaries:
        by_model.setdefault(summary.model, []).append(summary)
    model_groups = sorted(
        by_model.items(),
        key=lambda item: max(summary.best_gen for summary in item[1]),
        reverse=True,
    )
    lines = [
        BEGIN_MARKER,
        "## Benchmark Summary",
        "",
        "Generated from the CSV files in this directory by `python3 speed-bench/update_summary.py`.",
        "",
        f"`@ 32k ctx` means the row where `ctx_tokens` is `{TARGET_CTX}`.",
        "",
    ]
    for model, model_summaries in model_groups:
        lines.extend(
            [
                f"### {model}",
                "",
                "| Hardware | Best gen (t/s) | Gen @ 32k ctx (t/s) | Avg gen (t/s) | Best prefill (t/s) | Prefill @ 32k ctx (t/s) | Avg prefill (t/s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in sorted(model_summaries, key=lambda item: item.best_gen, reverse=True):
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary.hardware,
                        fmt_tps(summary.best_gen),
                        fmt_tps(summary.gen_at_target_ctx),
                        fmt_tps(summary.avg_gen),
                        fmt_tps(summary.best_prefill),
                        fmt_tps(summary.prefill_at_target_ctx),
                        fmt_tps(summary.avg_prefill),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend([END_MARKER, ""])
    return "\n".join(lines)


def replace_generated_section(readme: str, generated: str) -> str:
    begin = readme.find(BEGIN_MARKER)
    end = readme.find(END_MARKER)
    if begin == -1 and end == -1:
        return readme.rstrip() + "\n\n" + generated
    if begin == -1 or end == -1 or end < begin:
        raise SystemExit("README.md has mismatched generated summary markers")
    end += len(END_MARKER)
    return readme[:begin].rstrip() + "\n\n" + generated.rstrip() + readme[end:].rstrip() + "\n"


def main() -> None:
    csv_paths = sorted(BENCH_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"{BENCH_DIR}: no CSV files found")
    metadata = read_metadata()
    missing = [path.name for path in csv_paths if path.name not in metadata]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"{METADATA}: missing metadata for CSV file(s): {missing_list}")
    summaries = [read_summary(path, metadata[path.name]) for path in csv_paths]
    generated = render_summary(summaries)
    README.write_text(replace_generated_section(README.read_text(encoding="utf-8"), generated), encoding="utf-8")


if __name__ == "__main__":
    main()
