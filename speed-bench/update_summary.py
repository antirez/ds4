#!/usr/bin/env python3
"""Update the generated benchmark summary in speed-bench/README.md."""

import csv
from dataclasses import dataclass
from pathlib import Path


BEGIN_MARKER = "<!-- BEGIN GENERATED BENCHMARK SUMMARY -->"
END_MARKER = "<!-- END GENERATED BENCHMARK SUMMARY -->"
README = Path(__file__).with_name("README.md")
BENCH_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"ctx_tokens", "prefill_tps", "gen_tps"}
TARGET_CTX = 32768


@dataclass
class BenchSummary:
    name: str
    best_gen: float
    gen_at_target_ctx: float | None
    avg_gen: float
    best_prefill: float
    prefill_at_target_ctx: float | None
    avg_prefill: float


def display_name(path: Path) -> str:
    replacements = {
        "gb10": "GB10",
        "m2": "M2",
        "m3": "M3",
        "m4": "M4",
        "m5": "M5",
        "pro": "PRO",
        "max": "Max",
        "ultra": "Ultra",
    }
    words = path.stem.replace("-", "_").split("_")
    return " ".join(replacements.get(word.lower(), word) for word in words)


def fmt_tps(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} t/s"


def read_summary(path: Path) -> BenchSummary:
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
        name=display_name(path),
        best_gen=max(row["gen_tps"] for row in rows),
        gen_at_target_ctx=target_row["gen_tps"] if target_row else None,
        avg_gen=sum(row["gen_tps"] for row in rows) / len(rows),
        best_prefill=max(row["prefill_tps"] for row in rows),
        prefill_at_target_ctx=target_row["prefill_tps"] if target_row else None,
        avg_prefill=sum(row["prefill_tps"] for row in rows) / len(rows),
    )


def render_summary(summaries: list[BenchSummary]) -> str:
    summaries = sorted(summaries, key=lambda item: item.best_gen, reverse=True)
    lines = [
        BEGIN_MARKER,
        "## Benchmark Summary",
        "",
        "Generated from the CSV files in this directory by `python3 speed-bench/update_summary.py`.",
        "",
        f"`@ 32k ctx` means the row where `ctx_tokens` is `{TARGET_CTX}`.",
        "",
        "| Benchmark | Best gen | Gen @ 32k ctx | Avg gen | Best prefill | Prefill @ 32k ctx | Avg prefill |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.name,
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
    lines.extend(["", END_MARKER, ""])
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
    summaries = [read_summary(path) for path in csv_paths]
    generated = render_summary(summaries)
    README.write_text(replace_generated_section(README.read_text(encoding="utf-8"), generated), encoding="utf-8")


if __name__ == "__main__":
    main()
