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
    hardware: str
    model: str
    best_gen: float
    gen_at_target_ctx: float | None
    avg_gen: float
    best_prefill: float
    prefill_at_target_ctx: float | None
    avg_prefill: float


def benchmark_labels(path: Path) -> tuple[str, str]:
    name_overrides = {
        "gb10": ("NVIDIA DGX Spark / GB10", "DeepSeek V4 Flash q2"),
        "m2_ultra": ("Apple M2 Ultra", "DeepSeek V4 Flash q2"),
        "m4_max": ("Apple M4 Max", "DeepSeek V4 Flash q2"),
        "pro_model_m3_ultra": ("Apple M3 Ultra", "DeepSeek V4 PRO q2"),
    }
    if path.stem in name_overrides:
        return name_overrides[path.stem]

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
    return " ".join(replacements.get(word.lower(), word) for word in words), "Unspecified model"


def fmt_tps(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


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
    hardware, model = benchmark_labels(path)
    return BenchSummary(
        hardware=hardware,
        model=model,
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
    summaries = [read_summary(path) for path in csv_paths]
    generated = render_summary(summaries)
    README.write_text(replace_generated_section(README.read_text(encoding="utf-8"), generated), encoding="utf-8")


if __name__ == "__main__":
    main()
