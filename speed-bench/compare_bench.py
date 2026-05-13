#!/usr/bin/env python3
"""Plot two or more ds4-bench CSV runs as a speed comparison chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "ctx_tokens",
    "prefill_tps",
    "gen_tps",
}


def read_run(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: empty CSV")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
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


def context_label(ctx: int) -> str:
    if ctx < 1024:
        return f"{ctx / 1024:g}k"
    rounded_k = round(ctx / 1024)
    if abs(ctx - rounded_k * 1024) <= max(4, ctx * 0.001):
        return f"{rounded_k}k"
    return f"{ctx / 1024:.1f}k"


def annotate_points(ax, xs: list[int], ys: list[float], color: str, dy: float) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=8,
            color=color,
            fontweight="medium",
        )


def plot_metric(
    ax,
    xs: list[int],
    labels: list[str],
    series: list[list[float]],
    metric_title: str,
    run_labels: list[str],
    annotate: bool,
) -> None:
    colors = ["#2563eb", "#64748b", "#ea580c", "#16a34a", "#9333ea", "#dc2626"]
    markers = ["o", "s", "^", "D", "P", "X"]

    for i, (values, label) in enumerate(zip(series, run_labels)):
        color = colors[i % len(colors)]
        ax.plot(
            xs,
            values,
            marker=markers[i % len(markers)],
            markersize=7,
            linewidth=2.4,
            color=color,
            label=label,
        )

    if len(series) == 2:
        ax.fill_between(xs, series[0], series[1], color=colors[1], alpha=0.08)

    ax.set_title(metric_title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Context Size")
    ax.set_ylabel("Tokens/sec")
    ax.set_xticks(xs, labels)
    ax.grid(True, color="#d1d5db", linewidth=0.9, alpha=0.65)
    ax.set_axisbelow(True)
    ax.margins(x=0.05, y=0.18)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")

    if len(series) == 2:
        gain_color = "#14532d"
        ymin, ymax = ax.get_ylim()
        label_y = ymin + (ymax - ymin) * 0.05
        for x, b, a in zip(xs, series[0], series[1]):
            gain = ((a / b) - 1.0) * 100.0 if b else 0.0
            ax.annotate(
                f"{gain:+.0f}%",
                (x, label_y),
                ha="center",
                va="center",
                fontsize=8,
                color=gain_color if gain >= 0 else "#991b1b",
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "#ecfdf5" if gain >= 0 else "#fef2f2",
                    "edgecolor": "#bbf7d0" if gain >= 0 else "#fecaca",
                    "linewidth": 0.8,
                },
            )

    if annotate:
        offsets = [-16, 8, 22, 36, 50, 64]
        for i, values in enumerate(series):
            annotate_points(ax, xs, values, colors[i % len(colors)], offsets[i % len(offsets)])


def default_run_labels(paths: list[Path], args: argparse.Namespace) -> list[str]:
    if len(paths) == 2 and not args.labels:
        return [args.before_label, args.after_label]
    if args.labels:
        if len(args.labels) != len(paths):
            raise SystemExit("--labels count must match the number of CSV runs")
        return args.labels
    return [path.stem for path in paths]


def build_chart(args: argparse.Namespace) -> None:
    if len(args.runs) < 2:
        raise SystemExit("provide at least two ds4-bench CSV files")
    runs = [read_run(path) for path in args.runs]
    run_labels = default_run_labels(args.runs, args)
    contexts = sorted(set.intersection(*(set(run) for run in runs)))
    if not contexts:
        raise SystemExit("the CSV files have no shared ctx_tokens values")

    x_positions = list(range(len(contexts)))
    labels = [context_label(ctx) for ctx in contexts]
    prefill_series = [[run[ctx]["prefill_tps"] for ctx in contexts] for run in runs]
    gen_series = [[run[ctx]["gen_tps"] for ctx in contexts] for run in runs]

    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "font.family": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), constrained_layout=True)
    fig.suptitle(args.title, fontsize=22, fontweight="bold", y=1.04)

    plot_metric(
        axes[0],
        x_positions,
        labels,
        prefill_series,
        "Prompt Processing Speed",
        run_labels,
        not args.no_values,
    )
    plot_metric(
        axes[1],
        x_positions,
        labels,
        gen_series,
        "Text Generation Speed",
        run_labels,
        not args.no_values,
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(run_labels), 4),
        frameon=True,
        fancybox=True,
        shadow=False,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
    )

    output = args.output
    if output.suffix.lower() != ".png":
        raise SystemExit(f"{output}: output must be a .png file")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", format="png")
    plt.close(fig)

    print(f"Wrote {output}")
    header = ["ctx"]
    for label in run_labels:
        safe = label.lower().replace(" ", "_")
        header.extend([f"prefill_{safe}", f"gen_{safe}"])
    for label in run_labels[1:]:
        safe = label.lower().replace(" ", "_")
        base = run_labels[0].lower().replace(" ", "_")
        header.extend([f"prefill_gain_{safe}_vs_{base}", f"gen_gain_{safe}_vs_{base}"])
    print(",".join(header))
    for idx, ctx in enumerate(contexts):
        row = [str(ctx)]
        base_prefill = prefill_series[0][idx]
        base_gen = gen_series[0][idx]
        for prefill, gen in zip(prefill_series, gen_series):
            row.extend([f"{prefill[idx]:.2f}", f"{gen[idx]:.2f}"])
        for prefill, gen in zip(prefill_series[1:], gen_series[1:]):
            prefill_gain = ((prefill[idx] / base_prefill) - 1.0) * 100.0 if base_prefill else 0.0
            gen_gain = ((gen[idx] / base_gen) - 1.0) * 100.0 if base_gen else 0.0
            row.extend([f"{prefill_gain:.1f}", f"{gen_gain:.1f}"])
        print(",".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-panel comparison chart from ds4-bench CSV files."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="ds4-bench CSV files; first is the baseline")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/tmp/ds4-bench-compare.png"),
        help="output chart path; must end in .png",
    )
    parser.add_argument("--before-label", default="standard kernel")
    parser.add_argument("--after-label", default="Metal Tensor")
    parser.add_argument("--labels", nargs="+", help="Labels for each CSV run.")
    parser.add_argument("--title", default="ds4-bench Speed Comparison")
    parser.add_argument("--no-values", action="store_true", help="hide per-point value labels")
    return parser.parse_args()


if __name__ == "__main__":
    build_chart(parse_args())
