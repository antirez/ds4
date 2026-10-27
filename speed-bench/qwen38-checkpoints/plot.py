"""Regenerate benchmark charts: uv run --with matplotlib python plot.py."""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT / "results.csv").open() as source:
    ROWS = list(csv.DictReader(source))

plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})


def panel(ax, cases, labels, metric, title):
    for offset, build, label, color in [
        (-0.19, "baseline", "Before fixes (91095b7)", "#64748b"),
        (0.19, "fixed", "After fixes", "#2563eb"),
    ]:
        samples = [
            [float(r[metric]) for r in ROWS if r["case"] == case and r["build"] == build]
            for case in cases
        ]
        values = [statistics.median(s) for s in samples]
        bars = ax.bar([i + offset for i in range(len(cases))], values, 0.35,
                      color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:,.1f}" for v in values], padding=5, fontsize=10)
    ax.set(title=title, ylabel="Tokens / second", xticks=range(len(cases)), xticklabels=labels)
    ax.set_ylim(0, max(float(r[metric]) for r in ROWS if r["case"] in cases) * 1.22)
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def chart(name, panels, png_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, args in zip(axes, panels):
        panel(ax, *args)
    fig.suptitle("Qwen3.8 Flash Next IQ2 · Apple M3 Ultra", fontsize=17, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=2, frameon=False)
    fig.text(0.5, 0.025, "Median of two runs per build · ABBA order · zero-based axes",
             ha="center", color="#475569", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.82))
    fig.savefig(ROOT / f"{name}.svg", metadata={"Date": None})
    svg = ROOT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_dir / f"qwen38-{name}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, help="Also export PNG previews here")
    args = parser.parse_args()
    chart("context-throughput", [
        (["ctx8192", "ctx32768"], ["8K context", "32K context"], "prefill_tps", "Full-prefix prefill"),
        (["ctx8192", "ctx32768"], ["8K context", "32K context"], "decode_tps", "Teacher-forced decode · 128 tokens"),
    ], args.png_dir)
    chart("generation-throughput", [
        ([f"{c}-mtp{mode}" for c in ("hamlet", "fibonacci", "explanation")],
         ["Hamlet", "Fibonacci", "Networking"], "decode_tps", title)
        for mode, title in [(0, "Ordinary decode"), (1, "MTP decode")]
    ], args.png_dir)
