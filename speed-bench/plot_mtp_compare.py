#!/usr/bin/env python3
"""Generate a DS4 speed-bench SVG comparing no-MTP and exact MTP CSVs."""

import argparse
import csv
import html
import math
from pathlib import Path


TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#334155"
PREFILL_NOMTP = "#2563eb"
PREFILL_MTP = "#60a5fa"
GEN_NOMTP = "#dc2626"
GEN_MTP = "#16a34a"


def nice_ceil(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1, 2, 2.5, 3, 4, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def nice_step(span, target_ticks):
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for step in (1, 2, 2.5, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def frange(start, stop, step):
    value = start
    while value <= stop + step * 0.001:
        yield round(value, 10)
        value += step


def fmt_tick(value):
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"ctx_tokens", "prefill_tps", "gen_tps"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows.append({
                "ctx": int(row["ctx_tokens"]),
                "prefill": float(row["prefill_tps"]),
                "gen": float(row["gen_tps"]),
            })
    if len(rows) < 2:
        raise SystemExit(f"{path}: need at least two rows")
    rows.sort(key=lambda row: row["ctx"])
    return rows


def polyline(rows, key, x_min, x_max, y_max, plot):
    left, top, width, height = plot
    pts = []
    for row in rows:
        x = left + (row["ctx"] - x_min) / (x_max - x_min) * width
        y = top + height - row[key] / y_max * height
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)


def render_svg(nomtp, mtp, title, width, height):
    margin_left = 82
    margin_right = 82
    margin_top = 66
    margin_bottom = 72
    plot = (
        margin_left,
        margin_top,
        width - margin_left - margin_right,
        height - margin_top - margin_bottom,
    )
    left, top, plot_width, plot_height = plot
    right = left + plot_width
    bottom = top + plot_height

    ctx_values = [row["ctx"] for row in nomtp + mtp]
    x_min = 0
    x_max = max(ctx_values)
    prefill_max = nice_ceil(max(row["prefill"] for row in nomtp + mtp) * 1.05)
    gen_max = nice_ceil(max(row["gen"] for row in nomtp + mtp) * 1.05)

    x_step = nice_step(x_max - x_min, 6)
    x_ticks = list(frange(math.ceil(x_min / x_step) * x_step, x_max, x_step))
    prefill_ticks = list(frange(0, prefill_max, nice_step(prefill_max, 5)))
    gen_ticks = list(frange(0, gen_max, nice_step(gen_max, 5)))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 26px; font-weight: 700; fill: #1f2933; }",
        ".axis-label { font-size: 14px; font-weight: 600; fill: #334155; }",
        ".tick { font-size: 12px; fill: #64748b; }",
        ".legend { font-size: 13px; font-weight: 600; fill: #1f2933; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]

    for tick in prefill_ticks:
        y = bottom - tick / prefill_max * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{fmt_tick(tick)}</text>')

    for tick in gen_ticks:
        y = bottom - tick / gen_max * plot_height
        parts.append(f'<text class="tick" x="{right + 12}" y="{y + 4:.2f}" text-anchor="start">{fmt_tick(tick)}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / (x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 24}" text-anchor="middle">{fmt_tick(tick)}</text>')

    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
        f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>',
        f'<text class="axis-label" x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle">ctx size</text>',
        f'<text class="axis-label" x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">prefill t/s</text>',
        f'<text class="axis-label" x="{width - 22}" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 22} {top + plot_height / 2:.1f})">generation t/s</text>',
        f'<polyline fill="none" stroke="{PREFILL_NOMTP}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline(nomtp, "prefill", x_min, x_max, prefill_max, plot)}"/>',
        f'<polyline fill="none" stroke="{PREFILL_MTP}" stroke-width="3" stroke-dasharray="8 6" stroke-linecap="round" stroke-linejoin="round" points="{polyline(mtp, "prefill", x_min, x_max, prefill_max, plot)}"/>',
        f'<polyline fill="none" stroke="{GEN_NOMTP}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline(nomtp, "gen", x_min, x_max, gen_max, plot)}"/>',
        f'<polyline fill="none" stroke="{GEN_MTP}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline(mtp, "gen", x_min, x_max, gen_max, plot)}"/>',
    ])

    legend_x = right - 220
    legend_y = top + 18
    legend_items = [
        (PREFILL_NOMTP, "", "prefill no-MTP"),
        (PREFILL_MTP, "8 6", "prefill exact MTP"),
        (GEN_NOMTP, "", "generation no-MTP"),
        (GEN_MTP, "", "generation exact MTP"),
    ]
    parts.append(f'<rect x="{legend_x - 14}" y="{legend_y - 18}" width="226" height="116" rx="6" fill="#ffffff" stroke="#cbd5e1"/>')
    for i, (color, dash, label) in enumerate(legend_items):
        y = legend_y + i * 26
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 16}" y2="{y}" stroke="{color}" stroke-width="4"{dash_attr}/>')
        parts.append(f'<text class="legend" x="{legend_x + 26}" y="{y + 5}">{html.escape(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nomtp", type=Path, required=True)
    parser.add_argument("--mtp", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--title", default="GB10 No-MTP vs Exact MTP t/s")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    args.output.write_text(
        render_svg(read_rows(args.nomtp), read_rows(args.mtp), args.title, args.width, args.height),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
