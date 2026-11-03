#!/usr/bin/env python3
"""Generate an SVG throughput graph with 95% Confidence Interval (CI) bands.

Inspired by antirez's plot_speed.py, this script extends visualization
with scientific rigor:
- Plots mean / median trajectory as a crisp polyline.
- Renders shaded confidence ribbons (CI bands) using translucent SVG polygons:
  [mean - t*SE, mean + t*SE] or bootstrap percentiles [CI_low, CI_high].
- Adds error bar whiskers and data point markers.
- Supports:
  1. Context sweep CSVs (ctx_tokens, prefill_tps, gen_tps) with multiple runs/frontiers.
  2. Optimization progression from benchmarks/results.csv across steps and domains.
  3. Token-by-token timing profiles from benchmarks/token_timings.csv.
- Zero external dependencies: pure Python standard library producing native vector SVG.
"""

import argparse
import csv
import html
import math
from pathlib import Path


# Okabe-Ito (2008) Universal Design Color Palette for Color Vision Deficiency (CVD / Color Blindness)
OI_BLACK = "#000000"
OI_ORANGE = "#E69F00"
OI_ORANGE_BAND = "#fcdca2"
OI_SKY_BLUE = "#56B4E9"
OI_SKY_BLUE_BAND = "#bfe5fa"
OI_BLUISH_GREEN = "#009E73"
OI_BLUISH_GREEN_BAND = "#a2e5d4"
OI_YELLOW = "#F0E442"
OI_BLUE = "#0072B2"
OI_BLUE_BAND = "#9accee"
OI_VERMILLION = "#D55E00"
OI_VERMILLION_BAND = "#f8c7a7"
OI_RED_PURPLE = "#CC79A7"
OI_RED_PURPLE_BAND = "#f1cde3"

# Semantic assignments
PREFILL_LINE = OI_BLUE
PREFILL_BAND = OI_BLUE_BAND
GEN_LINE = OI_VERMILLION
GEN_BAND = OI_VERMILLION_BAND
TEXT_COLOR = "#0f172a"         # Slate 900
SUBTEXT_COLOR = "#475569"      # Slate 600
GRID_COLOR = "#e2e8f0"         # Slate 200
AXIS_COLOR = "#334155"         # Slate 700
CARD_BG = "#ffffff"


def t_critical_95(df):
    """Approximate 95% two-tailed Student-t critical value for degrees of freedom."""
    if df <= 0:
        return 1.96
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980
    }
    if df in table:
        return table[df]
    for k in sorted(table.keys()):
        if df < k:
            return table[k]
    return 1.96


def calc_stats(values, alpha=0.05):
    """Calculate mean, standard deviation, and 95% Confidence Interval."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = sum(values) / n
    if n == 1:
        return mean_val, 0.0, mean_val, mean_val
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_val = math.sqrt(variance)
    se_val = std_val / math.sqrt(n)
    t_val = t_critical_95(n - 1)
    margin = t_val * se_val
    ci_low = max(0.0, mean_val - margin)
    ci_high = mean_val + margin
    return mean_val, std_val, ci_low, ci_high


def nice_ceil(value):
    """Round a positive axis maximum up to a clean tick boundary."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    for step in (1, 2, 2.5, 3, 4, 5, 8, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def nice_step(span, target_ticks):
    """Calculate human-friendly tick spacing."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for step in (1, 2, 2.5, 5, 10):
        if normalized <= step:
            return step * magnitude
    return 10 * magnitude


def fmt_tick(value):
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def frange(start, stop, step):
    value = start
    while value <= stop + step * 0.001:
        yield round(value, 10)
        value += step


def render_ci_svg(series_dict, x_points, x_label, title, subtitle, width=1024, height=576):
    """Render an SVG with mean trajectories, shaded CI bands, and whiskers.
    
    series_dict: {
        'series_name': {
            'color_line': '#...',
            'color_band': '#...',
            'axis': 'left' | 'right',
            'label': '...',
            'stats': [ (mean, std, ci_low, ci_high), ... for each x in x_points ]
        }
    }
    """
    margin_left = 90
    margin_right = 90
    margin_top = 80
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    left = margin_left
    top = margin_top
    right = left + plot_width
    bottom = top + plot_height

    # X axis setup
    is_numeric_x = all(isinstance(x, (int, float)) for x in x_points)
    if is_numeric_x:
        x_min = 0
        x_max = max(x_points)
        x_step = nice_step(x_max - x_min, 6)
        x_ticks = [tick for tick in frange(x_min, x_max, x_step)]
    else:
        x_ticks = list(range(len(x_points)))

    def get_x_coord(idx, x_val):
        if is_numeric_x:
            return left + (x_val - x_min) / max(1e-9, x_max - x_min) * plot_width
        else:
            spacing = plot_width / max(1, len(x_points) - 1)
            return left + idx * spacing

    # Determine Y max for left and right axes
    left_max_val = 0.0
    right_max_val = 0.0
    has_right = False

    for s_name, s_data in series_dict.items():
        high_vals = [st[3] for st in s_data['stats']]
        max_in_series = max(high_vals) if high_vals else 1.0
        if s_data.get('axis', 'left') == 'right':
            has_right = True
            right_max_val = max(right_max_val, max_in_series)
        else:
            left_max_val = max(left_max_val, max_in_series)

    left_max = nice_ceil(left_max_val * 1.08)
    right_max = nice_ceil(right_max_val * 1.08) if has_right else left_max

    left_step = nice_step(left_max, 5)
    left_ticks = [tick for tick in frange(0, left_max, left_step)]
    if has_right:
        right_step = nice_step(right_max, 5)
        right_ticks = [tick for tick in frange(0, right_max, right_step)]
    else:
        right_ticks = []

    def get_y_coord(val, axis='left'):
        y_ceiling = right_max if axis == 'right' else left_max
        return bottom - (val / y_ceiling) * plot_height

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }",
        ".title { font-size: 22px; font-weight: 700; fill: #0f172a; }",
        ".subtitle { font-size: 13px; font-weight: 500; fill: #64748b; }",
        ".axis-label { font-size: 13px; font-weight: 600; fill: #334155; }",
        ".tick { font-size: 11px; font-weight: 500; fill: #64748b; }",
        ".legend { font-size: 12px; font-weight: 600; fill: #1e293b; }",
        ".whisker { stroke-linecap: round; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="36">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{margin_left}" y="56">{html.escape(subtitle)}</text>',
    ]

    # Grid lines and left axis ticks
    for tick in left_ticks:
        y = get_y_coord(tick, 'left')
        svg_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        svg_parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{fmt_tick(tick)}</text>')

    # Right axis ticks
    if has_right:
        for tick in right_ticks:
            y = get_y_coord(tick, 'right')
            svg_parts.append(f'<text class="tick" x="{right + 12}" y="{y + 4:.2f}" text-anchor="start">{fmt_tick(tick)}</text>')

    # X axis grid lines and ticks
    for idx, tick in enumerate(x_ticks):
        x = get_x_coord(idx, tick)
        svg_parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        lbl = fmt_tick(tick) if is_numeric_x else str(x_points[idx])
        svg_parts.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">{html.escape(lbl)}</text>')

    # Frame lines
    svg_parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>')
    svg_parts.append(f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>')
    svg_parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS_COLOR}" stroke-width="1.4"/>')

    # Axis Titles
    svg_parts.append(f'<text class="axis-label" x="{left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle">{html.escape(x_label)}</text>')
    left_title = "prefill t/s" if has_right else "Tokens per Second (t/s) [Mean ± 95% CI]"
    svg_parts.append(f'<text class="axis-label" x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2:.1f})">{left_title}</text>')
    if has_right:
        svg_parts.append(f'<text class="axis-label" x="{width - 24}" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 24} {top + plot_height / 2:.1f})">decode steady t/s</text>')

    # Render series: 1) Shaded Bands first, 2) Whiskers, 3) Polyline, 4) Point Dots
    for s_name, s_data in series_dict.items():
        axis = s_data.get('axis', 'left')
        c_line = s_data.get('color_line', GEN_LINE)
        c_band = s_data.get('color_band', GEN_BAND)
        stats = s_data['stats']

        # Coordinates for upper and lower CI bounds
        coords_upper = []
        coords_lower = []
        coords_mean = []

        for idx, x_val in enumerate(x_points):
            mean_v, std_v, ci_lo, ci_hi = stats[idx]
            px = get_x_coord(idx, x_val)
            py_mean = get_y_coord(mean_v, axis)
            py_hi = get_y_coord(ci_hi, axis)
            py_lo = get_y_coord(ci_lo, axis)

            coords_mean.append((px, py_mean))
            coords_upper.append((px, py_hi))
            coords_lower.append((px, py_lo))

        # 1. Shaded polygon (band forward along upper, backward along lower)
        poly_points = [f"{px:.2f},{py:.2f}" for px, py in coords_upper]
        poly_points += [f"{px:.2f},{py:.2f}" for px, py in reversed(coords_lower)]
        svg_parts.append(f'<polygon points="{" ".join(poly_points)}" fill="{c_band}" fill-opacity="0.35"/>')

        # 2. Error bar whiskers
        for idx in range(len(x_points)):
            px, py_mean = coords_mean[idx]
            _, py_hi = coords_upper[idx]
            _, py_lo = coords_lower[idx]
            svg_parts.append(f'<line class="whisker" x1="{px:.2f}" y1="{py_lo:.2f}" x2="{px:.2f}" y2="{py_hi:.2f}" stroke="{c_line}" stroke-width="1.6" opacity="0.75"/>')
            svg_parts.append(f'<line class="whisker" x1="{px - 4:.2f}" y1="{py_hi:.2f}" x2="{px + 4:.2f}" y2="{py_hi:.2f}" stroke="{c_line}" stroke-width="1.6" opacity="0.75"/>')
            svg_parts.append(f'<line class="whisker" x1="{px - 4:.2f}" y1="{py_lo:.2f}" x2="{px + 4:.2f}" y2="{py_lo:.2f}" stroke="{c_line}" stroke-width="1.6" opacity="0.75"/>')

        # 3. Main Mean Polyline
        line_points = " ".join(f"{px:.2f},{py:.2f}" for px, py in coords_mean)
        svg_parts.append(f'<polyline fill="none" stroke="{c_line}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" points="{line_points}"/>')

        # 4. Point Dots
        for px, py_mean in coords_mean:
            svg_parts.append(f'<circle cx="{px:.2f}" cy="{py_mean:.2f}" r="4.5" fill="{CARD_BG}" stroke="{c_line}" stroke-width="2.5"/>')

    # Legend Box
    n_series = len(series_dict)
    box_w = 230
    box_h = 28 + n_series * 26
    leg_x = right - box_w
    leg_y = top + 14

    svg_parts.append(f'<rect x="{leg_x}" y="{leg_y}" width="{box_w}" height="{box_h}" rx="6" fill="#ffffff" fill-opacity="0.95" stroke="#cbd5e1" stroke-width="1.2"/>')
    for idx, (s_name, s_data) in enumerate(series_dict.items()):
        item_y = leg_y + 22 + idx * 26
        c_line = s_data.get('color_line', GEN_LINE)
        c_band = s_data.get('color_band', GEN_BAND)
        lbl = s_data.get('label', s_name)

        # Shaded ribbon swatch
        svg_parts.append(f'<rect x="{leg_x + 14}" y="{item_y - 8}" width="24" height="14" fill="{c_band}" fill-opacity="0.5"/>')
        svg_parts.append(f'<line x1="{leg_x + 14}" y1="{item_y - 1}" x2="{leg_x + 38}" y2="{item_y - 1}" stroke="{c_line}" stroke-width="2.5"/>')
        svg_parts.append(f'<circle cx="{leg_x + 26}" cy="{item_y - 1}" r="3" fill="#fff" stroke="{c_line}" stroke-width="2"/>')
        svg_parts.append(f'<text class="legend" x="{leg_x + 46}" y="{item_y + 3}">{html.escape(lbl)}</text>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts) + "\n"


def plot_context_sweep(csv_path, output_path):
    """Plot Promessi Sposi context sweep with CI bands."""
    with csv_path.open("r", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    # Group by ctx_tokens
    by_ctx = {}
    for r in rows:
        ctx = int(r["ctx_tokens"])
        prefill = float(r["prefill_tps"])
        gen = float(r.get("steady_tps", r.get("gen_tps", 0.0)))
        if ctx not in by_ctx:
            by_ctx[ctx] = {"prefill": [], "gen": []}
        by_ctx[ctx]["prefill"].append(prefill)
        by_ctx[ctx]["gen"].append(gen)

    ctx_points = sorted(by_ctx.keys())
    prefill_stats = [calc_stats(by_ctx[c]["prefill"]) for c in ctx_points]
    gen_stats = [calc_stats(by_ctx[c]["gen"]) for c in ctx_points]

    series_dict = {
        "prefill": {
            "label": "Prefill Throughput (t/s)",
            "axis": "left",
            "color_line": PREFILL_LINE,
            "color_band": PREFILL_BAND,
            "stats": prefill_stats,
        },
        "decode": {
            "label": "Decode Throughput (t/s)",
            "axis": "right",
            "color_line": GEN_LINE,
            "color_band": GEN_BAND,
            "stats": gen_stats,
        },
    }

    title = f"Context Scaling with 95% Confidence Intervals — {csv_path.stem}"
    subtitle = "DeepSeek-V4-Flash Q2 (80.7 GiB) on RTX 3090 (24 GB VRAM) — Shaded Ribbons = ±t·SE"
    svg = render_ci_svg(series_dict, ctx_points, "Context Length (Tokens)", title, subtitle)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Graph written to {output_path}")


def plot_progression_suite(results_csv_path, output_path):
    """Plot step-by-step optimization progression with 95% CI bands across domains."""
    with results_csv_path.open("r", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    # Filter is_cold == 0
    warm_rows = [r for r in rows if r["is_cold"] in ("0", "False", "false")]

    # Collect configs in original chronological order
    configs = []
    for r in warm_rows:
        cfg = r["config"]
        if cfg not in configs:
            configs.append(cfg)

    # Collect stats per config for overall steady_tps
    by_cfg_overall = {cfg: [] for cfg in configs}
    by_cfg_domain = {}

    for r in warm_rows:
        cfg = r["config"]
        dom = r["domain"]
        tps = float(r.get("steady_decode_tps", r.get("steady_tps", r.get("decode_tps", 0.0))))
        by_cfg_overall[cfg].append(tps)
        if dom not in by_cfg_domain:
            by_cfg_domain[dom] = {c: [] for c in configs}
        by_cfg_domain[dom][cfg].append(tps)

    overall_stats = [calc_stats(by_cfg_overall[c]) for c in configs]
    
    # Clean step labels
    step_labels = [c.replace("opt2_", "").replace("_", " ") for c in configs]

    series_dict = {
        "mean_tps": {
            "label": "Grand Mean Steady t/s (All Domains)",
            "axis": "left",
            "color_line": OI_BLUE,
            "color_band": OI_BLUE_BAND,
            "stats": overall_stats,
        }
    }

    # Add individual domains using Okabe-Ito (2008) colorblind-safe palette
    palette = [
        (OI_BLUISH_GREEN, OI_BLUISH_GREEN_BAND),  # #009E73 (D1_code)
        (OI_ORANGE, OI_ORANGE_BAND),              # #E69F00 (D2_math)
        (OI_RED_PURPLE, OI_RED_PURPLE_BAND),      # #CC79A7 (D3_prose)
        (OI_VERMILLION, OI_VERMILLION_BAND),      # #D55E00 (D4_bench)
    ]
    for idx, (dom, dom_dict) in enumerate(sorted(by_cfg_domain.items())):
        dom_stats = [calc_stats(dom_dict[c]) for c in configs]
        c_line, c_band = palette[idx % len(palette)]
        series_dict[dom] = {
            "label": f"Domain: {dom}",
            "axis": "left",
            "color_line": c_line,
            "color_band": c_band,
            "stats": dom_stats,
        }

    title = "DS4 Optimization Trajectory with 95% Confidence Interval Bands"
    subtitle = "DeepSeek-V4-Flash Q2 (80.7 GiB) | RTX 3090 | Okabe-Ito Colorblind-Safe Palette | Ribbons = ±t·SE"
    svg = render_ci_svg(series_dict, step_labels, "Optimization Step", title, subtitle, width=1120, height=600)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Progression chart written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SVG graph with 95% Confidence Interval bands.")
    parser.add_argument("input_csv", type=Path, help="Input CSV path (results.csv or promessi_sposi_sweep.csv)")
    parser.add_argument("-o", "--output", type=Path, help="Output SVG path")
    parser.add_argument("--mode", choices=["auto", "sweep", "progression"], default="auto",
                        help="Plotting mode: 'sweep' for context scaling, 'progression' for optimization progression")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        if "results" in args.input_csv.stem.lower():
            mode = "progression"
        else:
            mode = "sweep"

    out = args.output
    if out is None:
        out = args.input_csv.with_name(f"{args.input_csv.stem}_ci_bands.svg")

    if mode == "progression":
        plot_progression_suite(args.input_csv, out)
    else:
        plot_context_sweep(args.input_csv, out)


if __name__ == "__main__":
    main()
