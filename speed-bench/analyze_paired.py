#!/usr/bin/env python3
"""Analyse the CSV produced by balanced_ab.sh.

Reports, per context point and overall:

  * the PAIRED ratio - each arm-A run divided by the arm-B run from the SAME
    (block, round), so the two measurements being divided are adjacent in time;
  * the NAIVE ratio - median of all A divided by median of all B, which is what
    an unpaired script computes;
  * a sign test for monotone drift within the session.

The naive figure is printed next to the paired one on purpose. If they differ
by an amount comparable to the effect you are reporting, the effect is not
established by the naive statistic.

Usage:
    ./analyze_paired.py paired.csv [--metric prefill_tps|gen_steady_tps]
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


def read_rows(path, metric):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            try:
                value = float(row[metric])
            except (TypeError, ValueError, KeyError):
                continue
            if value <= 0.0:
                continue
            try:
                ptok = int(row["prefill_tokens"])
            except (TypeError, ValueError, KeyError):
                ptok = None
            rows.append(
                {
                    "block": row["block"],
                    "round": row["round"],
                    "position": int(row["position"]),
                    "arm": row["arm"],
                    "ctx": int(row["ctx"]),
                    "prefill_tokens": ptok,
                    "value": value,
                }
            )
    return rows


def check_prefill_tokens(rows):
    """ds4-bench prefills only the step increment at each sweep point, so
    prefill_tokens equals ctx_tokens at the first point and nowhere else. Two
    runs can share a ctx and still be measuring different operations - a cold
    full prefill in one, a small increment onto a warm cache in the other.
    Averaging across that mismatch produces a number that describes neither."""
    seen = defaultdict(set)
    for r in rows:
        if r["prefill_tokens"] is not None:
            seen[r["ctx"]].add(r["prefill_tokens"])
    bad = {ctx: vals for ctx, vals in seen.items() if len(vals) > 1}
    if bad:
        lines = ["prefill_tokens is not consistent within a context point:"]
        for ctx in sorted(bad):
            lines.append(f"  ctx {ctx}: prefill_tokens "
                         f"{sorted(bad[ctx])}")
        lines.append("")
        lines.append("These rows measure different operations and must not be "
                     "pooled. Re-run with one --ctx-start and one step, or "
                     "split the CSV by sweep shape first.")
        sys.exit("\n".join(lines))
    return {ctx: next(iter(vals)) for ctx, vals in seen.items()}


def ci95(values):
    """Mean and half-width of the 95% CI. Falls back to the normal quantile
    for large n; uses a small-sample t table otherwise."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan")
    se = statistics.stdev(values) / math.sqrt(n)
    t_table = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145,
               20: 2.093, 30: 2.045, 60: 2.000}
    if n in t_table:
        t = t_table[n]
    else:
        smaller = [k for k in t_table if k <= n]
        t = t_table[max(smaller)] if smaller else 12.71
        if n > 60:
            t = 1.96
    return mean, t * se


def sign_test_drift(rows):
    """How often did a run come out lower than the previous run of the same arm
    at the same context? Under 'no drift' this is a fair coin."""
    series = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["block"], r["round"], r["position"])):
        series[(r["arm"], r["ctx"])].append(r["value"])
    down = total = 0
    for values in series.values():
        for prev, cur in zip(values, values[1:]):
            total += 1
            if cur < prev:
                down += 1
    return down, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--metric", default="prefill_tps",
                    choices=["prefill_tps", "gen_steady_tps"])
    args = ap.parse_args()

    rows = read_rows(args.csv_path, args.metric)
    if not rows:
        sys.exit(f"no usable rows for metric {args.metric} in {args.csv_path}")

    ptok_by_ctx = check_prefill_tokens(rows)

    arms = sorted({r["arm"] for r in rows})
    if len(arms) != 2:
        sys.exit(f"expected exactly 2 arms, found: {arms}")
    a_label, b_label = arms

    # Pair strictly within one (block, round): those two runs were adjacent in
    # time. Pairing across rounds would re-admit the drift the design removes.
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["block"], r["round"], r["ctx"])][r["arm"]] = r["value"]

    per_ctx = defaultdict(list)
    unpaired = 0
    for (_, _, ctx), arm_values in by_cell.items():
        if a_label in arm_values and b_label in arm_values:
            per_ctx[ctx].append(arm_values[a_label] / arm_values[b_label])
        else:
            unpaired += 1

    print(f"metric: {args.metric}   arms: {a_label} / {b_label}")
    print(f"paired samples: {sum(len(v) for v in per_ctx.values())}"
          f"   incomplete cells dropped: {unpaired}")
    print()
    print(f"{'ctx':>8}  {'pref':>6}  {'n':>3}  {'paired A/B':>12}  "
          f"{'95% CI':>18}  {'naive A/B':>10}")
    print("-" * 70)

    all_ratios = []
    for ctx in sorted(per_ctx):
        ratios = per_ctx[ctx]
        all_ratios.extend(ratios)
        mean, half = ci95(ratios)
        a_vals = [r["value"] for r in rows if r["ctx"] == ctx and r["arm"] == a_label]
        b_vals = [r["value"] for r in rows if r["ctx"] == ctx and r["arm"] == b_label]
        naive = statistics.median(a_vals) / statistics.median(b_vals)
        lo, hi = (mean - half) * 100 - 100, (mean + half) * 100 - 100
        ptok = ptok_by_ctx.get(ctx)
        ptok_s = str(ptok) if ptok is not None else "?"
        print(f"{ctx:>8}  {ptok_s:>6}  {len(ratios):>3}  "
              f"{mean * 100 - 100:>+11.2f}%  "
              f"[{lo:>+6.2f}%, {hi:>+6.2f}%]  {naive * 100 - 100:>+9.2f}%")

    mean, half = ci95(all_ratios)
    a_all = [r["value"] for r in rows if r["arm"] == a_label]
    b_all = [r["value"] for r in rows if r["arm"] == b_label]
    naive_all = statistics.median(a_all) / statistics.median(b_all)
    print("-" * 70)
    print(f"{'all':>8}  {'':>6}  {len(all_ratios):>3}  "
          f"{mean * 100 - 100:>+11.2f}%  "
          f"[{(mean - half) * 100 - 100:>+6.2f}%, {(mean + half) * 100 - 100:>+6.2f}%]  "
          f"{naive_all * 100 - 100:>+9.2f}%")

    down, total = sign_test_drift(rows)
    if total:
        # Two-sided sign test against p = 1/2.
        tail = sum(math.comb(total, k) for k in range(min(down, total - down) + 1))
        p = min(1.0, 2.0 * tail / (2.0 ** total))
        print()
        print(f"drift sign test: {down}/{total} consecutive runs slower than the "
              f"previous one, p = {p:.3g}")
        if p < 0.05:
            print("  Throughput is drifting within the session. The paired column "
                  "accounts for this; the naive column does not.")


if __name__ == "__main__":
    main()
