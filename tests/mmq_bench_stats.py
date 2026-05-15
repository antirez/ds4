#!/usr/bin/env python3
"""Statistics helper for the Sprint 0 bench harness.

Two subcommands:

  aggregate DIR                  Per-ctx median + IQR for one bench run dir.
  compare DIR_A DIR_B            Per-ctx Mann-Whitney U between two configs.

The aggregator's variance acceptance gate is:
  IQR/2 < 1% of the median, for both prefill_tps and gen_tps, at ctx=2048
  with gen_tokens=128.

The comparator declares a winner only when Mann-Whitney p < 0.05 AND
|median delta| / median_A > 1%. Anything weaker is reported as inconclusive.

Pure stdlib: avoid scipy because the pod environment is not guaranteed to
have it. The Mann-Whitney U implementation here is the small-N exact form
when N <= 20 and the asymptotic normal approximation otherwise. Sufficient
for our 5-vs-5 comparisons.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _percentile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _read_combined(out_dir: Path):
    combined = out_dir / "combined.csv"
    if not combined.is_file():
        sys.exit(f"error: {combined} not found. Did run_mmq_bench.sh finish?")
    rows = []
    with combined.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "run": int(r["run"]),
                "ctx": int(r["ctx_tokens"]),
                "prefill_tps": float(r["prefill_tps"]),
                "gen_tps": float(r["gen_tps"]),
            })
    return rows


def _by_ctx(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r["ctx"]].append(r[key])
    return dict(sorted(out.items()))


def cmd_aggregate(args):
    out_dir = Path(args.dir)
    rows = _read_combined(out_dir)
    if not rows:
        sys.exit("error: combined.csv has no rows")
    n_runs = max(r["run"] for r in rows)
    print(f"# aggregate: {out_dir}  (runs={n_runs})")
    print(f"# {'ctx':>6}  {'metric':<12}  {'median':>9}  {'iqr':>9}  {'iqr/2_pct':>10}  {'min':>9}  {'max':>9}  pass")
    headline_pass = True
    for metric in ("prefill_tps", "gen_tps"):
        by_ctx = _by_ctx(rows, metric)
        for ctx, vals in by_ctx.items():
            med = statistics.median(vals)
            iqr = _percentile(vals, 0.75) - _percentile(vals, 0.25)
            half_iqr_pct = (iqr / 2.0) / med * 100.0 if med else float("inf")
            ok = "Y" if half_iqr_pct < 1.0 else "n"
            print(f"  {ctx:>6}  {metric:<12}  {med:>9.2f}  {iqr:>9.3f}  {half_iqr_pct:>9.3f}%  {min(vals):>9.2f}  {max(vals):>9.2f}  {ok}")
            if ctx == 2048 and half_iqr_pct >= 1.0:
                headline_pass = False
    print()
    if headline_pass:
        print("ACCEPT: ctx=2048 variance is below 1% for both metrics.")
        return 0
    print("REJECT: ctx=2048 variance is at or above 1%. Re-run with more iterations or pin clocks.")
    return 1


def _mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U.

    Returns (U, p). Uses exact distribution when m + n <= 20, otherwise the
    normal approximation with tie correction.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    r_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u_a = r_a - m * (m + 1) / 2.0
    u_b = m * n - u_a
    u = min(u_a, u_b)

    if m + n <= 20 and m <= 10 and n <= 10:
        # Exact via enumeration of rank assignments. m,n <= 10 keeps cost low.
        from math import comb
        total = comb(m + n, m)
        count_le = 0
        for mask in _iter_choose(m + n, m):
            ranks_a = [k + 1 for k in mask]
            u_obs = sum(ranks_a) - m * (m + 1) / 2.0
            if min(u_obs, m * n - u_obs) <= u:
                count_le += 1
        p = count_le / total
        if p > 1.0:
            p = 1.0
        return u, p

    # Normal approximation with tie correction.
    mu = m * n / 2.0
    counts = []
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        counts.append(j - i + 1)
        i = j + 1
    tie_term = sum(c * (c * c - 1) for c in counts)
    N = m + n
    sigma2 = m * n / 12.0 * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else 0.0
    if sigma2 <= 0:
        return u, 1.0
    z = (u - mu) / math.sqrt(sigma2)
    p = 2.0 * (1.0 - _phi(abs(z)))
    return u, max(min(p, 1.0), 0.0)


def _iter_choose(n, k, start=0, chosen=None):
    if chosen is None:
        chosen = []
    if len(chosen) == k:
        yield tuple(chosen)
        return
    remaining = k - len(chosen)
    for i in range(start, n - remaining + 1):
        chosen.append(i)
        yield from _iter_choose(n, k, i + 1, chosen)
        chosen.pop()


def _phi(z):
    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cmd_compare(args):
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    rows_a = _read_combined(dir_a)
    rows_b = _read_combined(dir_b)
    print(f"# compare: A={dir_a}  B={dir_b}")
    print(f"# {'ctx':>6}  {'metric':<12}  {'med_a':>9}  {'med_b':>9}  {'delta_pct':>10}  {'p':>8}  {'verdict':<20}")
    for metric in ("prefill_tps", "gen_tps"):
        by_a = _by_ctx(rows_a, metric)
        by_b = _by_ctx(rows_b, metric)
        for ctx in sorted(set(by_a) | set(by_b)):
            if ctx not in by_a or ctx not in by_b:
                continue
            a = by_a[ctx]
            b = by_b[ctx]
            med_a = statistics.median(a)
            med_b = statistics.median(b)
            delta_pct = (med_b - med_a) / med_a * 100.0 if med_a else float("inf")
            _, p = _mannwhitney_u(a, b)
            sig = p < 0.05
            substantive = abs(delta_pct) > 1.0
            if sig and substantive:
                verdict = "B wins" if delta_pct > 0 else "A wins"
            elif sig and not substantive:
                verdict = "stat-sig but <1%"
            elif substantive and not sig:
                verdict = "directional"
            else:
                verdict = "inconclusive"
            print(f"  {ctx:>6}  {metric:<12}  {med_a:>9.2f}  {med_b:>9.2f}  {delta_pct:>9.2f}%  {p:>8.4f}  {verdict:<20}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("aggregate", help="Per-ctx median + IQR for one run dir.")
    a1.add_argument("dir")
    a1.set_defaults(fn=cmd_aggregate)
    a2 = sp.add_parser("compare", help="Per-ctx Mann-Whitney U between two run dirs.")
    a2.add_argument("dir_a")
    a2.add_argument("dir_b")
    a2.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
