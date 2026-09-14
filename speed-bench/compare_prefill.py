#!/usr/bin/env python3
"""Compare repeated pure-prefill ds4-bench CSVs from two revisions.

Use identical model, device, prompt, chunk and memory settings, with warmups
discarded and balanced run order. CSVs do not encode those settings: the
caller must verify them. The reported target is an observed median criterion,
not a confidence interval or a replacement for numerical validation.
"""
# SPDX-License-Identifier: MIT
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_run(path):
    rows = []
    previous = 0
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing ds4-bench CSV columns")
        for line, row in enumerate(reader, 2):
            try:
                context = int(row["ctx_tokens"])
                tokens = int(row["prefill_tokens"])
                rate = float(row["prefill_tps"])
                generated = int(row["gen_tokens"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: invalid numeric value") from error
            if generated != 0:
                raise ValueError(f"{path}:{line}: use --gen-tokens 0 for pure prefill")
            if tokens <= 0 or context - previous != tokens:
                raise ValueError(f"{path}:{line}: inconsistent incremental token counts")
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"{path}:{line}: prefill_tps must be positive and finite")
            rows.append((context, tokens, rate))
            previous = context
    if not rows:
        raise ValueError(f"{path}: empty benchmark")
    return rows


def summary(rates):
    return {"runs_tps": rates, "median_tps": statistics.median(rates),
            "min_tps": min(rates), "max_tps": max(rates)}


def compare(baseline_paths, candidate_paths, target_percent=8.0):
    if not math.isfinite(target_percent) or target_percent < 0:
        raise ValueError("target percentage must be finite and nonnegative")
    if len(baseline_paths) != len(candidate_paths) or len(baseline_paths) < 2:
        raise ValueError("provide the same number of runs per revision, at least two")
    all_paths = [Path(path).resolve() for path in baseline_paths + candidate_paths]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("each CSV must name a distinct run")
    baseline = [read_run(path) for path in baseline_paths]
    candidate = [read_run(path) for path in candidate_paths]
    shape = [(ctx, tokens) for ctx, tokens, _ in baseline[0]]
    if any([(ctx, tokens) for ctx, tokens, _ in run] != shape
           for run in baseline + candidate):
        raise ValueError("all runs must have identical context frontiers and suffix lengths")

    def comparison(before, after):
        a, b = summary(before), summary(after)
        gain = 100.0 * (b["median_tps"] / a["median_tps"] - 1.0)
        return {"baseline": a, "candidate": b, "observed_speedup_percent": gain,
                "target_met_by_medians": gain >= target_percent}

    frontiers = []
    for i, (ctx, tokens) in enumerate(shape):
        item = comparison([run[i][2] for run in baseline], [run[i][2] for run in candidate])
        item.update(ctx_tokens=ctx, prefill_tokens=tokens)
        frontiers.append(item)

    def aggregate(run):
        # Each CSV row times a new suffix. Sum time, not token rates.
        return sum(tokens for _, tokens, _ in run) / math.fsum(
            tokens / rate for _, tokens, rate in run)

    return {"target_percent": target_percent, "runs_per_revision": len(baseline),
            "baseline_files": [str(Path(p).resolve()) for p in baseline_paths],
            "candidate_files": [str(Path(p).resolve()) for p in candidate_paths],
            "frontiers": frontiers,
            "aggregate": comparison([aggregate(run) for run in baseline],
                                    [aggregate(run) for run in candidate]),
            "note": "Median criterion only; inspect run spread and validate model logits separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    parser.add_argument("--target-percent", type=float, default=8.0)
    parser.add_argument("--json", action="store_true", help="print all per-run results as JSON")
    args = parser.parse_args()
    try:
        report = compare(args.baseline, args.candidate, args.target_percent)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print(f"Runs/revision: {report['runs_per_revision']}; target: +{report['target_percent']:g}%")
    print("frontier\tbaseline median [min,max] t/s\tcandidate median [min,max] t/s\tgain\ttarget (median)")
    for label, row in [(str(r["ctx_tokens"]), r) for r in report["frontiers"]] + [("total", report["aggregate"])]:
        a, b = row["baseline"], row["candidate"]
        print(f"{label}\t{a['median_tps']:.3f} [{a['min_tps']:.3f},{a['max_tps']:.3f}]\t"
              f"{b['median_tps']:.3f} [{b['min_tps']:.3f},{b['max_tps']:.3f}]\t"
              f"{row['observed_speedup_percent']:+.2f}%\t{row['target_met_by_medians']}")
    print(report["note"])


if __name__ == "__main__":
    main()
