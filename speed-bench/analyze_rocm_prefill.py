#!/usr/bin/env python3
"""Summarize existing DS4 stage logs without counting nested Q timings twice.

This is an attribution aid, not a throughput benchmark. Stage profiling inserts
synchronizations and may cover only selected layers. Analyze each log separately;
use unprofiled paired runs to establish a whole-model improvement.
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ROCm uses the shared graph's historical "metal" log prefix.
STAGE = re.compile(
    r"ds4: metal (?:(layer stage) part=(\w+)|(Q path stage)) "
    r"layer=(\d+) pos=(\d+) tokens=(\d+) ([\w_]+)=(\S+) ms"
)
CAP = re.compile(r"context buffers .*prefill_chunk=(\d+)")


def read_profile(lines, min_pos):
    timings = defaultdict(list)
    layers = set()
    batches = set()
    caps = set()
    for line in lines:
        cap = CAP.search(line)
        if cap:
            caps.add(int(cap[1]))
        match = STAGE.search(line)
        if not match:
            continue
        _, part, q_stage, layer, pos, tokens, stage, raw_ms = match.groups()
        layer, pos, tokens = int(layer), int(pos), int(tokens)
        # Decode and final-output timings are separate from prefill stages.
        if pos < min_pos or tokens <= 1 or (not q_stage and part not in ("attn", "ffn")):
            continue
        ms = float(raw_ms)
        if not math.isfinite(ms) or ms < 0:
            raise ValueError(f"invalid stage duration: {raw_ms}")
        family = "Q" if q_stage else "layer"
        name = stage if q_stage else f"{part}/{stage}"
        timings[(family, name)].append(ms)
        layers.add(layer)
        batches.add((pos, tokens))
    return timings, layers, batches, caps


def required_saving(target_tps_pct):
    if not math.isfinite(target_tps_pct) or target_tps_pct <= 0:
        raise ValueError("target TPS increase must be finite and positive")
    return 1.0 - 1.0 / (1.0 + target_tps_pct / 100.0)


def render_profile(path, profile, target_tps_pct):
    timings, layers, batches, caps = profile
    if not timings:
        raise ValueError("no matching prefill stages; check profiler flags and --min-pos")
    saving = required_saving(target_tps_pct)
    print(f"\n{path}")
    print(f"configured caps: {sorted(caps) or 'not logged'}; layers seen: {sorted(layers)}")
    shapes = Counter(tokens for _, tokens in batches)
    print("observed batches (unique positions): " + ", ".join(
        f"N={tokens}: {count}" for tokens, count in sorted(shapes.items())))
    print(f"Target +{target_tps_pct:g}% TPS requires {saving * 100:.3f}% less total time.")

    # Q timings overlap attention stages; pre_q also includes hc_pre/norm.
    # Never add this second profiler's durations to recorded layer time.
    for family in ("layer", "Q"):
        rows = [(name, values) for (kind, name), values in timings.items() if kind == family]
        if not rows:
            continue
        total = sum(sum(values) for _, values in rows)
        print(f"\n{family} stages: {total:.3f} ms recorded")
        if family == "Q":
            print("Q profiler detail includes pre_q setup and overlaps attention stages.")
            print("Shares below refer only to this profiler's recorded time.")
        print(f"{'stage':32s} {'calls':>7s} {'sum ms':>12s} {'share':>8s} {'stage cut*':>12s}")
        for name, values in sorted(rows, key=lambda row: sum(row[1]), reverse=True):
            elapsed = sum(values)
            share = elapsed / total if total else 0.0
            cut = saving / share if share else math.inf
            estimate = (f"{cut * 100:.1f}%" if cut <= 1 else "insufficient") if family == "layer" else "--"
            print(f"{name:32s} {len(values):7d} {elapsed:12.3f} {share * 100:7.2f}% {estimate:>12s}")
    print("\n* Hypothetical cut if recorded layer stages represented total runtime.")
    print("Filtered layers, unrecorded work and profiler synchronization invalidate that assumption.")
    print("output_proj includes both A and B; it does not isolate attn_output_b.")
    print("Use this table to choose a target, then verify with unprofiled paired TPS runs.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--min-pos", type=int, default=8192,
                        help="include batches starting at/after this position (default: 8192)")
    parser.add_argument("--target-tps-pct", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.min_pos < 0:
        parser.error("--min-pos must be nonnegative")
    try:
        required_saving(args.target_tps_pct)
        for path in args.logs:
            with path.open(errors="replace") as source:
                profile = read_profile(source, args.min_pos)
            render_profile(path, profile, args.target_tps_pct)
    except (OSError, ValueError) as exc:
        print(f"analyze_rocm_prefill: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
