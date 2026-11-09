#!/usr/bin/env python3
"""Offline cache simulation over Flint routing logs (docs/RECURSIVE_FLINT.md
step 4: the miss fraction per cache size decides whether plain LRU is enough).

A log is written by decode with DS4_FLINT_ROUTE_LOG=FILE: per token a u32
position followed by 49 x 8 u32 (per layer the six selected expert ids in
words 0..5; dense layers are zero).  Several logs are concatenated in order.
For each cache size the script replays the selections through an exact LRU
over (layer, expert) keys and prints the steady-state miss fraction (after a
warm-up of the first quarter of the tokens), the bytes per token that implies,
and the decode time the step-4 speed model predicts at the given SSD rate.

    tests/flint/route_sim.py LOG [LOG ...] [--gb 4 8 12 16 20] [--ssd-gbps 5] [--compute-ms 7.5]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np

PACK = 1769472
N_LAYER = 49
N_EXPERT = 256
ALL_EXPERTS = 47 * 256


def load(paths):
    rows = []
    for p in paths:
        data = np.fromfile(p, dtype=np.uint32)
        per = 1 + N_LAYER * 8
        n = len(data) // per
        rec = data[: n * per].reshape(n, per)
        rows.append(rec)
    return np.concatenate(rows) if rows else np.zeros((0, 1 + N_LAYER * 8), np.uint32)


def simulate(tokens, slots, warm):
    lru = OrderedDict()
    misses = 0
    counted = 0
    for t, row in enumerate(tokens):
        ids = row[1:].reshape(N_LAYER, 8)[:, :6]
        for layer in range(2, N_LAYER):
            for e in ids[layer]:
                key = layer * N_EXPERT + int(e)
                if key in lru:
                    lru.move_to_end(key)
                else:
                    if len(lru) >= slots:
                        lru.popitem(last=False)
                    lru[key] = True
                    if t >= warm:
                        misses += 1
                if t >= warm:
                    counted += 1
    return misses, counted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--gb", type=float, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--ssd-gbps", type=float, default=5.0)
    ap.add_argument("--compute-ms", type=float, default=7.5, help="resident decode ms/token")
    ap.add_argument("--relaunch-ms", type=float, default=0.3, help="cost of one abort/relaunch")
    args = ap.parse_args()
    tokens = load(args.logs)
    n = len(tokens)
    if n < 8:
        raise SystemExit("need at least 8 logged tokens")
    warm = n // 4
    # distinct experts touched in the whole log, per layer average
    ids = tokens[:, 1:].reshape(n, N_LAYER, 8)[:, 2:, :6]
    distinct = len({(l, int(e)) for t in range(n) for l in range(47) for e in ids[t, l]})
    print("%d tokens from %d log(s); %d distinct (layer, expert) pairs touched of %d (%.1f%%); warm-up %d tokens" % (
        n, len(args.logs), distinct, ALL_EXPERTS, 100.0 * distinct / ALL_EXPERTS, warm))
    print("%8s %7s %10s %10s %12s %12s %10s" % ("cache", "slots", "miss frac", "MB/token", "miss layers", "read ms/tok", "tok/s est"))
    for gb in args.gb:
        slots = int(gb * (1 << 30) // PACK)
        misses, counted = simulate(tokens, min(slots, ALL_EXPERTS), warm)
        frac = misses / counted if counted else 0.0
        mb = frac * 282 * PACK / 1e6
        # a layer relaunches when any of its six missed: estimate from the per-selection miss rate
        p_layer = 1 - (1 - frac) ** 6
        read_ms = mb / 1e3 / args.ssd_gbps * 1e3
        total = args.compute_ms + read_ms + 47 * p_layer * args.relaunch_ms
        print("%6.1f GB %7d %10.3f %10.1f %12.1f %12.1f %10.0f" % (gb, slots, frac, mb, 47 * p_layer, read_ms, 1000.0 / total))


if __name__ == "__main__":
    main()
