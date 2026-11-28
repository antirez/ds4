#!/usr/bin/env python3
"""Aggregate a DS4_METAL_ENCODER_TIMELINE file: per kernel count, total GPU us, mean us, total gap us; overall busy vs gap."""
import sys, collections
d = collections.defaultdict(lambda: [0, 0.0, 0.0]); tot = 0.0; gap = 0.0; n = 0
first = None; last = None
for l in open(sys.argv[1]):
    if not l.startswith("E "): continue
    f = l.split()
    dur, g, k = float(f[5]), float(f[6]), f[-1]
    if dur < 0: continue
    s, e = int(f[3]), int(f[4])
    d[k][0] += 1; d[k][1] += dur; d[k][2] += g
    tot += dur; gap += g; n += 1
print("dispatches %d, sum of kernel us %.0f, sum of gaps us %.0f" % (n, tot, gap))
for k, (c, t, g) in sorted(d.items(), key=lambda x: -x[1][1]):
    print("%-34s n=%6d total %9.0f us  mean %8.2f us  gap %8.0f us (%.1f%%)" % (k, c, t, t / c, g, 100 * t / tot))
