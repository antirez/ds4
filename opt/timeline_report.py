#!/usr/bin/env python3
"""Aggregate DS4_METAL_ENCODER_TIMELINE files: per-kernel GPU time and gaps."""
import sys, re
from collections import defaultdict

def load(path):
    batches = {}  # seq -> (gpu_start, gpu_end)
    recs = defaultdict(list)  # seq -> list of recs
    with open(path) as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            p = line.split()
            if p[0] == "B":
                batches[int(p[1])] = (float(p[3]), float(p[4]))
            elif p[0] == "E":
                seq = int(p[1])
                dur_us = float(p[5]); gap_us = float(p[6])
                caller = p[7]; nd = int(p[8]); tg = p[9]; tpt = p[10]
                kernel = " ".join(p[11:])
                recs[seq].append((dur_us, gap_us, caller, nd, tg, tpt, kernel))
    return batches, recs

def main():
    path = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    seq_filter = int(sys.argv[3]) if len(sys.argv) > 3 else None
    batches, recs = load(path)
    # Drop the first batch (warmup/prefill) unless user picked one.
    seqs = sorted(recs.keys())
    if seq_filter is not None:
        seqs = [s for s in seqs if s == seq_filter]
    elif len(seqs) > 1:
        seqs = seqs[1:]
    per_kernel = defaultdict(lambda: [0.0, 0])  # total_us, count
    total_dur = 0.0; total_gap = 0.0
    per_caller = defaultdict(float)
    for s in seqs:
        for (dur, gap, caller, nd, tg, tpt, kernel) in recs[s]:
            per_kernel[kernel][0] += max(dur, 0.0)
            per_kernel[kernel][1] += 1
            total_dur += max(dur, 0.0)
            total_gap += max(gap, 0.0)
            per_caller[caller] += max(dur, 0.0)
    print(f"batches analyzed: {len(seqs)}  total kernel GPU: {total_dur/1000:.2f} ms  total inter-pass gap: {total_gap/1000:.2f} ms")
    print(f"{'kernel':<58} {'total us':>10} {'count':>7} {'us/each':>8} {'%':>6}")
    rows = sorted(per_kernel.items(), key=lambda kv: -kv[1][0])
    for k, (tot, cnt) in rows[:topn]:
        print(f"{k:<58} {tot:>10.0f} {cnt:>7} {tot/max(cnt,1):>8.1f} {100*tot/max(total_dur,1):>5.1f}%")
    if len(rows) > topn:
        rest = sum(t for _, (t, _) in rows[topn:])
        print(f"{'(remaining %d kernels)' % (len(rows)-topn):<58} {rest:>10.0f}")

if __name__ == "__main__":
    main()
