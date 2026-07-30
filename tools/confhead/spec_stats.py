#!/usr/bin/env python3
"""Aggregate DSpark fused-step stats from a ds4-server log (PULSAR_DSPARK_STATS=1).

Reports, whole-log and per-request (requests delimited by the server's
'completion ... prompt start' markers -- n_batch==1 is NOT a boundary, see
parse_requests):
  eff_tps     1000 * sum(committed+1) / sum(step_ms)   (spec throughput)
  tok/step    mean(committed+1)                        (tokens per round)
  alpha       sum(committed) / sum(n_batch-1)          (accepted / verified)
  rows/step   mean(n_batch)                            (verify budget actually spent)

'nodraft' fused steps print no pend= field and are excluded everywhere, so
eff_tps is over drafted steps only (~one nodraft per request at EOS);
symmetric across compared legs.

Never quote bare t/s from spec runs: spec throughput is stochastic across
seeds; compare eff_tps AND alpha AND tok/step at fixed seeds.

Usage: spec_stats.py <server.log> [--per-request] [--label NAME]
"""
import argparse, re, sys
import numpy as np

RX = re.compile(r"dspark fused n_batch=(\d+) committed=(\d+) pend=(\d+) step_ms=([\d.]+)")

def parse(path):
    out = []
    for m in RX.finditer(open(path).read()):
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
    return out

def parse_requests(path):
    """Fused lines grouped per request, delimited by the server's own
    'completion ... prompt start' markers. n_batch==1 is NOT a request
    boundary: pending drafts also reset mid-request (e.g. the periodic disk-KV
    checkpoint store), which run1 measured at ~17 extra n_batch==1 steps
    across 48 requests."""
    segs, cur = [], None
    for line in open(path):
        if "ds4-server: completion" in line and "prompt start" in line:
            cur = []
            segs.append(cur)
            continue
        m = RX.search(line)
        if m and cur is not None:
            cur.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
    return segs

def agg(lines):
    if not lines:
        return None
    nb = np.array([l[0] for l in lines], float)
    cm = np.array([l[1] for l in lines], float)
    ms = np.array([l[3] for l in lines], float)
    verified = np.maximum(nb - 1, 0)
    return {
        "steps": len(lines),
        "eff_tps": 1000.0 * (cm + 1).sum() / ms.sum(),
        "tok_step": (cm + 1).mean(),
        "alpha": cm.sum() / max(verified.sum(), 1),
        "rows_step": nb.mean(),
        "ms_step": ms.mean(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--per-request", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    lines = parse(args.log)
    a = agg(lines)
    if not a:
        print(f"{args.label or args.log}: no fused stats lines")
        return
    print(f"{args.label or args.log}: steps={a['steps']} eff_tps={a['eff_tps']:.2f} "
          f"tok/step={a['tok_step']:.2f} alpha={a['alpha']:.3f} "
          f"rows/step={a['rows_step']:.2f} ms/step={a['ms_step']:.1f}")
    if args.per_request:
        segs = [s for s in parse_requests(args.log) if s]
        for i, s in enumerate(segs):
            r = agg(s)
            print(f"  req{i:02d}: steps={r['steps']:4d} eff_tps={r['eff_tps']:6.2f} "
                  f"tok/step={r['tok_step']:.2f} alpha={r['alpha']:.3f} ms/step={r['ms_step']:.1f}")

if __name__ == "__main__":
    main()
