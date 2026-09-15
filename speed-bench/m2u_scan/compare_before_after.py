#!/usr/bin/env python3
"""Compare M2 Ultra baseline vs m2ultra-pro patched results."""
import csv, json

def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ctx = int(r["ctx_tokens"])
            rows[ctx] = {
                "prefill": float(r["prefill_tps"]),
                "decode": float(r["gen_tps"]),
                "gen_first_ms": float(r["gen_first_ms"]),
            }
    return rows

before = load("/Users/shadow/Work/mlx/ds4/speed-bench/m2_ultra.csv")
after = load("/Users/shadow/Work/mlx/ds4/speed-bench/m2_ultra_after.csv")

ctxs = sorted(set(before) & set(after))
print(f"{'ctx':>7} | {'prefill before':>14} {'after':>8} {'d%':>7} | {'decode before':>13} {'after':>8} {'d%':>7}")
print("-" * 80)
tot_pb = tot_pa = tot_db = tot_da = 0.0
for ctx in ctxs:
    pb, pa = before[ctx]["prefill"], after[ctx]["prefill"]
    db, da = before[ctx]["decode"], after[ctx]["decode"]
    dp = 100 * (pa / pb - 1)
    dd = 100 * (da / db - 1)
    tot_pb += pb; tot_pa += pa; tot_db += db; tot_da += da
    print(f"{ctx:>7} | {pb:>14.2f} {pa:>8.2f} {dp:>+7.2f} | {db:>13.2f} {da:>8.2f} {dd:>+7.2f}")

n = len(ctxs)
print("-" * 80)
print(f"{'AVG':>7} | {tot_pb/n:>14.2f} {tot_pa/n:>8.2f} {100*(tot_pa/tot_pb-1):>+7.2f} | {tot_db/n:>13.2f} {tot_da/n:>8.2f} {100*(tot_da/tot_db-1):>+7.2f}")
