#!/usr/bin/env python3
"""Analyze DS4 Pro prefill chunk sweep for M2 Ultra."""
import csv, glob, os, json

base = "/Users/shadow/Work/mlx/ds4/speed-bench/m2u_scan"
files = sorted(glob.glob(f"{base}/prefill_chunk_*.csv"))
baseline = "/Users/shadow/Work/mlx/ds4/speed-bench/m2_ultra.csv"  # default chunk=4096

def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ctx = int(r["ctx_tokens"])
            rows[ctx] = {
                "prefill": float(r["prefill_tps"]),
                "decode": float(r["gen_tps"]),
            }
    return rows

data = {}
for fp in files:
    chunk = int(os.path.basename(fp).split("_")[-1].split(".")[0])
    data[chunk] = load(fp)

base_data = load(baseline)
ctxs = sorted(data[4096].keys())

print("=== PREFILL t/s by chunk size (selected ctx) ===")
key_ctx = [2048, 8192, 16384, 32768, 49152, 65536]
hdr = "ctx      " + "".join(f"{c:>8}" for c in sorted(data.keys()))
print(hdr)
for ctx in key_ctx:
    if ctx not in ctxs: continue
    row = f"{ctx:<8} " + "".join(f"{data[c].get(ctx,{}).get('prefill',0):>8.1f}" for c in sorted(data.keys()))
    print(row)

print("\n=== best chunk per ctx (prefill) ===")
best = {}
for ctx in ctxs:
    scores = [(data[c][ctx]["prefill"], c) for c in data if ctx in data[c]]
    scores.sort(reverse=True)
    best[ctx] = scores[0]
for ctx in key_ctx:
    if ctx in best:
        tps, c = best[ctx]
        base_tps = data[4096][ctx]["prefill"]
        print(f"  ctx={ctx:>6}: best chunk={c:>5} @ {tps:.1f} t/s  (default4096: {base_tps:.1f}, delta {100*(tps/base_tps-1):+.2f}%)")

print("\n=== decode t/s sanity (should be ~invariant to chunk) ===")
for ctx in key_ctx:
    if ctx not in ctxs: continue
    vals = [data[c][ctx]["decode"] for c in sorted(data.keys()) if ctx in data[c]]
    print(f"  ctx={ctx:>6}: min {min(vals):.2f} max {max(vals):.2f} spread {100*(max(vals)/min(vals)-1):.2f}%")

out = {
    "chunks": sorted(data.keys()),
    "best_chunk_by_ctx": {str(k): {"chunk": best[k][1], "tps": best[k][0], "default4096_tps": data[4096][k]["prefill"]} for k in best},
    "full": {str(c): {str(ctx): data[c][ctx] for ctx in data[c]} for c in data},
}
with open(f"{base}/prefill_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved:", f"{base}/prefill_analysis.json")
