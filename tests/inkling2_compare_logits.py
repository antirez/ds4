#!/usr/bin/env python3
"""Compare ds4 inkling2 top-K output with llama.cpp vectors .
usage: inkling2_compare_logits.py VECTORS.jsonl NAME OURS.jsonl [NPOS=4]
Position k (0..8) = logits after prompt + forced_ids[:k]; ours is keyed by absolute token index plen-1+k.
Reports per position: top-1 match, max |logprob diff| over llama's top-20 (and top-50)."""
import json, sys
vec, name, ours = sys.argv[1], sys.argv[2], sys.argv[3]
npos = int(sys.argv[4]) if len(sys.argv) > 4 else 4
row = [json.loads(l) for l in open(vec) if json.loads(l)["name"] == name][0]
plen = len(row["prompt_ids"])
mine = {}
for l in open(ours):
    d = json.loads(l); mine[d["pos"]] = {i: lp for i, lp in d["top"]}
worst20 = 0.0; ok = 0
for k, p in enumerate(row["positions"][:npos]):
    m = mine.get(plen - 1 + k)
    if m is None: print(name, k, "missing"); continue
    ref = p["top"]
    top1_ref = ref[0][0]; top1_mine = max(m, key=m.get)
    diffs = [abs(m[i] - lp) if i in m else float("inf") for i, lp in ref]
    d20 = max(diffs[:20]); d50 = max(diffs)
    ok += top1_ref == top1_mine; worst20 = max(worst20, d20)
    print("%s pos %d top1 %s (ref %d mine %d) max|d| top20 %.5f top50 %.5f" % (name, k, "ok " if top1_ref == top1_mine else "BAD", top1_ref, top1_mine, d20, d50))
print("%s: top-1 %d/%d, worst top-20 diff %.5f" % (name, ok, min(npos, len(row["positions"])), worst20))
