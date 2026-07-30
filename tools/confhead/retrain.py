#!/usr/bin/env python3
"""Confidence-head retrain, stage 2: train + per-regime calibration report.

Successor of temp/conf_retrain.py (the argmax-era trainer). What changed and
why:
  - Labels now come from sampled p/q acceptance (the engine's committed
    counts under temperature-matched drafting) across a temp x workload x
    depth mixture, aligned to the collector's manifest — the old head was
    trained on argmax proposals only, ~1k ctx only, which is the measured
    miscalibration being fixed.
  - Shipped weights come from a gguf (tools/confhead/gguf_io.py), not the HF
    snapshot (not present on this box).
  - NO BIAS TERM: the engine computes sigmoid(proj . x) with no bias
    (ds4_cuda_dspark.cu confidence kernel), so training a bias and telling
    tau to absorb it (the old script) wrecks calibration by construction.
    The head must be calibrated as the engine will actually evaluate it.
  - Judged by PER-REGIME CALIBRATION at the operating range (tau 0.2-0.5),
    not AUC: the conf3 lesson was that AUC was fine while calibration at the
    operating tau was the problem.

Dump/label mechanics are unchanged from the old pipeline: lean dump record r
(features: batch_ffn_cur rows + markov_w1[refined_ids], the exact conf-kernel
input) is labeled by fused stats line r+1's committed count, valid only when
line r+1 actually verified r's drafts (n_batch == 1 + pend_r); request
boundaries fail this automatically. Collect with PULSAR_DSPARK_CONF_SCHED=off.

Usage: retrain.py --run-dir temp/confhead/run1 [--gguf gguf/dspark-conf3-v1.gguf]
                  [--out temp/confhead/run1/proj_new.npy]
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_io import read_f32

NEMB, MK = 4096, 256
TAUS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)

def parse_dump(path):
    data = open(path, "rb").read()
    p = 0
    recs = []
    while p < len(data):
        pos, tok, nd = np.frombuffer(data, np.int32, 3, p); p += 12
        rid = np.frombuffer(data, np.int32, nd + 1, p).copy(); p += 4 * (nd + 1)
        ffn = np.frombuffer(data, np.float32, nd * NEMB, p).reshape(nd, NEMB).copy()
        p += 4 * nd * NEMB
        recs.append((int(nd), rid, ffn))
    return recs

def parse_log(path):
    lines = []
    rx = re.compile(r"dspark fused n_batch=(\d+) committed=(\d+) pend=(\d+) step_ms=([\d.]+)")
    for m in rx.finditer(open(path).read()):
        lines.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
    return lines

def auc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[yy == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def reliability(conf, yy, lo=0.0, hi=1.0, nbins=10):
    edges = np.linspace(lo, hi, nbins + 1)
    rows = []
    for b in range(nbins):
        m = (conf >= edges[b]) & (conf < edges[b + 1])
        if m.sum() == 0:
            rows.append((edges[b], edges[b + 1], 0, float("nan"), float("nan")))
        else:
            rows.append((edges[b], edges[b + 1], int(m.sum()),
                         float(conf[m].mean()), float(yy[m].mean())))
    return rows

def ece(conf, yy, nbins=10):
    tot, n = 0.0, len(yy)
    for lo, hi, cnt, mc, acc in reliability(conf, yy, nbins=nbins):
        if cnt:
            tot += cnt / n * abs(mc - acc)
    return tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+")
    ap.add_argument("--gguf", default="gguf/dspark-conf3-v1.gguf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1e-3)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir[0], "proj_new.npy")

    proj0, pdims = read_f32(args.gguf, "dspark.2.confidence_head.proj.weight")
    mkw, mdims = read_f32(args.gguf, "dspark.2.markov_head.markov_w1.weight")
    assert proj0.shape == (NEMB + MK,), pdims
    vocab = int(mdims[1])
    mkw = mkw.reshape(vocab, MK)
    print(f"shipped proj [4352] from {args.gguf}, markov_w1 ({vocab},{MK})")

    def depth_bucket(pt):
        return "1k" if pt < 2500 else "8k" if pt < 16000 else "30k"

    # ---- build the labeled dataset across run dirs ----
    X, y, R = [], [], []
    steps = []   # (x_start, nd, commit_capped, req) for prefix-trim simulation
    gen = []     # generating manifest entries, request-order, across runs
    from spec_stats import parse_requests
    for rd in args.run_dir:
        recs = parse_dump(os.path.join(rd, "dump.bin"))
        segs = [s for s in parse_requests(os.path.join(rd, "server.log")) if s]
        lines = [l for s in segs for l in s]
        req_id = [i for i, s in enumerate(segs) for _ in s]
        print(f"{rd}: {len(recs)} dump records (n_draft={recs[0][0]}), "
              f"{len(lines)} fused stats lines")
        assert len(lines) == len(recs), f"line/record mismatch: {len(lines)} vs {len(recs)}"
        n_req = len(segs)

        manifest = [json.loads(l) for l in open(os.path.join(rd, "manifest.jsonl"))]
        rgen = [e for e in manifest if e["status"] == "ok" and e["completion_tokens"] >= 1]
        print(f"  {n_req} request segments, {len(rgen)} generating manifest entries")
        assert n_req == len(rgen), "manifest/segment mismatch -- cannot attribute regimes"
        # effective temperature: with thinking on (run1 predates the think
        # field) the server forces temp>0 requests to the thinking defaults
        # (temp 1.0) -- label the regime by what actually ran.
        for e in rgen:
            if e.get("think", "on") == "on" and e["temperature"] > 0:
                e["eff_temp"] = 1.0
            else:
                e["eff_temp"] = e["temperature"]
        base = len(gen)
        gen.extend(rgen)

        for r in range(len(recs) - 1):
            nd, rid, ffn = recs[r]
            nb1, c1 = lines[r + 1][0], lines[r + 1][1]
            if nb1 != 1 + lines[r][2] and nb1 != 1 + nd:
                continue
            steps.append((len(y), nd, min(c1, nd), base + req_id[r]))
            for i in range(nd):
                X.append(np.concatenate([ffn[i], mkw[int(rid[i])]]))
                y.append(1.0 if c1 > i else 0.0)
                R.append(base + req_id[r])
    X = np.stack(X).astype(np.float32)
    y = np.array(y, np.float32)
    R = np.array(R)
    print(f"dataset: {len(y)} labeled positions over {len(steps)} steps, accept={y.mean():.1%}")

    # regime arrays per position
    ent = [gen[r] for r in R]
    depth = np.array([depth_bucket(e["prompt_tokens"]) for e in ent])
    workload = np.array([e["workload"] for e in ent])
    variant = np.array([e["variant"] for e in ent])
    temp = np.array([e["eff_temp"] for e in ent])
    sampled = temp > 0

    all_temps = sorted(set(temp.tolist()))
    print("\n=== collection mix (labeled positions per cell, accept rate) ===")
    for w in ("prose", "code", "structured"):
        for d in ("1k", "8k", "30k"):
            row = []
            for t in all_temps:
                m = (workload == w) & (depth == d) & (temp == t)
                row.append(f"t{t:g}: {m.sum():5d}/{y[m].mean():5.1%}" if m.sum() else f"t{t:g}: 0")
            print(f"  {w:10s} {d:3s}  " + "  ".join(row))

    # held-out = prompt variant 1 (no request overlap, every cell on both sides)
    tr = variant == 0
    te = variant == 1
    print(f"\ntrain {tr.sum()} / test {te.sum()} (split on prompt variant)")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    w0 = torch.from_numpy(proj0.copy())
    s_old = (Xt @ w0).numpy()   # engine-visible logit (no bias)

    Xtr = Xt[torch.from_numpy(tr)].to(dev)
    ytr = yt[torch.from_numpy(tr)].to(dev)
    w = torch.nn.Parameter(w0.clone().to(dev))
    w0d = w0.to(dev)
    opt = torch.optim.Adam([w], lr=3e-4)
    n = Xtr.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s0 in range(0, n, 8192):
            idx = perm[s0:s0 + 8192]
            z = Xtr[idx] @ w      # NO bias: engine computes sigmoid(proj . x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, ytr[idx]) \
                 + args.lam * ((w - w0d) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if epoch % 40 == 39:
            with torch.no_grad():
                s_new = (Xt.to(dev) @ w).cpu().numpy()
            print(f"epoch {epoch+1}: loss={tot/n:.4f} "
                  f"AUC train={auc(s_new[tr], y[tr]):.3f} test={auc(s_new[te], y[te]):.3f}")
    with torch.no_grad():
        s_new = (Xt.to(dev) @ w).cpu().numpy()

    c_old = 1 / (1 + np.exp(-s_old))
    c_new = 1 / (1 + np.exp(-s_new))

    def regime_report(name, m):
        m = m & te
        if m.sum() < 100:
            print(f"  {name}: n={m.sum()} (too small)")
            return
        print(f"  {name}: n={m.sum()} accept={y[m].mean():.1%}  "
              f"AUC old={auc(s_old[m], y[m]):.3f} new={auc(s_new[m], y[m]):.3f}  "
              f"ECE old={ece(c_old[m], y[m]):.3f} new={ece(c_new[m], y[m]):.3f}")
        print(f"    calibration at operating range (bins over conf 0.1-0.7):")
        ro = reliability(c_old[m], y[m], 0.1, 0.7, 6)
        rn = reliability(c_new[m], y[m], 0.1, 0.7, 6)
        for (lo, hi, no_, mo, ao), (_, _, nn_, mn, an) in zip(ro, rn):
            print(f"    [{lo:.1f},{hi:.1f})  old: n={no_:5d} conf={mo:5.2f} acc={ao:5.2f}"
                  f"   new: n={nn_:5d} conf={mn:5.2f} acc={an:5.2f}")

    print("\n=== AUC + calibration, per regime (held-out) ===")
    print(f"pooled: AUC old={auc(s_old[te], y[te]):.3f} new={auc(s_new[te], y[te]):.3f}  "
          f"ECE old={ece(c_old[te], y[te]):.3f} new={ece(c_new[te], y[te]):.3f}")
    regime_report("greedy (all depths)", ~sampled)
    regime_report("sampled (all depths)", sampled)
    for d in ("1k", "8k", "30k"):
        regime_report(f"sampled @{d}", sampled & (depth == d))
        regime_report(f"greedy  @{d}", ~sampled & (depth == d))

    # ---- prefix-trim tau simulation on held-out steps (engine semantics:
    # keep while conf >= tau, stop at first below; committed tokens preserved
    # = min(commit, keep)) ----
    print("\n=== tau operating table (held-out steps; prefix-trim simulation) ===")
    print("tau  | regime            | keep  | tokens kept | rows saved/step")
    step_te = [(s0, nd, cm, gen[rq]) for (s0, nd, cm, rq) in steps if gen[rq]["variant"] == 1]
    for tau in TAUS:
        for rname, sel in (("greedy", lambda e: e["eff_temp"] == 0),
                           ("sampled", lambda e: e["eff_temp"] > 0),
                           ("sampled@30k", lambda e: e["eff_temp"] > 0 and
                            depth_bucket(e["prompt_tokens"]) == "30k")):
            ks, kept, tot_c, saved = [], 0, 0, 0
            for (s0, nd, cm, e) in step_te:
                if not sel(e):
                    continue
                cn = c_new[s0:s0 + nd]
                k = 0
                while k < nd and cn[k] >= tau:
                    k += 1
                ks.append(k)
                kept += min(cm, k)
                tot_c += cm
                saved += nd - k
            if ks:
                print(f"{tau:.2f} | {rname:17s} | {np.mean(ks):5.2f} | "
                      f"{kept}/{tot_c} ({kept/max(tot_c,1):5.1%}) | {saved/len(ks):5.2f}")

    np.save(out, w.detach().cpu().numpy().astype(np.float32))
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
