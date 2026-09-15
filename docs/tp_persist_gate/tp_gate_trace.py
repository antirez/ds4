#!/usr/bin/env python3
"""Decompose the persistent TP gate window from both ranks' trace CSVs.

    scripts/tp_gate_trace.py leader.csv worker.csv [--khz 100000]

Each CSV comes from DS4_TP_GATE_TRACE=<path> (ds4_rocm.cu). Per gate seq
the resident kernel stamps wall_clock64 at: TX released (ts_tx), RX warp
started spinning (ts_rxq), peer stamp first seen (ts_seen), RX copy done
(ts_rx). The two nodes' GPU clocks are unrelated, but the arrival window
X = ts_seen - ts_tx is measured on each rank for the SAME gate: with one-way
transport latency L and skew s = (peer TX release) - (my TX release),

    X_rank0 = L + s      X_rank1 = L - s      =>  L = (X0+X1)/2, s = (X0-X1)/2

so the join separates wire latency from rank load imbalance without any
cross-node clock sync. Host columns give the service thread's submit cost.
"""
import csv
import statistics
import sys


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if k == "ev_us" else int(v)) for k, v in r.items()}
            row.setdefault("ev_us", 0.0)
            rows[row["seq"]] = row
    return rows


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def describe(name, xs):
    if not xs:
        print(f"  {name:36s} (no samples)")
        return
    print(f"  {name:36s} n={len(xs):6d} mean={statistics.fmean(xs):8.1f} "
          f"p50={pct(xs, 50):8.1f} p90={pct(xs, 90):8.1f} "
          f"p99={pct(xs, 99):8.1f} max={max(xs):9.1f} us")


def valid(r):
    return all(r[k] not in (0, 0xFFFFFFFFFFFFFFFF)
               for k in ("ts_tx", "ts_rxq", "ts_seen", "ts_rx")) \
        and r["ts_seen"] >= r["ts_tx"] and r["ts_rx"] >= r["ts_seen"]


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    khz = 100000.0
    if "--khz" in argv:
        khz = float(argv[argv.index("--khz") + 1])
    us = 1000.0 / khz
    a, b = load(argv[1]), load(argv[2])

    for name, rows in (("rank0/leader", a), ("rank1/worker", b)):
        print(f"{name}: {len(rows)} gates from {argv[1] if rows is a else argv[2]}")
        v = [r for r in rows.values() if valid(r)]
        describe("window  tx_released->rx_done", [(r["ts_rx"] - r["ts_tx"]) * us for r in v])
        describe("arrival tx_released->peer seen", [(r["ts_seen"] - r["ts_tx"]) * us for r in v])
        describe("copy    seen->rx_done", [(r["ts_rx"] - r["ts_seen"]) * us for r in v])
        describe("rx spin start rel. tx_released", [(r["ts_rxq"] - r["ts_tx"]) * us for r in v])
        describe("legacy  event window (ev_us, decode rows)", [r["ev_us"] for r in rows.values() if r["ev_us"] != 0.0 and r["ts_tx"] == 0])
        h = [r for r in rows.values() if r["h_detect_ns"] and r["h_submit_ns"] >= r["h_detect_ns"]]
        describe("host    detect->submit returned", [(r["h_submit_ns"] - r["h_detect_ns"]) / 1000.0 for r in h])
        # Host detect lag: host clock vs GPU clock offset is unknown, but
        # within a 1000-gate window drift is negligible, so the minimum of
        # (h_detect - ts_tx) per window is the offset (+ the fastest wake).
        lags = []
        keys = sorted(r["seq"] for r in v if r["h_detect_ns"])
        for i in range(0, len(keys), 1000):
            chunk = [rows[k] for k in keys[i:i + 1000]]
            d = [r["h_detect_ns"] / 1000.0 - r["ts_tx"] * us for r in chunk]
            m = min(d)
            lags.extend(x - m for x in d)
        describe("host    tx_released->detect (rel. min)", lags)
        print()

    common = sorted(set(a) & set(b))
    common = [s for s in common if valid(a[s]) and valid(b[s])]
    print(f"joined gates: {len(common)}")
    if not common:
        return 1
    L = []
    S = []
    per_gate = {}
    for s in common:
        x0 = (a[s]["ts_seen"] - a[s]["ts_tx"]) * us
        x1 = (b[s]["ts_seen"] - b[s]["ts_tx"]) * us
        L.append((x0 + x1) / 2.0)
        S.append((x0 - x1) / 2.0)
        g = a[s]["gate"]
        per_gate.setdefault(g, ([], []))
        per_gate[g][0].append((x0 + x1) / 2.0)
        per_gate[g][1].append(abs(x0 - x1) / 2.0)
    describe("one-way latency L=(X0+X1)/2", L)
    describe("skew |s|=|X0-X1|/2", [abs(x) for x in S])
    describe("skew signed s (+: rank1 released later)", S)
    print("  per gate kind (gate field):")
    for g in sorted(per_gate):
        l, sk = per_gate[g]
        print(f"    gate={g}: n={len(l)} L mean={statistics.fmean(l):7.1f} "
              f"|skew| mean={statistics.fmean(sk):7.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
