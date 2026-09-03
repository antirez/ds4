#!/usr/bin/env python3
"""Qwen3.8 M5 Max MTP probe driver (session twenty-one).

Reproduces the M5 decode/MTP attribution and configuration measurements on
the standard Q4_0-routed pack: staggered-load waves of ds4-server
instances, greedy probes on three prompt families (deterministic counting
that engages MTP near-perfectly, factual enumeration that engages
partially, novel prose that the scheduler bypasses), the server's own
MTP timing/histogram lines parsed via the archived session-twenty module,
and an interleaved round-robin protocol that cancels this host's +-10
tok/s run-order drift.  GPU stage attribution reuses DS4_QWEN4_PROFILE +
DS4_METAL_GPU_STAGE_PROFILE on single-depth servers.

Usage (GPU must be otherwise idle; each phase re-loads the pack per
server, 60-120 s per load):

  python3 speed-bench/qwen4_m5_mtp_probe.py attrib   # plain + d2..d8 wall stages
  python3 speed-bench/qwen4_m5_mtp_probe.py gprof    # verify-pass kernel tables d4/d7
  python3 speed-bench/qwen4_m5_mtp_probe.py config   # interleaved depth/family matrix

The prompt constructions are part of the measurement contract: counting is
"Continue the sequence, one number per line" seeded with 1..300, factual
recites months then days, prose is a fixed promessi_sposi slice.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_mtp_sweep as S  # noqa: E402

OUT = "/tmp/ds4_m5_round"
S.OUT_DIR = OUT

COUNTING = ("Continue the sequence, one number per line, no commentary:\n"
            + "\n".join(str(i) for i in range(1, 301)) + "\n")
FACTUAL = ("List the months of the year in order, one per line, with no "
           "commentary, then continue with the days of each month in "
           "order, one per line:\nJanuary\nFebruary\nMarch\nApril\nMay\n"
           "June\nJuly\n")
with open(S.CORPUS, "rb") as f:
    f.seek(400_000)
    PROSE = f.read(2400).decode("utf-8", errors="replace")

FAMILIES = {"counting": COUNTING, "factual": FACTUAL, "prose": PROSE}

STAGE_LINE_RE = re.compile(
    r"  gpu-stage +([0-9.]+) ms +(\d+) cb (\S.*?) +[0-9.]+ ms/cb")


def parse_forward_tables(text):
    """(rows, cache_pos, {label: (ms, cb)}) for every profiled forward."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"ds4: Qwen profile rows=(\d+) cache_pos=(\d+) ", lines[i])
        if not m:
            i += 1
            continue
        rows, pos = int(m.group(1)), int(m.group(2))
        j = i + 1
        table = {}
        if j < len(lines) and lines[j].startswith(
                "ds4: Metal GPU stage profile:"):
            j += 1
            while j < len(lines) and lines[j].startswith("  gpu-stage"):
                sm = STAGE_LINE_RE.match(lines[j])
                if sm:
                    table[sm.group(3).strip()] = (float(sm.group(1)),
                                                  int(sm.group(2)))
                j += 1
        out.append((rows, pos, table))
        i = j
    return out


def wave(names_specs, families, tokens_by_family, repeats, profile_env=None):
    """Staggered-load wave: each server waits ready before the next maps
    the pack (five simultaneous cold mappings OOM a 128 GiB host), then
    families run sequentially per server with the GPU serial."""
    servers = []
    try:
        for name, depth in names_specs:
            extra = dict(profile_env or {})
            extra["DS4_LOCK_FILE"] = f"{OUT}/{name}.lock"
            srv = S.Server(name, 8700 + len(servers) * 2,
                           mtp_depth=depth, ctx=2048, extra_env=extra)
            srv.wait_ready()
            ev = srv.startup_evidence()
            engaged = ev.get("graph") and (depth is None or ev.get("sidecar"))
            print(f"  {name}: graph={ev.get('graph')} sidecar="
                  f"{ev.get('sidecar')} moe_q40={ev.get('moe_down_q40')}"
                  f"{'  !! ENGAGEMENT EVIDENCE MISSING' if not engaged else ''}")
            sys.stdout.flush()
            servers.append((name, depth, srv))
    except Exception:
        for name, depth, srv in servers:
            srv.stop()
        raise
    records = []
    gprof_forwards = {}
    try:
        for name, depth, srv in servers:
            for fam, prompt in families.items():
                for _ in range(repeats):
                    rec, resp = srv.request(prompt, tokens_by_family[fam])
                    rec["family"] = fam
                    records.append(rec)
                    mtp = rec.get("mtp", {})
                    print(f"    [{name}/{fam}] tps={rec.get('decode_tps')} "
                          f"cyc={mtp.get('mean_cycle_ms')} "
                          f"acc={mtp.get('mean_accepted')} "
                          f"byp={mtp.get('bypass_cycles')}")
                    sys.stdout.flush()
            if profile_env and "DS4_METAL_GPU_STAGE_PROFILE" in profile_env:
                with open(srv.log_path, "rb") as f:
                    text = f.read().decode("utf-8", errors="replace")
                gprof_forwards[name] = parse_forward_tables(text)
    finally:
        for name, depth, srv in servers:
            srv.stop()
    return records, gprof_forwards


def wave_roundrobin(names_specs, prompts_tokens, rounds):
    """Interleaved fairness protocol: all servers resident, requests
    issued round-robin so every config's k-th request sees the same
    machine state."""
    servers = []
    try:
        for spec in names_specs:
            name, depth = spec[0], spec[1]
            extra = dict(spec[2]) if len(spec) > 2 else {}
            extra["DS4_LOCK_FILE"] = f"{OUT}/{name}.lock"
            srv = S.Server(name, 8700 + len(servers) * 2,
                           mtp_depth=depth, ctx=2048, extra_env=extra)
            srv.wait_ready()
            servers.append((name, depth, srv))
    except Exception:
        for name, depth, srv in servers:
            srv.stop()
        raise
    records = []
    try:
        for r in range(rounds):
            for name, depth, srv in servers:
                for fam, (prompt, tokens) in prompts_tokens.items():
                    rec, resp = srv.request(prompt, tokens)
                    rec["family"] = fam
                    rec["round"] = r
                    records.append(rec)
                    print(f"    r{r} [{name}/{fam}] "
                          f"tps={rec.get('decode_tps')} "
                          f"acc={rec.get('mtp', {}).get('mean_accepted')}")
                    sys.stdout.flush()
    finally:
        for name, depth, srv in servers:
            srv.stop()
    return records


def summarize(records):
    print("\n=== summary (engaged cycles only) ===")
    for srv in sorted({r["server"] for r in records}):
        for fam in FAMILIES:
            frs = [r for r in records
                   if r["server"] == srv and r.get("family") == fam]
            if not frs:
                continue
            tps = [r["decode_tps"] for r in frs if "decode_tps" in r]
            eng = [r for r in frs
                   if "mtp" in r and r["mtp"]["block_cycles"] > 0]
            line = f"{srv:10s} {fam:9s} tps={'/'.join(f'{t:.1f}' for t in tps)}"
            if eng:
                m = [r["mtp"] for r in eng]
                line += (f" acc={sum(x['mean_accepted'] for x in m)/len(m):.2f}"
                         f" cyc={sum(x['mean_cycle_ms'] for x in m)/len(m):.1f}")
                sts = [r["stages"] for r in eng if "stages" in r]
                if sts:
                    n = len(sts)
                    line += (" ms/cyc: draft "
                             f"{sum(s['draft_ms'] for s in sts)/n:.1f}"
                             f" verify {sum(s['verify_ms'] for s in sts)/n:.1f}"
                             f" hist {sum(s['history_ms'] for s in sts)/n:.1f}"
                             f" commit {sum(s['commit_ms'] for s in sts)/n:.1f}")
            elif any("mtp" in r for r in frs):
                line += " NEVER-ENGAGED"
            print(line)
    sys.stdout.flush()


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "attrib"
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if phase == "attrib":
        records = []
        for specs in ([("plain", None), ("d2", 2), ("d3", 3),
                       ("d4", 4), ("d5", 5)],
                      [("d6", 6), ("d7", 7), ("d8", 8)]):
            recs, _ = wave(specs, FAMILIES,
                           {"counting": 192, "factual": 128, "prose": 128},
                           repeats=3)
            records.extend(recs)
        with open(f"{OUT}/attrib_records.json", "w") as f:
            json.dump(records, f, indent=1)
        summarize(records)
    elif phase == "gprof":
        _, forwards = wave(
            [("d4p", 4), ("d7p", 7)], {"counting": COUNTING},
            {"counting": 96}, repeats=1,
            profile_env={"DS4_QWEN4_PROFILE": "1",
                         "DS4_METAL_GPU_STAGE_PROFILE": "1"})
        for name, fwd in forwards.items():
            buckets = {}
            for rows, pos, table in fwd:
                if rows > 8:
                    continue  # prompt prefill chunks
                key = "rows=1" if rows == 1 else f"rows={rows}"
                agg = buckets.setdefault(key, {"n": 0, "ms": {}})
                agg["n"] += 1
                for label, (ms, cnt) in table.items():
                    agg["ms"][label] = agg["ms"].get(label, 0.0) + ms
            print(f"\n=== gprof {name} ===")
            for key, agg in sorted(buckets.items()):
                total = sum(agg["ms"].values())
                print(f"  {key}: {agg['n']} forwards, "
                      f"{total/agg['n']:.2f} ms/forward")
                for label, ms in sorted(agg["ms"].items(),
                                        key=lambda kv: -kv[1])[:18]:
                    print(f"    {ms/agg['n']*1000:8.1f} us/fwd  {label}")
            sys.stdout.flush()
    elif phase == "config":
        records = wave_roundrobin(
            [("ip", None), ("i4", 4), ("i5", 5), ("i6", 6), ("i7", 7),
             ("i8", 8)],
            {"counting": (COUNTING, 192), "factual": (FACTUAL, 128),
             "prose": (PROSE, 128)},
            rounds=3)
        with open(f"{OUT}/config_records.json", "w") as f:
            json.dump(records, f, indent=1)
        summarize(records)
    else:
        raise SystemExit(f"unknown phase {phase}")
    print(f"\n[{phase}] wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
