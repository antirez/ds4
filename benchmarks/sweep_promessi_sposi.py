#!/usr/bin/env python3
"""
sweep_promessi_sposi.py — Context sweep benchmark with live progress bar & ETA
Uses ds4-bench on speed-bench/promessi_sposi.txt across scalable frontiers (2K to 16K).
"""

import sys
import os
import time
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "speed-bench" / "promessi_sposi.txt"
CSV_OUT = ROOT / "benchmarks" / "promessi_sposi_sweep.csv"
LOGITS_DIR = ROOT / "benchmarks" / "promessi_sposi_logits"

FRONTIERS = [2048, 4096, 8192, 16384]
GEN_TOKENS = 128

def format_eta(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s:02d}s"

def progress_bar(current: int, total: int, width: int = 25) -> str:
    filled = int(width * current / total)
    bar = "=" * filled + (">" if filled < width else "")
    bar = bar.ljust(width, " ")
    percent = (current / total) * 100.0
    return f"[{bar}] {percent:5.1f}%"

def run_frontier(ctx: int, frontier_idx: int, total_frontiers: int, elapsed_so_far: float) -> dict:
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / "ds4-bench"),
        "--cuda",
        "--ssd-streaming",
        "-m", "ds4flash.gguf",
        "--prompt-file", str(PROMPT_FILE),
        "--ctx-start", str(ctx),
        "--ctx-max", str(ctx),
        "--step-incr", str(ctx),
        "--gen-tokens", str(GEN_TOKENS),
        "--dump-frontier-logits-dir", str(LOGITS_DIR),
    ]

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout_lines = []
    # Stream output
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            stdout_lines.append(line.strip())

    stderr_text = proc.stderr.read()
    rc = proc.wait()
    duration = time.monotonic() - t0

    if rc != 0:
        print(f"\n[ERROR] ds4-bench failed for ctx={ctx}:\n{stderr_text}", file=sys.stderr)
        sys.exit(rc)

    # Parse CSV line from ds4-bench stdout:
    # ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes
    res = {
        "ctx": ctx,
        "prefill_tokens": 0,
        "prefill_tps": 0.0,
        "gen_tokens": GEN_TOKENS,
        "gen_tps": 0.0,
        "first_ms": 0.0,
        "steady_tokens": 0,
        "steady_tps": 0.0,
        "duration": duration,
    }

    for line in stdout_lines:
        parts = line.split(",")
        if len(parts) >= 8 and parts[0].isdigit() and int(parts[0]) == ctx:
            res["prefill_tokens"] = int(parts[1])
            res["prefill_tps"] = float(parts[2])
            res["gen_tokens"] = int(parts[3])
            res["gen_tps"] = float(parts[4])
            res["first_ms"] = float(parts[5])
            res["steady_tokens"] = int(parts[6])
            res["steady_tps"] = float(parts[7])
            break

    return res

def main():
    print("=" * 78)
    print("  DwarfStar (DS4) — Benchmark di Contesto su 'I Promessi Sposi'")
    print(f"  Frontiere: {FRONTIERS} token | Generazione: {GEN_TOKENS} token/frontiera")
    print("=" * 78)

    total_frontiers = len(FRONTIERS)
    results = []
    start_time = time.monotonic()

    # CSV Header
    with open(CSV_OUT, "w", encoding="utf-8") as f:
        f.write("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,wall_sec\n")

    for idx, ctx in enumerate(FRONTIERS, start=1):
        elapsed = time.monotonic() - start_time
        # ETA calculation
        if idx > 1:
            avg_per_step = elapsed / (idx - 1)
            eta_sec = avg_per_step * (total_frontiers - idx + 1)
            eta_str = format_eta(eta_sec)
        else:
            eta_str = "Stima in corso..."

        pbar = progress_bar(idx - 1, total_frontiers)
        print(f"\n{pbar} | Frontiera {idx}/{total_frontiers} ({ctx} token) | ETA: {eta_str}")
        print(f"--> Esecuzione prefill {ctx} token + decode {GEN_TOKENS} token...")

        r = run_frontier(ctx, idx, total_frontiers, elapsed)
        results.append(r)

        with open(CSV_OUT, "a", encoding="utf-8") as f:
            f.write(f"{r['ctx']},{r['prefill_tokens']},{r['prefill_tps']:.2f},{r['gen_tokens']},{r['gen_tps']:.2f},{r['first_ms']:.2f},{r['steady_tokens']},{r['steady_tps']:.2f},{r['duration']:.2f}\n")

        print(f"    Completato: Prefill = {r['prefill_tps']:.2f} t/s | Steady Decode = {r['steady_tps']:.2f} t/s ({1000.0/r['steady_tps'] if r['steady_tps']>0 else 0:.1f} ms/tok) [in {r['duration']:.1f}s]")

    total_wall = time.monotonic() - start_time
    print("\n" + "=" * 78)
    print(f"  Benchmark Completato con Successo in {format_eta(total_wall)}!")
    print(f"  Dati salvati in: {CSV_OUT.relative_to(ROOT)}")
    print(f"  Logits frontiere salvati in: {LOGITS_DIR.relative_to(ROOT)}")
    print("=" * 78)

    # Print summary table
    print("\n| Contesto (tokens) | Prefill (t/s) | Steady Decode (t/s) | Latenza Token (ms) | Tempo Frontiera |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        lat_ms = 1000.0 / r["steady_tps"] if r["steady_tps"] > 0 else 0.0
        print(f"| {r['ctx']:,} | {r['prefill_tps']:.2f} t/s | **{r['steady_tps']:.2f} t/s** | {lat_ms:.1f} ms | {r['duration']:.1f}s |")

if __name__ == "__main__":
    main()
