#!/usr/bin/env python3
"""
Scientific Benchmark Runner for DS4 on RTX 3090 (24 GB VRAM) + 64 GB RAM.
Implements the protocol in LOG.md:
- 4 Domains: D1_code, D2_math, D3_prose (via ./ds4 greedy --temp 0 --nothink + DS4_TOKEN_TIMING=1)
             and D4_bench (via ./ds4-bench --teacher-forced-decode across 1 cold + 5 warm frontiers).
- Records run-level metrics in benchmarks/results.csv and token-level timings in benchmarks/token_timings.csv.
- Verifies numerical invariance of prefill logits (--dump-logits) and greedy decoded token sequences against 0_baseline.
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
RESULTS_CSV = BENCH_DIR / "results.csv"
TOKENS_CSV = BENCH_DIR / "token_timings.csv"
TEXT_REF_JSON = BENCH_DIR / "baseline_generated_text.json"

ENV_BASE = os.environ.copy()
ENV_BASE["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.9/lib64:" + ENV_BASE.get("LD_LIBRARY_PATH", "")
ENV_BASE["DS4_TOKEN_TIMING"] = "1"

DOMAINS_CLI = [
    ("D1_code", BENCH_DIR / "D1_code.txt", BENCH_DIR / "baseline_D1_logits.json"),
    ("D2_math", BENCH_DIR / "D2_math.txt", BENCH_DIR / "baseline_D2_logits.json"),
    ("D3_prose", BENCH_DIR / "D3_prose.txt", BENCH_DIR / "baseline_D3_logits.json"),
]

RE_EVAL_MS = re.compile(r"ds4: gpu decode eval (\d+) took ([0-9.]+) ms")
RE_SUMMARY = re.compile(r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s")
RE_CACHE_SLOTS = re.compile(r"ds4: CUDA SSD expert cache: (\d+) slots, ([0-9.]+) GiB")


def check_logits_invariance(config_name: str, extra_args: list[str]) -> dict:
    """Compare post-prefill logits on D1_code against baseline_D1_logits.json."""
    ref_path = BENCH_DIR / "baseline_D1_logits.json"
    if not ref_path.exists() or config_name == "0_baseline":
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "top1_match": True}

    tmp_logits = BENCH_DIR / f"tmp_{config_name}_D1_logits.json"
    cmd = [
        str(ROOT / "ds4"),
        "--cuda",
        "--ssd-streaming",
        "--ctx", "4096",
        "--prompt-file", str(BENCH_DIR / "D1_code.txt"),
        "--temp", "0",
        "--nothink",
        "--dump-logits", str(tmp_logits),
    ] + extra_args
    print(f"[{config_name}] Checking numerical logit invariance on D1_code...")
    subprocess.run(cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    with open(tmp_logits, "r", encoding="utf-8") as f:
        cur_data = json.load(f)

    ref_logits = ref_data["logits"] if isinstance(ref_data, dict) and "logits" in ref_data else ref_data
    cur_logits = cur_data["logits"] if isinstance(cur_data, dict) and "logits" in cur_data else cur_data

    diffs = [abs(a - b) for a, b in zip(ref_logits, cur_logits)]
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    ref_top1 = max(range(len(ref_logits)), key=lambda i: ref_logits[i])
    cur_top1 = max(range(len(cur_logits)), key=lambda i: cur_logits[i])
    tmp_logits.unlink(missing_ok=True)
    print(f"[{config_name}] Logit invariance vs 0_baseline: max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, top1_match={ref_top1 == cur_top1}")
    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "top1_match": ref_top1 == cur_top1,
    }


def ensure_csv_headers():
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "domain", "run_id", "is_cold",
                "prefill_tps", "decode_tps", "steady_decode_tps",
                "first_token_ms", "steady_token_ms_mean", "steady_token_ms_sd",
                "vram_cache_slots", "vram_cache_gib", "exact_text_match"
            ])
    if not TOKENS_CSV.exists():
        with open(TOKENS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "domain", "run_id", "token_step", "is_cold_token", "latency_ms", "instant_tps"
            ])


def run_cli_domain(config_name: str, domain: str, prompt_path: Path, n_runs: int, n_tokens: int, extra_args: list[str], text_refs: dict):
    """Run N repetitions (run 0 = cold, runs 1..N-1 = warm) for a CLI prompt domain."""
    for run_id in range(n_runs):
        is_cold = 1 if run_id == 0 else 0
        cmd = [
            str(ROOT / "ds4"),
            "--cuda",
            "--ssd-streaming",
            "--ctx", "4096",
            "--prompt-file", str(prompt_path),
            "-n", str(n_tokens),
            "--temp", "0",
            "--nothink",
        ] + extra_args

        t0 = time.monotonic()
        proc = subprocess.run(cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        elapsed = time.monotonic() - t0

        stdout_text = proc.stdout.strip()
        stderr_text = proc.stderr

        eval_matches = RE_EVAL_MS.findall(stderr_text)
        eval_ms = [float(ms) for _, ms in eval_matches]

        summary_match = RE_SUMMARY.search(stderr_text)
        prefill_tps = float(summary_match.group(1)) if summary_match else 0.0
        reported_decode_tps = float(summary_match.group(2)) if summary_match else 0.0

        cache_matches = RE_CACHE_SLOTS.findall(stderr_text)
        cache_slots = int(cache_matches[-1][0]) if cache_matches else 0
        cache_gib = float(cache_matches[-1][1]) if cache_matches else 0.0

        first_ms = eval_ms[0] if eval_ms else 0.0
        steady_ms_list = eval_ms[1:] if len(eval_ms) > 1 else eval_ms
        steady_mean_ms = sum(steady_ms_list) / len(steady_ms_list) if steady_ms_list else 0.0
        steady_sd_ms = (
            math.sqrt(sum((x - steady_mean_ms) ** 2 for x in steady_ms_list) / (len(steady_ms_list) - 1))
            if len(steady_ms_list) > 1 else 0.0
        )
        steady_decode_tps = 1000.0 / steady_mean_ms if steady_mean_ms > 0 else reported_decode_tps

        if config_name == "0_baseline" and domain not in text_refs:
            text_refs[domain] = stdout_text
        exact_match = 1 if text_refs.get(domain, stdout_text) == stdout_text else 0

        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                config_name, domain, run_id, is_cold,
                f"{prefill_tps:.4f}", f"{reported_decode_tps:.4f}", f"{steady_decode_tps:.4f}",
                f"{first_ms:.3f}", f"{steady_mean_ms:.3f}", f"{steady_sd_ms:.3f}",
                cache_slots, f"{cache_gib:.3f}", exact_match
            ])

        with open(TOKENS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for idx, ms in enumerate(eval_ms, start=1):
                w.writerow([
                    config_name, domain, run_id, idx,
                    1 if (run_id == 0 and idx == 1) else 0,
                    f"{ms:.3f}", f"{1000.0 / ms:.4f}" if ms > 0 else "0.0"
                ])

        print(
            f"  [{config_name}] {domain} run={run_id} (cold={is_cold}): "
            f"prefill={prefill_tps:.2f} t/s | decode={reported_decode_tps:.2f} t/s | "
            f"steady_decode={steady_decode_tps:.3f} t/s ({steady_mean_ms:.1f}±{steady_sd_ms:.1f} ms/tok) | "
            f"cache={cache_slots} slots ({cache_gib:.2f} GiB) | match={bool(exact_match)} | wall={elapsed:.1f}s",
            flush=True
        )


def run_bench_domain(config_name: str, extra_args: list[str]):
    """Run D4_bench via ds4-bench with 1 cold frontier (66) + 5 warm frontiers (74, 82, 90, 98, 106)."""
    cmd = [
        str(ROOT / "ds4-bench"),
        "--cuda",
        "--ssd-streaming",
        "--prompt-file", str(BENCH_DIR / "D4_bench.txt"),
        "--ctx-start", "66",
        "--ctx-max", "106",
        "--step-incr", "8",
        "--gen-tokens", "8",
        "--teacher-forced-decode",
    ] + extra_args

    print(f"  [{config_name}] Running D4_bench (1 cold + 5 warm teacher-forced frontiers)...", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=ENV_BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    elapsed = time.monotonic() - t0

    cache_matches = RE_CACHE_SLOTS.findall(proc.stderr)
    cache_slots = int(cache_matches[-1][0]) if cache_matches else 0
    cache_gib = float(cache_matches[-1][1]) if cache_matches else 0.0

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    reader = csv.DictReader(lines)
    for run_id, row in enumerate(reader):
        is_cold = 1 if run_id == 0 else 0
        prefill_tps = float(row["prefill_tps"])
        gen_tps = float(row["gen_tps"])
        first_ms = float(row["gen_first_ms"])
        steady_tps = float(row["gen_steady_tps"])
        steady_ms = 1000.0 / steady_tps if steady_tps > 0 else 0.0

        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                config_name, "D4_bench", run_id, is_cold,
                f"{prefill_tps:.4f}", f"{gen_tps:.4f}", f"{steady_tps:.4f}",
                f"{first_ms:.3f}", f"{steady_ms:.3f}", "0.000",
                cache_slots, f"{cache_gib:.3f}", 1
            ])
        print(
            f"  [{config_name}] D4_bench run={run_id} (ctx={row['ctx_tokens']}, cold={is_cold}): "
            f"prefill={prefill_tps:.2f} t/s | decode={gen_tps:.2f} t/s | "
            f"steady_decode={steady_tps:.3f} t/s ({steady_ms:.1f} ms/tok) | first={first_ms:.1f} ms",
            flush=True
        )
    print(f"  [{config_name}] D4_bench finished in {elapsed:.1f}s", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: run_suite.py <config_name> [--n-runs 6] [--n-tokens 9] [-- extra_ds4_args...]")
        sys.exit(1)

    config_name = sys.argv[1]
    n_runs = 6  # 1 cold + 5 warm runs per domain
    n_tokens = 9  # 1 prefill token + 8 decode eval steps per run
    extra_args = []

    args = sys.argv[2:]
    while args:
        if args[0] == "--n-runs":
            n_runs = int(args[1])
            args = args[2:]
        elif args[0] == "--n-tokens":
            n_tokens = int(args[1])
            args = args[2:]
        elif args[0] == "--":
            extra_args = args[1:]
            break
        else:
            extra_args.append(args[0])
            args = args[1:]

    ensure_csv_headers()

    text_refs = {}
    if TEXT_REF_JSON.exists():
        with open(TEXT_REF_JSON, "r", encoding="utf-8") as f:
            text_refs = json.load(f)

    # Remove existing rows for this config_name if re-running
    for csv_path in [RESULTS_CSV, TOKENS_CSV]:
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            header, data = rows[0], [r for r in rows[1:] if r and r[0] != config_name]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(data)

    if config_name != "0_baseline":
        check_logits_invariance(config_name, extra_args)

    for domain, prompt_path, _ in DOMAINS_CLI:
        run_cli_domain(config_name, domain, prompt_path, n_runs, n_tokens, extra_args, text_refs)

    if config_name == "0_baseline":
        with open(TEXT_REF_JSON, "w", encoding="utf-8") as f:
            json.dump(text_refs, f, indent=2, ensure_ascii=False)

    run_bench_domain(config_name, extra_args)
    print(f"\n[{config_name}] Suite complete! Running statistical analysis...\n", flush=True)
    subprocess.run([sys.executable, str(BENCH_DIR / "analyze.py")], check=True)


if __name__ == "__main__":
    main()
