#!/usr/bin/env python3
"""DSpark DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT benchmark.

Companion to tests/dspark_acceptance_fixture.sh: that fixture asserts exact
byte-for-byte output equality on short (32-token) generations, which the
fast-commit flag preserves in every case tested so far. This script instead
runs longer (~300+ token) free-running generations, where a divergence that
starts as one flipped near-tied logit has room to compound into a visibly
different continuation, and reports where (if anywhere) that happens.

Usage:
  DS4_DSPARK_MODEL=./ds4flash.gguf \
  DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  python3 tests/dspark_fast_commit_bench.py
"""
import os
import re
import subprocess
import time

DS4 = os.environ.get("DS4_BIN", "./ds4")
MODEL = os.environ.get("DS4_DSPARK_MODEL", "./ds4flash.gguf")
DSPARK_GGUF = os.environ.get(
    "DS4_DSPARK_SUPPORT", "gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf")

PROMPTS = {
    "essay": "Write a 300 word essay about the history of Rome.",
    "code": "Write a Python function that implements binary search on a "
            "sorted list, with a docstring and type hints.",
    "math": "Solve step by step: A train travels 120 miles in 2 hours, "
            "then 180 miles in 3 hours. What is its average speed for the "
            "whole trip?",
    "dialogue": "Write a short conversation between two characters "
                "debating whether cats or dogs make better pets.",
}


def run(args, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True, env=env,
                        timeout=180)
    return p.stdout + p.stderr, time.time() - t0


def extract_text(output):
    return "\n".join(l for l in output.splitlines()
                      if not l.startswith("ds4:")).strip()


def extract_speed(output):
    m = re.search(r"generation: ([\d.]+) t/s", output)
    return float(m.group(1)) if m else None


def extract_stats(output):
    m = re.search(r"ds4: DSpark stats (.+)", output)
    return m.group(1) if m else None


def word_divergence(baseline, candidate):
    bw, cw = baseline.split(), candidate.split()
    for i in range(min(len(bw), len(cw))):
        if bw[i] != cw[i]:
            return i, bw[i], cw[i]
    if len(bw) != len(cw):
        return min(len(bw), len(cw)), "<end>", "<end>"
    return None, None, None


def main():
    if not os.path.isfile(MODEL):
        print(f"skipped: missing model {MODEL}")
        return
    if not os.path.isfile(DSPARK_GGUF):
        print(f"skipped: missing DSpark support model {DSPARK_GGUF}")
        return

    results = {}
    for name, prompt in PROMPTS.items():
        print(f"=== {name} ===", flush=True)
        common = [DS4, "--model", MODEL, "--temp", "0", "--metal",
                  "--ctx", "8192", "--nothink", "-p", prompt]

        out, wall = run(common)
        baseline_text = extract_text(out)
        baseline_speed = extract_speed(out)
        print(f"  baseline:        {baseline_speed} t/s ({wall:.1f}s wall)")

        dspark_common = [DS4, "--model", MODEL, "--mtp", DSPARK_GGUF,
                          "--dspark", "--mtp-draft", "2", "--temp", "0",
                          "--metal", "--ctx", "8192", "--nothink",
                          "-p", prompt]

        out, wall = run(dspark_common, {"DS4_DSPARK_STATS": "1"})
        off_text = extract_text(out)
        off_speed = extract_speed(out)
        off_stats = extract_stats(out)
        off_div = word_divergence(baseline_text, off_text)
        print(f"  dspark flag-off: {off_speed} t/s ({wall:.1f}s wall) "
              f"diverges_at_word={off_div[0]}")

        out, wall = run(dspark_common, {
            "DS4_DSPARK_STATS": "1",
            "DS4_DSPARK_FULL_ACCEPT_FAST_COMMIT": "1",
        })
        on_text = extract_text(out)
        on_speed = extract_speed(out)
        on_stats = extract_stats(out)
        on_div = word_divergence(baseline_text, on_text)
        print(f"  dspark flag-on:  {on_speed} t/s ({wall:.1f}s wall) "
              f"diverges_at_word={on_div[0]}")

        results[name] = dict(
            baseline_speed=baseline_speed,
            off_speed=off_speed, off_stats=off_stats, off_div=off_div,
            on_speed=on_speed, on_stats=on_stats, on_div=on_div,
        )

    print("\n\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(f"baseline:  {r['baseline_speed']} t/s")
        print(f"flag-off:  {r['off_speed']} t/s  "
              f"diverges_at_word={r['off_div'][0]}")
        print(f"flag-on:   {r['on_speed']} t/s  "
              f"diverges_at_word={r['on_div'][0]}")
        if r["off_stats"]:
            print(f"flag-off stats: {r['off_stats']}")
        if r["on_stats"]:
            print(f"flag-on  stats: {r['on_stats']}")


if __name__ == "__main__":
    main()
