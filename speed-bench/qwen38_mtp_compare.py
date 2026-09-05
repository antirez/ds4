#!/usr/bin/env python3
"""Interleave greedy CLI MTP runs and retain timing and output parity evidence.

Save the baseline executable, metal/qwen4.metal, and metal/moe.metal before
editing: the executable loads Metal sources at runtime. Model loading and
prefill are excluded from generation rate. Warmups are excluded from medians.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "hamlet": ("Write a three-sentence summary of the plot of Hamlet.", 120),
    "fibonacci": ("List the first 30 Fibonacci numbers with indices.", 400),
    "explanation": (
        "Explain how a computer sends a web request over TCP and receives the "
        "response. Write four clear paragraphs for a programmer learning networking.",
        256,
    ),
}
RATE = re.compile(r"ds4: (?:Qwen3\.8 )?prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s")
ACCEPT = re.compile(r"ds4: Qwen3\.8 mtp: (\d+) verify cycles, (\d+) drafts accepted")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ple", type=Path)
    ap.add_argument("--candidate", type=Path, default=ROOT / "ds4")
    ap.add_argument("--candidate-source", type=Path, default=ROOT / "metal/qwen4.metal")
    ap.add_argument("--candidate-moe-source", type=Path, default=ROOT / "metal/moe.metal")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--baseline-source", type=Path, required=True)
    ap.add_argument("--baseline-moe-source", type=Path, required=True)
    ap.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--case", choices=CASES, action="append")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    overrides = {}
    for item in args.candidate_env:
        key, sep, value = item.partition("=")
        if not sep or not key.startswith("DS4_") or key.startswith("DS4_METAL_") and key.endswith("_SOURCE"):
            ap.error("--candidate-env needs DS4_NAME=VALUE")
        overrides[key] = value
    if "DS4_QWEN4_SPEC_FORCE_ACCEPT" in overrides:
        ap.error("forced draft acceptance is not a valid throughput benchmark")
    args.out.mkdir(parents=True, exist_ok=True)
    configs = {
        "baseline": (args.baseline.resolve(), args.baseline_source.resolve(), args.baseline_moe_source.resolve()),
        "candidate": (args.candidate.resolve(), args.candidate_source.resolve(), args.candidate_moe_source.resolve()),
    }
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "model": str(args.model.resolve()), "ctx": args.ctx,
        "ple": str(args.ple.resolve()) if args.ple else None,
        "configs": {name: {"binary": str(binary), "binary_sha256": sha256(binary),
                           "source": str(source), "source_sha256": sha256(source),
                           "moe_source": str(moe_source), "moe_source_sha256": sha256(moe_source)}
                    for name, (binary, source, moe_source) in configs.items()},
        "shared_metal_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "metal").glob("*"))
                                if p.is_file() and p.name not in ("qwen4.metal", "moe.metal")},
        "candidate_env": overrides, "records": [],
    }

    def run(name, case, repeat):
        binary, source, moe_source = configs[name]
        prompt, tokens = CASES[case]
        cmd = [str(binary), "-m", str(args.model.resolve()), "--metal", "--ctx", str(args.ctx),
               "-p", prompt, "-n", str(tokens), "--temp", "0", "--nothink", "--mtp", "--mtp-timing"]
        if args.ple:
            cmd += ["--ple", str(args.ple.resolve())]
        # Only explicit overrides participate; inherited diagnostic and tuning
        # flags must not silently alter either side of the comparison.
        env = {key: value for key, value in os.environ.items() if not key.startswith("DS4_")}
        env["DS4_METAL_QWEN4_SOURCE"] = str(source)
        env["DS4_METAL_MOE_SOURCE"] = str(moe_source)
        if name == "candidate":
            env.update(overrides)
        stem = args.out / f"{name}-{case}-r{repeat}"
        start = time.monotonic()
        with stem.with_suffix(".stdout").open("wb") as out, stem.with_suffix(".stderr").open("wb") as err:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err, check=True)
        log = stem.with_suffix(".stderr").read_text()
        rates, accepts = RATE.findall(log), ACCEPT.findall(log)
        if not rates or not accepts or int(accepts[-1][0]) == 0:
            raise RuntimeError(f"Missing generation or MTP verification evidence in {stem}")
        rec = {"config": name, "case": case, "repeat": repeat, "warmup": repeat < 0,
               "command": cmd, "prefill_tps": float(rates[-1][0]),
               "decode_tps": float(rates[-1][1]), "cycles": int(accepts[-1][0]),
               "accepted": int(accepts[-1][1]), "wall_s": time.monotonic() - start,
               "output_sha256": sha256(stem.with_suffix(".stdout"))}
        report["records"].append(rec)
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}/{case}/r{repeat}: {rec['decode_tps']:.2f} t/s, "
              f"accepted {rec['accepted']}/{rec['cycles']}", flush=True)

    cases = args.case or list(CASES)
    for name in configs:
        run(name, cases[0], -1)
    for repeat in range(args.repeats):
        for case in cases:
            for name in list(configs)[::1 if repeat % 2 == 0 else -1]:
                run(name, case, repeat)
    report["summary"] = {}
    for case in cases:
        records = [r for r in report["records"] if r["case"] == case and not r["warmup"]]
        medians = {name: statistics.median(r["decode_tps"] for r in records if r["config"] == name)
                   for name in configs}
        summary = {"median_tps": medians,
                   "range_tps": {name: [min(r["decode_tps"] for r in records if r["config"] == name),
                                         max(r["decode_tps"] for r in records if r["config"] == name)]
                                 for name in configs},
                   "speedup": medians["candidate"] / medians["baseline"],
                   "all_outputs_identical": len({r["output_sha256"] for r in records}) == 1,
                   "acceptance_identical": len({(r["accepted"], r["cycles"]) for r in records}) == 1}
        report["summary"][case] = summary
        print(f"{case}: {json.dumps(summary)}", flush=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(s["all_outputs_identical"] for s in report["summary"].values()):
        raise SystemExit("Output parity failed; inspect the saved stdout files before accepting the speedup")


if __name__ == "__main__":
    main()
