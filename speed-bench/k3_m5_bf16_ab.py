#!/usr/bin/env python3
"""Compare the reference and TensorOps Kimi K3 BF16 prefill on a 128 GiB M5 Max."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_TOKENS = 45
GEN_TOKENS = 32
MARKER = "k3: M5 Max BF16 TensorOps prefill dispatched"


def run_bench(model, output, variant, dump):
    directory = output / (variant + ("-logits" if dump else "-speed"))
    directory.mkdir()
    args = [str(ROOT / "ds4-bench"), "-m", str(model), "--metal", "--ssd-streaming",
            "--simulate-used-memory", "64GB", "--ssd-streaming-cache-experts", "10GB",
            "--prefill-chunk", "64", "--ctx-start", str(PROMPT_TOKENS),
            "--ctx-max", str(PROMPT_TOKENS), "--gen-tokens", str(GEN_TOKENS),
            "--chat-prompt-file", str(output / "prompt.txt")]
    if dump:
        args += ["--dump-frontier-logits-dir", str(directory), "--dump-decode-logits"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("K3_", "DS4_"))}
    env.update(K3_M5_BF16_TENSOROPS="1" if variant == "tensorops" else "0",
               K3_GGUF_STATIC_CACHE="", K3_MEMORY_RESERVE_GIB="12", K3_IO_PROFILE="1")
    log = directory / "stderr.log"
    print(f"Running {directory.name} (log: {log})", flush=True)
    csv_path = directory / "bench.csv"
    with csv_path.open("w") as csv_file, log.open("w") as stderr:
        status = subprocess.run(args, cwd=ROOT, env=env, stdout=csv_file, stderr=stderr)
    if status.returncode:
        raise SystemExit(f"ds4-bench exited {status.returncode}; see {log}")
    lines = log.read_text().splitlines()
    markers = [s for s in lines if MARKER in s]
    if variant == "tensorops" and not any(re.search(r"\brows=45\b", s) for s in markers):
        raise SystemExit(f"Missing 45-row TensorOps dispatch in {log}")
    if variant == "reference" and markers:
        raise SystemExit(f"Unexpected TensorOps dispatch in {log}")
    with csv_path.open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1 or int(rows[0]["prefill_tokens"]) != PROMPT_TOKENS or int(rows[0]["gen_tokens"]) != GEN_TOKENS:
        raise SystemExit(f"Incomplete benchmark in {csv_path}")
    if not dump:
        seconds = [float(m[1]) for m in re.finditer(
            r"^k3: evaluation timing .* total=([0-9.]+) s$", "\n".join(lines), re.MULTILINE)]
        if len(seconds) != 1 + GEN_TOKENS or min(seconds) <= 0:
            raise SystemExit(f"Incomplete evaluation timings in {log}")
        decode = sum(seconds[1:]) / GEN_TOKENS
        print(f"  ds4-bench: prefill {rows[0]['prefill_tps']} tok/s, generation {rows[0]['gen_tps']} tok/s")
        print(f"  Evaluation: prefill {seconds[0]:.3f} s, generation {decode:.3f} s/token ({1 / decode:.5f} tok/s)")
        return seconds[0], decode


def read_logits(directory, step):
    suffix = f"_decode_{step:06d}" if step else ""
    path = directory / f"frontier_{PROMPT_TOKENS:06d}{suffix}.logits.json"
    # The writer uses %.9g for binary32, which may render negative zero as -0.
    # parse_int=float preserves that sign when reconstructing the logit bits.
    data = json.loads(path.read_bytes(), parse_int=float)
    if data["frontier_tokens"] != PROMPT_TOKENS or data.get("decode_step", 0) != step:
        raise ValueError(f"Unexpected frontier or decode step in {path}")
    if step and data["session_position"] != PROMPT_TOKENS + step:
        raise ValueError(f"Unexpected session position in {path}")
    values = data["logits"]
    if len(values) != data["vocab"] or not values or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError(f"Incomplete or nonfinite logits in {path}")
    token = int(data["input_token_id"]) if step else None
    return token, int(data["argmax_id"]), array("f", values)


def compare_logits(output):
    first_token_difference = None
    identical = compared = 0
    worst_abs = 0.0
    with (output / "logits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "reference_input", "tensorops_input", "reference_argmax",
                         "tensorops_argmax", "same_history", "identical_fp32", "max_abs", "rms"])
        for step in range(GEN_TOKENS + 1):
            token_a, best_a, a = read_logits(output / "reference-logits", step)
            token_b, best_b, b = read_logits(output / "tensorops-logits", step)
            if len(a) != len(b):
                raise ValueError(f"Vocabulary sizes differ at step {step}")
            if token_a != token_b and first_token_difference is None:
                first_token_difference = step
            if first_token_difference is None:
                exact = a.tobytes() == b.tobytes()
                max_abs = max(abs(x - y) for x, y in zip(a, b))
                rms = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
                worst_abs = max(worst_abs, max_abs)
                identical += exact
                compared += 1
                writer.writerow([step, token_a, token_b, best_a, best_b, True, exact, max_abs, rms])
                if step == 0:
                    print(f"Prefill logits: max abs {max_abs:.8g}, RMS {rms:.8g}, argmax {best_a} / {best_b}")
            else:
                writer.writerow([step, token_a, token_b, best_a, best_b, False, "", "", ""])
    print(f"Same-history logits: {identical}/{compared} vectors bit-identical, worst abs {worst_abs:.8g}")
    print(f"First differing generated token (1-based): {first_token_difference or 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the original Kimi K3 mixed-Q2 GGUF")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if chip != "Apple M5 Max" or ram <= 64 * 2**30:
        parser.error(f"This 64 GiB pressure benchmark requires an M5 Max with more than 64 GiB (found {chip}, {ram / 2**30:g} GiB)")
    output = Path(tempfile.mkdtemp(prefix="k3-m5-bf16-", dir=str(Path.cwd())))
    print(f"Results: {output}", flush=True)
    (output / "prompt.txt").write_text("Ciao, come stai?")
    # Warm both paths while collecting logits, then time them in reverse order.
    for variant in ("reference", "tensorops"):
        run_bench(model, output, variant, dump=True)
    candidate_prefill, candidate_decode = run_bench(model, output, "tensorops", dump=False)
    reference_prefill, reference_decode = run_bench(model, output, "reference", dump=False)
    print(f"TensorOps evaluation time change vs reference: prefill "
          f"{(candidate_prefill / reference_prefill - 1):+.1%}, "
          f"generation {(candidate_decode / reference_decode - 1):+.1%}")
    compare_logits(output)


if __name__ == "__main__":
    main()
