#!/usr/bin/env python3
"""Serial CLI arithmetic comparisons; requires binaries built separately.

Resident runs use the reduced, repeated-weight fixture, not a quality model.
SSD runs use the full Q4 GGUF. Source/binary SHA-256 values and model identity
must remain invariant. Each binary runs from its own source directory so its
runtime Metal sources match its build. No build or fixture generation occurs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


LANE = Path(__file__).resolve().parent
REPO = LANE.parents[1]
ARCHIVE = LANE.parent / "qwen-drift-2026-10-03"
SERIES = ("resident-logits", "resident-greedy", "ssd-logits")
SOURCE_SUFFIXES = {".c", ".h", ".m", ".cu", ".cuh", ".inc", ".metal"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "device": stat.st_dev,
            "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns}


def sources(directory):
    paths = [p for p in directory.iterdir()
             if p.is_file() and (p.suffix in SOURCE_SUFFIXES or p.name == "Makefile")]
    for subtree in ("metal", "third_party"):
        paths.extend(p for p in (directory / subtree).rglob("*")
                     if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(set(paths))}


def frozen_arm(directory):
    binary = directory / "ds4"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"Build the executable before running: {binary}")
    for required in ("ds4.c", "ds4_cli.c", "ds4_metal.m", "metal/qwen4.metal"):
        if not (directory / required).is_file():
            raise RuntimeError(f"Missing runtime source: {directory / required}")
    return {"cwd": str(directory), "binary": str(binary),
            "binary_sha256": sha256(binary), "source_sha256": sources(directory)}


def check_frozen(arms, models, fixtures):
    for label, arm in arms.items():
        directory = Path(arm["cwd"])
        if sha256(Path(arm["binary"])) != arm["binary_sha256"]:
            raise RuntimeError(f"Binary changed during the comparison: {label}")
        if sources(directory) != arm["source_sha256"]:
            raise RuntimeError(f"Sources changed during the comparison: {label}")
    for model in models.values():
        if identity(Path(model["path"])) != model:
            raise RuntimeError(f"Model identity changed: {model['path']}")
    for fixture in fixtures.values():
        if sha256(Path(fixture["path"])) != fixture["sha256"]:
            raise RuntimeError(f"Prompt fixture changed: {fixture['path']}")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_dump(data, kind, expected_tokens, vocab):
    if data.get("prompt_tokens") != expected_tokens or data.get("ctx") != 16384:
        raise RuntimeError(f"Unexpected prompt/context: {data.get('prompt_tokens')}/{data.get('ctx')}")
    if kind == "logits":
        values = data.get("logits", [])
        if data.get("vocab") != vocab or len(values) != vocab or not all(map(finite, values)):
            raise RuntimeError("Incomplete or nonfinite full-vocabulary logits")
        argmax = data.get("argmax_token", {}).get("id")
        if not finite(data.get("argmax_logit")) or argmax != max(range(vocab), key=values.__getitem__):
            raise RuntimeError("Logits argmax metadata does not match the full vocabulary")
        return b"".join(struct.pack("<f", value) for value in values)
    steps = data.get("steps", [])
    if len(steps) != 64:
        raise RuntimeError(f"Greedy fixture produced {len(steps)} steps; this series requires 64")
    for index, step in enumerate(steps):
        selected = step.get("selected", {})
        if step.get("step") != index or not isinstance(selected.get("id"), int) or not 0 <= selected["id"] < vocab:
            raise RuntimeError("Invalid greedy step/token metadata")
        token_bytes = selected.get("bytes")
        if not isinstance(token_bytes, list) or not all(isinstance(x, int) and 0 <= x <= 255 for x in token_bytes):
            raise RuntimeError("Greedy selected token lacks valid raw bytes")
        scores = step.get("top_logprobs", [])
        if not scores or not all(finite(score.get("logit")) and finite(score.get("logprob")) for score in scores):
            raise RuntimeError("Missing or nonfinite greedy logprobs")
    return None


def compare(series, name, reference, candidate):
    a, b = reference["data"], candidate["data"]
    result = {"series": series, "name": name,
              "reference": reference["row"]["arm"], "candidate": candidate["row"]["arm"],
              "json_identical": reference["raw"] == candidate["raw"],
              "reference_wall_seconds": reference["row"]["wall_seconds"],
              "candidate_wall_seconds": candidate["row"]["wall_seconds"],
              "wall_speedup_percent": 100 * (reference["row"]["wall_seconds"] / candidate["row"]["wall_seconds"] - 1)}
    if series.endswith("logits"):
        bits_a, bits_b = reference["bits"], candidate["bits"]
        delta = [abs(x - y) for x, y in zip(a["logits"], b["logits"])]
        different = [i for i in range(len(delta)) if bits_a[4*i:4*i+4] != bits_b[4*i:4*i+4]]
        result.update(different_float32=len(different), same_argmax=a["argmax_token"] == b["argmax_token"],
                      max_abs=max(delta), max_abs_token=delta.index(max(delta)),
                      mean_abs=sum(delta) / len(delta), first_different_float32=different[0] if different else None)
        result["exact"] = not different and result["json_identical"]
    else:
        ids_a = [step["selected"]["id"] for step in a["steps"]]
        ids_b = [step["selected"]["id"] for step in b["steps"]]
        first = next((i for i, pair in enumerate(zip(ids_a, ids_b)) if pair[0] != pair[1]), None)
        result.update(same_tokens=ids_a == ids_b, common_token_prefix=64 if first is None else first,
                      first_different_step_zero_based=first,
                      selected_bytes_identical=[s["selected"]["bytes"] for s in a["steps"]] ==
                                               [s["selected"]["bytes"] for s in b["steps"]])
        if first is not None:
            result["selected_reference"] = a["steps"][first]["selected"]
            result["selected_candidate"] = b["steps"][first]["selected"]
        result["exact"] = result["same_tokens"] and result["selected_bytes_identical"] and result["json_identical"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", choices=SERIES, default=list(SERIES))
    parser.add_argument("--base-dir", type=Path, default=LANE / "base0aa")
    parser.add_argument("--before-dir", type=Path, default=LANE / "before-fix")
    parser.add_argument("--fixed-dir", type=Path, default=REPO)
    parser.add_argument("--resident-model", type=Path, default=LANE / "resident-fixture.gguf")
    parser.add_argument("--ssd-model", type=Path, default=REPO / "gguf/Qwen3.8-Flash-Next-Q4.gguf")
    parser.add_argument("--fixtures", type=Path, default=ARCHIVE / "fixtures")
    parser.add_argument("--output", type=Path, default=LANE / "fix-ab")
    parser.add_argument("--vocab", type=int, default=248320)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTING", "series": list(dict.fromkeys(args.series)), "runs": [], "comparisons": [],
              "scope": "Resident repeated-real-weight fixture: base main 0aaea5a vs root fix. Full Q4 SSD: before-fix vs root fix. Arithmetic regression, not model quality or exact M4 Q2 replication.",
              "timing_scope": "Serial process wall time includes model initialization, prefill, dump I/O, and greedy decode where selected; it is not isolated kernel throughput.",
              "model_identity_scope": "GGUF files are guarded by device/inode/size/mtime/ctime; whole large-model SHA-256 is not computed.",
              "started_unix": time.time(), "runner_sha256": sha256(Path(__file__).resolve()),
              "environment": {"DS4_QWEN4_PREFILL_CHUNK": "2048"}}

    def save():
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "results.json")

    save()
    try:
        directory_args = {"rootfix": args.fixed_dir.resolve()}
        if any(s.startswith("resident-") for s in args.series): directory_args["base0aa"] = args.base_dir.resolve()
        if "ssd-logits" in args.series: directory_args["before-fix"] = args.before_dir.resolve()
        arms = {label: frozen_arm(directory) for label, directory in directory_args.items()}
        report["arms"] = arms
        manifest_path = LANE / "source-snapshots-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            report["snapshot_manifest_sha256"] = sha256(manifest_path)
            report["base_commit"] = manifest.get("base_commit")
            report["working_tree_head_before_fix"] = manifest.get("working_tree_head")
            for label in ("base0aa", "before-fix"):
                if label not in arms: continue
                snapshot = manifest.get("snapshots", {}).get(label, {})
                if Path(snapshot.get("root", "")).resolve() != directory_args[label]: continue
                for name, record in snapshot.get("records", {}).items():
                    if sha256(directory_args[label] / name) != record["sha256"]:
                        raise RuntimeError(f"Snapshot differs from its recorded origin: {label}/{name}")
        models = {}
        if "base0aa" in arms: models["resident"] = identity(args.resident_model.resolve())
        if "before-fix" in arms: models["ssd"] = identity(args.ssd_model.resolve())
        report["models"] = models
        names = {"rome29": 29, "book575": 575, "book5942": 5942}
        needed_names = list(names) if "resident-logits" in args.series else ["rome29", "book575"]
        fixtures = {name: {"path": str((args.fixtures / f"{name}.txt").resolve()),
                           "sha256": sha256(args.fixtures / f"{name}.txt"), "expected_tokens": names[name]}
                    for name in needed_names}
        fixture_manifest = ARCHIVE / "fixtures.json"
        if args.fixtures.resolve() == (ARCHIVE / "fixtures").resolve() and fixture_manifest.is_file():
            expected = {row["name"]: row["sha256"] for row in json.loads(fixture_manifest.read_text())}
            if any(row["sha256"] != expected[name] for name, row in fixtures.items()):
                raise RuntimeError("Archived prompt SHA-256 differs from the fixture manifest")
        report["fixtures"] = fixtures
        environment = {key: value for key, value in os.environ.items() if not key.startswith("DS4_")}
        environment["DS4_QWEN4_PREFILL_CHUNK"] = "2048"
        report["status"] = "RUNNING"
        save()
        for series in report["series"]:
            streaming = series == "ssd-logits"
            kind = "greedy" if series.endswith("greedy") else "logits"
            model = models["ssd" if streaming else "resident"]["path"]
            reference_label = "before-fix" if streaming else "base0aa"
            series_names = needed_names if series == "resident-logits" else ["rome29", "book575"]
            for name in series_names:
                reference = None
                for label in (reference_label, "rootfix"):
                    check_frozen(arms, models, fixtures)
                    stem = f"{series}-{name}-{label}"
                    target, log_path = output / f"{stem}.json", output / f"{stem}.log"
                    if target.exists() or log_path.exists(): raise RuntimeError(f"Refusing to overwrite run output: {stem}")
                    command = [arms[label]["binary"], "-m", model, "--metal", "--ctx", "16384",
                               "--prefill-chunk", "2048", "--nothink", "--temp", "0",
                               "--prompt-file", fixtures[name]["path"]]
                    if streaming: command.append("--ssd-streaming")
                    command += ["-n", "64", "--dump-logprobs", str(target)] if kind == "greedy" else ["--dump-logits", str(target)]
                    print("START", series, name, label, flush=True)
                    started = time.monotonic()
                    with log_path.open("xb") as log:
                        process = subprocess.run(command, cwd=arms[label]["cwd"], env=environment,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    row = {"series": series, "name": name, "arm": label, "command": command,
                           "cwd": arms[label]["cwd"], "env": report["environment"],
                           "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
                           "log": str(log_path), "dump": str(target)}
                    report["runs"].append(row)
                    save()
                    check_frozen(arms, models, fixtures)
                    if process.returncode or not target.is_file():
                        raise RuntimeError(f"CLI failed: {stem}, return code {process.returncode}; see {log_path}")
                    raw = target.read_bytes()
                    data = json.loads(raw)
                    bits = validate_dump(data, kind, names[name], args.vocab)
                    row.update(prompt_tokens=data["prompt_tokens"], json_sha256=hashlib.sha256(raw).hexdigest(),
                               log_sha256=sha256(log_path))
                    if kind == "logits":
                        row.update(argmax=data["argmax_token"]["id"], float32_sha256=hashlib.sha256(bits).hexdigest())
                    else:
                        row.update(steps=len(data["steps"]), selected_ids=[s["selected"]["id"] for s in data["steps"]])
                    current = {"row": row, "data": data, "raw": raw, "bits": bits}
                    if reference is None: reference = current
                    else:
                        comparison = compare(series, name, reference, current)
                        report["comparisons"].append(comparison)
                        print("COMPARE", json.dumps(comparison), flush=True)
                    save()
                    print("DONE", series, name, label, round(row["wall_seconds"], 3), flush=True)
        check_frozen(arms, models, fixtures)
        report["status"] = "PASS" if all(c["exact"] for c in report["comparisons"]) else "FAIL_RESIDUAL_DIFFERENCE"
        report["finished_unix"] = time.time()
        save()
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        report["status"] = "FAIL_VALIDATION_OR_EXECUTION"
        report["error"] = f"{type(error).__name__}: {error}"
        report["finished_unix"] = time.time()
        save()
        print(report["error"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
