#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Run the native DS4 Qwen3.8 acceptance benchmark.

The runner owns process lifetime, cache-distinct raw completions, warmup
discard, log-derived compute timings, RSS sampling, and an atomically resumable
result document.  The independent qwen4_acceptance.py checker owns the
pass/fail policy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import plistlib
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


CORE_CASES = (
    ("prefill-10k+1", "prefill", 1),
    ("prefill-50k+1", "prefill", 1),
    ("prefill-100k+1", "prefill", 1),
    ("decode-code-500", "decode", 500),
    ("decode-prose-500", "decode", 500),
    ("decode-8.5k-500", "decode", 500),
    ("end-to-end-10k+500", "end_to_end", 500),
)
COLD_PLE_CASE = ("cold-ple-10k+1", "diagnostic", 1)
MTP_CASES = (("mtp-10k+500", "end_to_end", 500),)
MTP_DRAFT_TOKENS = 4
MTP_TIMING_SCHEMA = "ds4.qwen4.mtp-timing"

PLE_COLD_SCHEMA = "ds4.qwen4.ple-cold-evidence"
PLE_COLD_LOG_PREFIX = "ds4: Qwen PLE cold evidence "
COLD_PHASE_MARKER = "cold_ple_ds4_only_first_request"
COLD_PROMPT_TOKENS = 10_000
COLD_COMPLETION_TOKENS = 1
COLD_SELECTED_CHUNK = 8192
COLD_LOOKUP_ROWS = 160_000
COLD_ROW_BYTES = 100
COLD_LOGICAL_BYTES = 16_000_000
PLE_BINDING_SCHEMA = "ds4.qwen4.ple-binding"
SSD_BACKING_SCHEMA = "ds4.qwen4.ssd-backing"
PACK_MANIFEST_NAME = "qwen3.8-flash-next-q4.manifest.json"

DS4_PREFILL_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) "
    r"prompt done ([0-9.]+)s"
)
DS4_DECODE_RE = re.compile(
    r"ds4-server: completion ctx=.*? gen=(\d+).*? "
    r"decoding chunk=[0-9.]+ t/s avg=([0-9.]+) t/s ([0-9.]+)s"
)
DS4_CHUNK_RE = re.compile(r"Qwen prefill selected chunk=(\d+)")
DS4_MTP_TIMING_PREFIX = "ds4: Qwen MTP timing "
DS4_MTP_TIMING_RE = re.compile(
    r"ds4: Qwen MTP timing drafted=(\d+) accepted=(\d+) "
    r"target_tokens=(\d+) cycle=([0-9]+(?:\.[0-9]+)?) ms"
    r"(?: verifier=([A-Za-z0-9_-]+))?"
)
class RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def proc_pid_rusage_v2(pid: int) -> dict | None:
    """Read Darwin's process-level cumulative I/O counters."""
    if sys.platform != "darwin":
        return None
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = libproc.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    info = RusageInfoV2()
    if function(pid, 2, ctypes.byref(info)) != 0:
        return None
    return {
        "proc_start_abstime": int(info.ri_proc_start_abstime),
        "pageins": int(info.ri_pageins),
        "diskio_bytesread": int(info.ri_diskio_bytesread),
        "diskio_byteswritten": int(info.ri_diskio_byteswritten),
        "resident_size": int(info.ri_resident_size),
        "phys_footprint": int(info.ri_phys_footprint),
    }


def rusage_v2_evidence(before: dict, after: dict) -> dict:
    before_start = int(before["proc_start_abstime"])
    after_start = int(after["proc_start_abstime"])
    if before_start <= 0 or after_start != before_start:
        raise ValueError(
            "proc_pid_rusage V2 snapshots are from different process lifetimes"
        )
    cumulative = ("pageins", "diskio_bytesread", "diskio_byteswritten")
    delta = {}
    for name in cumulative:
        left = int(before[name])
        right = int(after[name])
        if right < left:
            raise ValueError(f"proc_pid_rusage V2 counter regressed: {name}")
        delta[name] = right - left
    return {
        "flavor": 2,
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
    }


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def url_json(url: str, payload: dict | None = None,
             timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with status {process.returncode} before ready"
            )
        try:
            return url_json(url, timeout=2.0)
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready: {last_error}")


@dataclass
class ServerProcess:
    name: str
    command: list[str]
    port: int
    log_path: Path
    startup_timeout: float
    environment: dict[str, str] | None = None
    process: subprocess.Popen | None = None
    model_id: str | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=self.environment,
            )
        finally:
            log.close()
        models = wait_ready(
            f"http://127.0.0.1:{self.port}/v1/models",
            self.process,
            self.startup_timeout,
        )
        data = models.get("data")
        if not isinstance(data, list) or not data or not data[0].get("id"):
            raise RuntimeError(f"{self.name}: /v1/models returned no model")
        self.model_id = str(data[0]["id"])

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()

    def size(self) -> int:
        try:
            return self.log_path.stat().st_size
        except FileNotFoundError:
            return 0

    def log_since(self, offset: int) -> str:
        with self.log_path.open("rb") as handle:
            handle.seek(offset)
            return handle.read().decode(errors="replace")


class RssSampler:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_kib = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            rss = int(result.stdout.strip())
        except ValueError:
            return
        self.peak_kib = max(self.peak_kib, rss)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(0.1)
        self._sample()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        self.thread.join()
        return self.peak_kib / (1024.0 * 1024.0)


def parse_ple_cold_evidence(text: str) -> dict | None:
    rows = []
    for line in text.splitlines():
        position = line.find(PLE_COLD_LOG_PREFIX)
        if position < 0:
            continue
        payload = line[position + len(PLE_COLD_LOG_PREFIX):]
        try:
            rows.append(json.loads(payload))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"DS4 PLE cold evidence is not valid JSON: {error}"
            ) from error
    if len(rows) > 1:
        raise ValueError("DS4 log contains multiple PLE cold evidence records")
    return rows[0] if rows else None


def parse_qwen_mtp_timing(text: str) -> dict | None:
    records = []
    for line in text.splitlines():
        position = line.find(DS4_MTP_TIMING_PREFIX)
        if position < 0:
            continue
        record = line[position:].strip()
        match = DS4_MTP_TIMING_RE.fullmatch(record)
        if match is None:
            raise ValueError(f"malformed DS4 Qwen MTP timing line: {record}")
        drafted, accepted, target_tokens, cycle_ms, verifier = match.groups()
        drafted = int(drafted)
        accepted = int(accepted)
        target_tokens = int(target_tokens)
        if accepted > drafted:
            raise ValueError(
                "malformed DS4 Qwen MTP timing line: accepted exceeds drafted"
            )
        if target_tokens != 1 + accepted:
            raise ValueError(
                "malformed DS4 Qwen MTP timing line: "
                "target_tokens must equal 1 + accepted"
            )
        records.append({
            "drafted": drafted,
            "accepted": accepted,
            "target_tokens": target_tokens,
            "cycle_ms": float(cycle_ms),
            "verifier": verifier or "unlabeled",
        })
    if not records:
        return None
    verifier = {}
    for record in records:
        mode = record["verifier"]
        verifier[mode] = verifier.get(mode, 0) + 1
    return {
        "schema": MTP_TIMING_SCHEMA,
        "version": 1,
        "cycles": len(records),
        "drafted": sum(record["drafted"] for record in records),
        "max_drafted": max(record["drafted"] for record in records),
        "accepted": sum(record["accepted"] for record in records),
        "target_tokens": sum(record["target_tokens"] for record in records),
        "cycle_ms": sum(record["cycle_ms"] for record in records),
        "verifier": verifier,
    }


def require_ds4_mtp_timing(evidence: dict | None,
                           completion_tokens: int) -> None:
    if not isinstance(evidence, dict) or (
            evidence.get("schema") != MTP_TIMING_SCHEMA) or (
            evidence.get("version") != 1):
        raise RuntimeError("DS4 MTP phase emitted no supported timing evidence")
    try:
        cycles = int(evidence["cycles"])
        drafted = int(evidence["drafted"])
        max_drafted = int(evidence["max_drafted"])
        accepted = int(evidence["accepted"])
        target_tokens = int(evidence["target_tokens"])
        block = int(evidence.get("verifier", {}).get("block", 0))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RuntimeError("DS4 MTP timing evidence is incomplete") from error
    if cycles <= 0 or drafted <= 0 or accepted <= 0:
        raise RuntimeError("DS4 MTP phase performed no successful speculation")
    if accepted > drafted:
        raise RuntimeError("DS4 MTP accepted more tokens than it drafted")
    if max_drafted <= 0 or max_drafted > 4 or max_drafted > drafted:
        raise RuntimeError("DS4 MTP evidence exceeds the configured draft depth")
    if target_tokens != cycles + accepted:
        raise RuntimeError("DS4 MTP cycle/token accounting is inconsistent")
    if block <= 0:
        raise RuntimeError("DS4 MTP block verifier was not observed")
    if target_tokens != completion_tokens:
        raise RuntimeError(
            "DS4 MTP timing target-token total differs from completion usage"
        )


def parse_ds4_log(text: str, completion_tokens: int) -> dict:
    prefill = list(DS4_PREFILL_RE.finditer(text))
    if not prefill:
        raise ValueError("DS4 log has no completed prefill timing")
    cached, prompt, suffix, seconds = prefill[-1].groups()
    cached_n = int(cached)
    prompt_n = int(prompt)
    suffix_n = int(suffix)
    prefill_s = float(seconds)
    if suffix_n != prompt_n - cached_n or prefill_s <= 0:
        raise ValueError("DS4 prefill timing is inconsistent")
    result = {
        "prompt_tokens": prompt_n,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_n,
        "prefill_seconds": prefill_s,
        "prefill_tps": suffix_n / prefill_s,
    }
    chunks = list(DS4_CHUNK_RE.finditer(text))
    if chunks:
        result["selected_chunk"] = int(chunks[-1].group(1))
    cold_evidence = parse_ple_cold_evidence(text)
    if cold_evidence is not None:
        result["ple_cold_evidence"] = cold_evidence
    mtp_timing = parse_qwen_mtp_timing(text)
    if mtp_timing is not None:
        result["mtp_timing"] = mtp_timing
    decode = [
        match for match in DS4_DECODE_RE.finditer(text)
        if int(match.group(1)) == completion_tokens
    ]
    if completion_tokens >= 50:
        if not decode:
            raise ValueError(
                f"DS4 log has no final {completion_tokens}-token decode timing"
            )
        result["decode_tps"] = float(decode[-1].group(2))
        result["decode_seconds"] = float(decode[-1].group(3))
    return result


def wait_for_metrics(server: ServerProcess, offset: int,
                     completion_tokens: int) -> dict:
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        text = server.log_since(offset)
        try:
            return parse_ds4_log(text, completion_tokens)
        except ValueError as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"{server.name}: {last_error}")


def measured_request(server: ServerProcess, prompt: str, expected_prompt: int,
                     max_tokens: int, mtp: bool,
                     require_rusage_v2: bool = False) -> dict:
    assert server.process is not None and server.model_id is not None
    offset = server.size()
    payload = {
        "model": server.model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 1,
        "stream": False,
    }
    if mtp:
        payload["enable_mtp"] = True
    sampler = RssSampler(server.process.pid)
    rusage_before = proc_pid_rusage_v2(server.process.pid)
    if require_rusage_v2 and rusage_before is None:
        raise RuntimeError(
            "cold PLE phase requires Darwin proc_pid_rusage V2"
        )
    sampler.start()
    started = time.monotonic()
    try:
        response = url_json(
            f"http://127.0.0.1:{server.port}/v1/completions",
            payload,
            timeout=7200.0,
        )
    finally:
        elapsed = time.monotonic() - started
        peak_gib = sampler.stop()
    rusage_after = proc_pid_rusage_v2(server.process.pid)
    if require_rusage_v2 and rusage_after is None:
        raise RuntimeError(
            "cold PLE phase could not read post-request proc_pid_rusage V2"
        )
    usage = response.get("usage", {})
    response_prompt = int(usage.get("prompt_tokens", -1))
    response_completion = int(usage.get("completion_tokens", -1))
    if response_prompt != expected_prompt:
        raise RuntimeError(
            f"{server.name}: tokenized prompt to {response_prompt}, "
            f"expected {expected_prompt}"
        )
    if response_completion != max_tokens:
        raise RuntimeError(
            f"{server.name}: generated {response_completion} tokens, "
            f"expected fixed {max_tokens}"
        )
    metrics = wait_for_metrics(server, offset, response_completion)
    if metrics["prompt_tokens"] != response_prompt:
        raise RuntimeError(f"{server.name}: log/response prompt count differs")
    if metrics["cached_tokens"] != 0:
        raise RuntimeError(
            f"{server.name}: request reused {metrics['cached_tokens']} tokens"
        )
    if mtp and server.name == "ds4" and max_tokens > 1:
        require_ds4_mtp_timing(
            metrics.get("mtp_timing"), response_completion
        )
    metrics["request_elapsed_ms"] = elapsed * 1000.0
    metrics["peak_memory_gib"] = peak_gib
    if rusage_before is not None and rusage_after is not None:
        metrics["rusage_v2"] = rusage_v2_evidence(
            rusage_before, rusage_after
        )
    decode_tps = metrics.get("decode_tps")
    metrics["first_token_ms"] = (
        metrics["prefill_seconds"] * 1000.0 +
        (1000.0 / decode_tps if decode_tps else
         max(0.0, elapsed * 1000.0 - metrics["prefill_seconds"] * 1000.0))
    )
    if decode_tps:
        total_seconds = (
            metrics["prefill_seconds"] + metrics["decode_seconds"]
        )
        metrics["total_tps"] = (
            response_prompt + response_completion
        ) / total_seconds
    return metrics


def add_probe_adjusted_timing(metrics: dict, evidence: dict) -> None:
    probe_ms = float(evidence.get("probe_ms", 0.0))
    if not (probe_ms > 0.0 and probe_ms < float("inf")):
        raise RuntimeError("DS4 cold evidence has no finite positive probe_ms")
    raw = {
        name: float(metrics[name])
        for name in (
            "prefill_seconds", "prefill_tps", "first_token_ms",
            "request_elapsed_ms",
        )
    }
    adjusted_prefill_seconds = raw["prefill_seconds"] - probe_ms / 1000.0
    adjusted_first_token_ms = raw["first_token_ms"] - probe_ms
    adjusted_request_ms = raw["request_elapsed_ms"] - probe_ms
    if min(adjusted_prefill_seconds, adjusted_first_token_ms,
           adjusted_request_ms) <= 0.0:
        raise RuntimeError("PLE probe time exceeds an observed cold timing")
    suffix_tokens = int(metrics["prompt_tokens"]) - int(metrics["cached_tokens"])
    metrics["timing"] = {
        "probe_ms": probe_ms,
        "raw_observed": raw,
        "probe_adjusted": {
            "prefill_seconds": adjusted_prefill_seconds,
            "prefill_tps": suffix_tokens / adjusted_prefill_seconds,
            "first_token_ms": adjusted_first_token_ms,
            "request_elapsed_ms": adjusted_request_ms,
        },
    }


def load_prompts(path: Path) -> tuple[dict[tuple[str, int], dict], dict]:
    records = json.loads((path / "manifest.json").read_text())
    if not isinstance(records, list):
        raise ValueError("prompt manifest must be an array")
    by_key = {}
    first_tokens = set()
    for record in records:
        key = (record["case"], int(record["sample"]))
        if key in by_key:
            raise ValueError(f"duplicate prompt record {key}")
        prompt_path = path / record["path"]
        raw = prompt_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"prompt checksum mismatch: {prompt_path}")
        first = int(record["first_token_id"])
        if first in first_tokens:
            raise ValueError(f"first token {first} is not cache-distinct")
        first_tokens.add(first)
        by_key[key] = {**record, "text": raw.decode(), "prompt_id": digest}
    warmup = by_key.get(("warmup-10k+1", 0))
    if not warmup:
        raise ValueError("prompt manifest has no warmup-10k+1 sample")
    return by_key, warmup


def case_object(document: dict, name: str, kind: str) -> dict:
    for case in document["cases"]:
        if case["name"] == name:
            if case["kind"] != kind:
                raise ValueError(f"existing case {name} has a different kind")
            return case
    case = {"name": name, "kind": kind, "ds4": []}
    if name == "cold-ple-10k+1":
        case["ple_cache"] = "darwin-mincore-target-pages"
    if name == "mtp-10k+500":
        case["mtp_enabled"] = True
    document["cases"].append(case)
    return case


def new_document() -> dict:
    return {
        "metadata": {
            "hardware": "Apple M3 Ultra",
            "memory_gib": 512,
            "warmups_discarded": 1,
            "batch_size": 1,
            "mtp_enabled": False,
            "execution_sequence": [],
        },
        "cases": [],
        "chunk_comparisons": [],
    }


def reject_managed_mtp_extras(args) -> None:
    managed_ds4 = {"--mtp", "--mtp-model", "--mtp-draft", "--mtp-timing"}
    for extra in args.ds4_extra:
        flag = str(extra).split("=", 1)[0]
        if flag in managed_ds4:
            raise ValueError(
                f"--ds4-extra cannot override harness-managed MTP flag {flag}"
            )


def command_lines(args, mtp: bool, log_dir: Path,
                  cold_evidence: bool = False) -> ServerProcess:
    reject_managed_mtp_extras(args)
    ds4_prefill_chunk = "8192" if cold_evidence else args.prefill_chunk
    ds4_command = [
        str(args.ds4_server), "--model", str(args.ds4_model),
        "--ple", str(args.ple), "--metal", "--ctx", str(args.ctx),
        "--host", "127.0.0.1", "--port", str(args.ds4_port),
        "--prefill-chunk", ds4_prefill_chunk,
    ]
    ds4_command.extend(args.ds4_extra)
    if mtp:
        if not args.ds4_mtp:
            raise ValueError("--ds4-mtp is required for the MTP phase")
        ds4_command.extend([
            "--mtp-model", str(args.ds4_mtp), "--mtp",
            "--mtp-draft", str(MTP_DRAFT_TOKENS), "--mtp-timing",
        ])

    phase = "mtp" if mtp else "core"
    ds4_environment = None
    if cold_evidence:
        ds4_environment = dict(os.environ)
        ds4_environment["DS4_QWEN4_PLE_COLD_EVIDENCE"] = "1"
    return ServerProcess(
        "ds4", ds4_command, args.ds4_port,
        log_dir / f"ds4-{phase}.log", args.startup_timeout,
        environment=ds4_environment,
    )


def stat_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def pack_ple_identity(ds4_model: Path, ple: Path) -> dict:
    manifest_path = ds4_model.resolve().parent / PACK_MANIFEST_NAME
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Qwen pack manifest {manifest_path}: {error}") from error
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Qwen pack manifest has no artifacts: {manifest_path}")
    ple_rows = [row for row in artifacts if isinstance(row, dict) and
                row.get("kind") == "ple"]
    if len(ple_rows) != 1:
        raise ValueError("Qwen pack manifest must contain exactly one PLE artifact")
    artifact = ple_rows[0]
    artifact_path = (manifest_path.parent / str(artifact.get("path", ""))).resolve()
    if artifact_path != ple.resolve():
        raise ValueError(
            f"--ple {ple.resolve()} is not the manifest PLE {artifact_path}"
        )
    ple_sha256 = artifact.get("sha256")
    if not isinstance(ple_sha256, str) or len(ple_sha256) != 64:
        raise ValueError("Qwen pack manifest has no valid PLE SHA-256")
    try:
        ple_bytes = int(artifact.get("bytes", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("Qwen pack manifest has invalid PLE size") from error
    if ple_bytes != ple.stat().st_size:
        raise ValueError("Qwen pack manifest PLE size does not match --ple")
    pack_id = manifest.get("pack_id")
    tensor_digest = manifest.get("tensor_manifest_sha256")
    if not isinstance(pack_id, str) or not pack_id or (
            not isinstance(tensor_digest, str) or len(tensor_digest) != 64):
        raise ValueError("Qwen pack manifest has incomplete pack identity")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "pack_id": pack_id,
        "tensor_manifest_sha256": tensor_digest,
        "ple_sha256": ple_sha256,
        "ple_bytes": ple_bytes,
    }


def ssd_backing_evidence(path: Path) -> dict:
    resolved = path.resolve()
    df = subprocess.run(
        ["df", "-P", str(resolved)], check=False, capture_output=True, text=True
    )
    if df.returncode != 0:
        raise ValueError(f"df -P failed for {resolved}: {df.stderr.strip()}")
    lines = [line for line in df.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"df -P returned no mount for {resolved}")
    fields = lines[-1].split(None, 5)
    if len(fields) != 6:
        raise ValueError(f"cannot parse df -P mount for {resolved}")
    filesystem, mount_point = fields[0], fields[5]
    if not filesystem.startswith("/dev/"):
        raise ValueError(
            f"PLE must be on a local block device, got {filesystem}"
        )
    diskutil = subprocess.run(
        ["diskutil", "info", "-plist", mount_point],
        check=False, capture_output=True,
    )
    if diskutil.returncode != 0:
        stderr = diskutil.stderr.decode(errors="replace").strip()
        raise ValueError(f"diskutil info failed for {mount_point}: {stderr}")
    try:
        info = plistlib.loads(diskutil.stdout)
    except (plistlib.InvalidFileException, ValueError) as error:
        raise ValueError(f"diskutil returned invalid plist for {mount_point}") from error
    device_node = info.get("DeviceNode")
    if not isinstance(device_node, str) or not device_node.startswith("/dev/"):
        raise ValueError("diskutil did not resolve a local PLE device")
    if info.get("SolidState") is not True:
        raise ValueError(
            f"PLE backing device {device_node} is not reported as solid state"
        )
    return {
        "schema": SSD_BACKING_SCHEMA,
        "version": 1,
        "probe": "df-P+diskutil-info-plist",
        "filesystem": filesystem,
        "mount_point": mount_point,
        "device_node": device_node,
        "device_identifier": info.get("DeviceIdentifier"),
        "parent_whole_disk": info.get("ParentWholeDisk"),
        "bus_protocol": info.get("BusProtocol"),
        "internal": info.get("Internal"),
        "local": True,
        "solid_state": True,
    }


def ple_binding(args) -> dict:
    return {
        "schema": PLE_BINDING_SCHEMA,
        "version": 1,
        "path": str(args.ple.resolve()),
        "descriptor_identity": stat_identity(args.ple),
        "pack": pack_ple_identity(args.ds4_model, args.ple),
        "ssd_backing": ssd_backing_evidence(args.ple),
    }


def require_cold_ple_binding(document: dict, binding: dict) -> None:
    cold = document.get("metadata", {}).get("cold_ple_configuration")
    if not isinstance(cold, dict) or cold.get("ple_binding") != binding:
        raise ValueError(
            "benchmark phase PLE descriptor/pack/SSD identity differs from cold evidence"
        )


def document_has_measurements(document: dict) -> bool:
    for case in document.get("cases", []):
        if not isinstance(case, dict):
            continue
        if case.get("ds4"):
            return True
    return bool(document.get("chunk_comparisons"))


def prepare_cold_phase_order(document: dict) -> bool:
    """Return true when the first-request phase still needs to run."""
    metadata = document.setdefault("metadata", {})
    sequence = metadata.setdefault("execution_sequence", [])
    cold = next((
        case for case in document.get("cases", [])
        if isinstance(case, dict) and case.get("name") == COLD_PLE_CASE[0]
    ), None)
    completed = bool(cold and cold.get("ds4"))
    if completed:
        if not sequence or sequence[0] != COLD_PHASE_MARKER:
            raise ValueError(
                "cold PLE evidence is not the first recorded phase"
            )
        return False
    allowed_resume = sequence == [COLD_PHASE_MARKER]
    if document_has_measurements(document) or (sequence and not allowed_resume):
        raise ValueError(
            "cold PLE evidence must be collected before every warmup or "
            "other measurement; use a fresh output file"
        )
    if not sequence:
        sequence.append(COLD_PHASE_MARKER)
    return True


def record_phase(document: dict, marker: str) -> None:
    sequence = document["metadata"].setdefault("execution_sequence", [])
    if marker not in sequence:
        sequence.append(marker)


def run_cold_ple_phase(args, document: dict, prompts: dict,
                       output: Path) -> None:
    binding = ple_binding(args)
    if not prepare_cold_phase_order(document):
        configuration = document["metadata"].get("cold_ple_configuration", {})
        if configuration.get("ple_binding") != binding:
            raise ValueError(
                "completed cold PLE evidence belongs to a different artifact"
            )
        return
    name, kind, max_tokens = COLD_PLE_CASE
    prompt = prompts.get((name, 0))
    if not prompt:
        raise ValueError(f"missing prompt {name} sample 0")
    if int(prompt.get("tokens", -1)) != COLD_PROMPT_TOKENS:
        raise ValueError(
            f"{name} must contain exactly {COLD_PROMPT_TOKENS} tokens"
        )
    case = case_object(document, name, kind)
    log_dir = output.parent / (output.stem + "-logs")
    ds4 = command_lines(args, False, log_dir, cold_evidence=True)
    ds4.log_path = log_dir / "ds4-cold-ple-first-request.log"
    identity_before = binding["descriptor_identity"]
    document["metadata"]["cold_ple_configuration"] = {
        "ds4_server": str(args.ds4_server.resolve()),
        "ds4_model": str(args.ds4_model.resolve()),
        "ds4_ple": str(args.ple.resolve()),
        "ctx": args.ctx,
        "prefill_chunk": str(COLD_SELECTED_CHUNK),
        "ple_identity_before": identity_before,
        "ple_binding": binding,
    }
    atomic_json(output, document)
    try:
        print("starting DS4-only cold-PLE first-request server", flush=True)
        ds4.start()
        print("measure: cold PLE first request (no warmup)", flush=True)
        metrics = measured_request(
            ds4, prompt["text"], int(prompt["tokens"]), max_tokens,
            False, require_rusage_v2=True,
        )
        evidence = metrics.get("ple_cold_evidence")
        if not isinstance(evidence, dict):
            raise RuntimeError(
                "DS4 cold request emitted no target-page residency evidence"
            )
        if evidence.get("schema") != PLE_COLD_SCHEMA or (
                evidence.get("version") != 1):
            raise RuntimeError("DS4 emitted an unsupported PLE evidence schema")
        target = evidence.get("target", {})
        if metrics.get("prompt_tokens") != COLD_PROMPT_TOKENS or (
                metrics.get("completion_tokens") != COLD_COMPLETION_TOKENS) or (
                metrics.get("selected_chunk") != COLD_SELECTED_CHUNK) or (
                target.get("tokens") != COLD_PROMPT_TOKENS) or (
                target.get("ngram_heads") != 16) or (
                target.get("ngram_rows") != COLD_LOOKUP_ROWS) or (
                target.get("lookup_rows") != COLD_LOOKUP_ROWS) or (
                target.get("row_bytes") != COLD_ROW_BYTES) or (
                target.get("logical_bytes") != COLD_LOGICAL_BYTES):
            raise RuntimeError(
                "cold PLE request did not use the required 10K/1/8K/160K/16MB geometry"
            )
        runtime_identity = evidence.get("ple_identity", {})
        if runtime_identity.get("sha256") != binding["pack"]["ple_sha256"]:
            raise RuntimeError("runtime PLE SHA-256 differs from the bound pack")
        disk_read_delta = metrics.get("rusage_v2", {}).get("delta", {}).get(
            "diskio_bytesread", 0
        )
        if int(disk_read_delta) < COLD_LOGICAL_BYTES:
            raise RuntimeError("cold PLE request read fewer than 16MB from disk")
        add_probe_adjusted_timing(metrics, evidence)
        identity_after = stat_identity(args.ple)
        if identity_after != identity_before:
            raise RuntimeError("PLE artifact changed during the cold request")
        document["metadata"]["cold_ple_configuration"][
            "ple_identity_after"
        ] = identity_after
        metrics["prompt_id"] = prompt["prompt_id"]
        metrics["model_id"] = ds4.model_id
        metrics["ple_identity_before"] = identity_before
        metrics["ple_identity_after"] = identity_after
        case["ds4"].append(metrics)
        atomic_json(output, document)
    finally:
        ds4.stop()


def run_phase(args, document: dict, prompts: dict, warmup: dict,
              output: Path, mtp: bool) -> None:
    cases = MTP_CASES if mtp else CORE_CASES
    log_dir = output.parent / (output.stem + "-logs")
    ds4 = command_lines(args, mtp, log_dir)
    configuration_key = "mtp_configuration" if mtp else "core_configuration"
    binding = ple_binding(args)
    require_cold_ple_binding(document, binding)
    configuration = {
        "ds4_server": str(args.ds4_server.resolve()),
        "ds4_model": str(args.ds4_model.resolve()),
        "ds4_ple": str(args.ple.resolve()),
        "ds4_mtp": str(args.ds4_mtp.resolve()) if mtp else None,
        "ctx": args.ctx,
        "prefill_chunk": args.prefill_chunk,
        "mtp_enabled": mtp,
        "ds4_mtp_draft": MTP_DRAFT_TOKENS if mtp else None,
        "ds4_mtp_timing": mtp,
        "ple_binding": binding,
        "ple_identity_before": binding["descriptor_identity"],
    }
    document["metadata"][configuration_key] = configuration
    record_phase(document, "mtp_ds4" if mtp else "core_ds4")
    atomic_json(output, document)
    try:
        print(f"starting {'MTP' if mtp else 'core'} DS4 server", flush=True)
        ds4.start()
        print("discarding DS4 warmup", flush=True)
        measured_request(ds4, warmup["text"], int(warmup["tokens"]), 1, mtp)
        for name, kind, max_tokens in cases:
            case = case_object(document, name, kind)
            for sample in range(args.samples):
                prompt = prompts.get((name, sample))
                if not prompt:
                    raise ValueError(f"missing prompt {name} sample {sample}")
                if any(row.get("prompt_id") == prompt["prompt_id"]
                       for row in case["ds4"]):
                    print(f"resume: {name} sample {sample} DS4", flush=True)
                    continue
                print(f"measure: {name} sample {sample} DS4", flush=True)
                metrics = measured_request(
                    ds4, prompt["text"], int(prompt["tokens"]),
                    max_tokens, mtp,
                )
                metrics["prompt_id"] = prompt["prompt_id"]
                metrics["model_id"] = ds4.model_id
                case["ds4"].append(metrics)
                atomic_json(output, document)
    finally:
        ds4.stop()
    identity_after = stat_identity(args.ple)
    if identity_after != binding["descriptor_identity"]:
        raise RuntimeError(f"PLE artifact changed during {configuration_key}")
    configuration["ple_identity_after"] = identity_after
    atomic_json(output, document)


def run_chunk_phase(args, document: dict, prompts: dict, warmup: dict,
                    output: Path) -> None:
    log_dir = output.parent / (output.stem + "-logs")
    samples_by_mode = {}
    selected_by_mode = {}
    original_mode = args.prefill_chunk
    binding = ple_binding(args)
    require_cold_ple_binding(document, binding)
    configuration = {
        "ds4_server": str(args.ds4_server.resolve()),
        "ds4_model": str(args.ds4_model.resolve()),
        "ds4_ple": str(args.ple.resolve()),
        "ctx": args.ctx,
        "ple_binding": binding,
        "ple_identity_before": binding["descriptor_identity"],
    }
    document["metadata"]["chunk_configuration"] = configuration
    record_phase(document, "chunk_uncached_prefix")
    atomic_json(output, document)
    try:
        for mode in ("2048", "8192", "auto"):
            args.prefill_chunk = mode
            ds4 = command_lines(args, False, log_dir)
            ds4.log_path = log_dir / f"ds4-chunk-{mode}.log"
            mode_samples = []
            try:
                print(f"starting DS4 chunk={mode} server", flush=True)
                ds4.start()
                print(f"discarding DS4 chunk={mode} warmup", flush=True)
                measured_request(
                    ds4, warmup["text"], int(warmup["tokens"]), 1, False
                )
                for sample in range(args.samples):
                    prompt = prompts.get(("prefill-10k+1", sample))
                    if not prompt:
                        raise ValueError(
                            f"missing prompt prefill-10k+1 sample {sample}"
                        )
                    print(
                        f"measure: chunk={mode} sample {sample}", flush=True
                    )
                    metrics = measured_request(
                        ds4, prompt["text"], int(prompt["tokens"]), 1, False
                    )
                    metrics["prompt_id"] = prompt["prompt_id"]
                    metrics["model_id"] = ds4.model_id
                    mode_samples.append(metrics)
                selected = {row.get("selected_chunk") for row in mode_samples}
                if len(selected) != 1 or None in selected:
                    raise RuntimeError(
                        f"chunk={mode}: runtime did not report one chunk selection"
                    )
                selected_by_mode[mode] = selected.pop()
                samples_by_mode[mode] = mode_samples
            finally:
                ds4.stop()
    finally:
        args.prefill_chunk = original_mode
    row = {
        "name": "uncached-prefix-10k",
        "chunk_2048_tps": statistics.median(
            sample["prefill_tps"] for sample in samples_by_mode["2048"]
        ),
        "chunk_8192_tps": statistics.median(
            sample["prefill_tps"] for sample in samples_by_mode["8192"]
        ),
        "auto_selected": selected_by_mode["auto"],
        "samples": samples_by_mode,
    }
    document["chunk_comparisons"] = [row]
    identity_after = stat_identity(args.ple)
    if identity_after != binding["descriptor_identity"]:
        raise RuntimeError("PLE artifact changed during chunk_configuration")
    configuration["ple_identity_after"] = identity_after
    atomic_json(output, document)


def verify_paths(args) -> None:
    files = (args.ds4_server, args.ds4_model, args.ple)
    for path in files:
        if not path.is_file():
            raise ValueError(f"required file does not exist: {path}")
    if "mtp" in args.phase and (not args.ds4_mtp or not args.ds4_mtp.is_file()):
        raise ValueError(f"DS4 MTP sidecar does not exist: {args.ds4_mtp}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds4-server", type=Path, default=Path("./ds4-server"))
    parser.add_argument("--ds4-model", type=Path, required=True)
    parser.add_argument("--ple", type=Path, required=True)
    parser.add_argument("--ds4-mtp", type=Path)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", action="append", choices=("core", "mtp", "chunks"), default=[]
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--ctx", type=int, default=101_024)
    parser.add_argument("--ds4-port", type=int, default=18123)
    parser.add_argument("--prefill-chunk", choices=("auto", "2048", "4096", "8192"), default="auto")
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--ds4-extra", action="append", default=[])
    args = parser.parse_args(argv)
    if not args.phase:
        args.phase = ["core"]
    if args.samples < 3 or args.samples % 2 == 0:
        parser.error("--samples must be an odd number of at least three")
    verify_paths(args)
    prompts, warmup = load_prompts(args.prompt_dir)
    if args.output.exists():
        document = json.loads(args.output.read_text())
    else:
        document = new_document()
        atomic_json(args.output, document)
    # Every acceptance phase is anchored to one DS4-only first request.  This
    # runs (or verifies a completed binding) before any warmup or phase marker.
    run_cold_ple_phase(args, document, prompts, args.output)
    for phase in args.phase:
        if phase == "chunks":
            run_chunk_phase(args, document, prompts, warmup, args.output)
        else:
            run_phase(
                args, document, prompts, warmup, args.output,
                mtp=phase == "mtp",
            )
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError,
            urllib.error.URLError) as error:
        print(f"qwen4-benchmark: {error}", file=sys.stderr)
        raise SystemExit(2)
