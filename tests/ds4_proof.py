#!/usr/bin/env python3
"""General DS4 engine proof runner.

The runner executes process-isolated engine profiles across prompt cases and
comparison contracts.  MTP proof is the first built-in suite, but the data model
is intentionally engine-wide: profiles describe backend/env/arguments, suites
describe what behavior is under proof, and contracts define pass/fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROOF_PROMPTS = [
    "List 100 prime numbers, comma-separated, just numbers.",
    "Write a concise explanation of how speculative decoding works, then give three caveats.",
    "Write a short C function that returns the maximum value in an int array.",
    "Summarize why GPU kernel launch overhead matters for small decode batches.",
    "Give eight terse bullet points about deterministic testing for inference engines.",
    "Continue this sequence with one item per line: alpha, beta, gamma, delta.",
    "Explain the tradeoff between exact speculative decoding and approximate draft acceptance.",
    "Write a compact JSON object with keys name, purpose, risks, and validation.",
]

DEFAULT_PROMPTS = DEFAULT_PROOF_PROMPTS[:2]


@dataclass(frozen=True)
class BudgetPreset:
    name: str
    tokens: int
    prompt_count: int
    description: str


BUDGET_PRESETS = {
    "smoke": BudgetPreset(
        "smoke",
        tokens=64,
        prompt_count=2,
        description="Fast crash and obvious drift check.",
    ),
    "candidate": BudgetPreset(
        "candidate",
        tokens=512,
        prompt_count=4,
        description="Default optimization proof loop beyond the load-dominated region.",
    ),
    "default-on": BudgetPreset(
        "default-on",
        tokens=1024,
        prompt_count=8,
        description="Stronger bar before making an optimization default-on.",
    ),
    "nightly": BudgetPreset(
        "nightly",
        tokens=2048,
        prompt_count=8,
        description="Expensive pre-PR confidence run across the built-in prompt suite.",
    ),
}

FAST_MTP_ENV = {
    "DS4_CUDA_MTP_TOP2": "1",
    "DS4_CUDA_MTP_VERIFY_TOP2": "1",
    "DS4_MTP_NONEXACT_FAST": "1",
}

SHADOW_RE = re.compile(
    r"mtp shadow b_n2_q8 .*?\sagree=(?P<agree>[01]).*?"
    r"\slogit_agree=(?P<logit_agree>[01]).*?"
    r"logit_max_abs=(?P<max_abs>[-+0-9.eE]+).*?"
    r"logit_rms=(?P<rms>[-+0-9.eE]+)"
)
VERIFY_V2_SHADOW_RE = re.compile(
    r"mtp shadow verify_v2_decode3 ok=(?P<ok>[01]) committed=(?P<committed>\d+) "
    r"row0=(?P<row0>-?\d+) row1=(?P<row1>-?\d+) "
    r"draft1=(?P<draft1>-?\d+) draft2=(?P<draft2>-?\d+) "
    r"margin0=(?P<margin0>[-+0-9.eE]+) margin1=(?P<margin1>[-+0-9.eE]+)"
)
GEN_STEP_PROFILE_RE = re.compile(
    r"ds4: gen step profile cycle=(?P<cycle>\d+) pos=(?P<pos>\d+) mtp=(?P<mtp>[01]) "
    r"accepted=(?P<accepted>\d+) eval_ms=(?P<eval_ms>[-+0-9.eE]+) "
    r"generated_before=(?P<generated_before>\d+)"
)
WEIGHT_PLAN_RE = re.compile(
    r"ds4_weight_server: (?P<model>\w+) plan model=(?P<model_gib>[-+0-9.eE]+) GiB "
    r"raw_tensor_ranges=(?P<raw_gib>[-+0-9.eE]+) GiB ranges=(?P<ranges>\d+)"
)
WEIGHT_MEMORY_RE = re.compile(
    r"ds4_weight_server: memory preflight .*?need=(?P<need_gib>[-+0-9.eE]+) GiB "
    r"reserve=(?P<reserve_gib>[-+0-9.eE]+) GiB free=(?P<free_gib>[-+0-9.eE]+) GiB "
    r"total=(?P<total_gib>[-+0-9.eE]+) GiB"
)
WEIGHT_UPLOAD_RE = re.compile(
    r"ds4_weight_server: (?P<model>\w+) uploaded (?P<gib>[-+0-9.eE]+) GiB across (?P<ranges>\d+) ranges"
)
WEIGHT_READY_RE = re.compile(
    r"ds4_weight_server: ready manifest=(?P<manifest>\S+) ranges=(?P<ranges>\d+)"
)
WEIGHT_LOCK_RE = re.compile(r"ds4_weight_server: acquired lock (?P<path>\S+)")
WEIGHT_VMM_SUPPORT_RE = re.compile(
    r"ds4_weight_server: vmm support vmm=(?P<vmm>\d+) posix_fd=(?P<posix_fd>\d+) "
    r"uva=(?P<uva>\d+) gran_min=(?P<gran_min>\d+) gran_rec=(?P<gran_rec>\d+)"
)
WEIGHT_VMM_PLAN_RE = re.compile(
    r"ds4_weight_server: (?P<model>\w+) vmm plan logical=(?P<logical_gib>[-+0-9.eE]+) GiB "
    r"allocated=(?P<allocated_gib>[-+0-9.eE]+) GiB granularity=(?P<granularity>\d+)"
)
WEIGHT_BACKEND_RE = re.compile(
    r"ds4_weight_server: backend=(?P<backend>\w+) logical_upload=(?P<logical_gib>[-+0-9.eE]+) GiB "
    r"allocation_plan=(?P<allocated_gib>[-+0-9.eE]+) GiB"
)
WEIGHT_BROKER_RE = re.compile(r"ds4_weight_server: broker listening (?P<path>\S+)")
WEIGHT_BROKER_SERVED_RE = re.compile(
    r"ds4_weight_server: broker served alloc=(?P<alloc>\d+) bytes=(?P<bytes>\d+) requests=(?P<requests>\d+)"
)
WEIGHT_SHUTDOWN_RE = re.compile(r"ds4_weight_server: shutting down(?: broker_requests=(?P<requests>\d+))?")


@dataclass(frozen=True)
class EngineProfile:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    backend: str = "cuda"
    use_mtp: bool = False
    mtp_draft: int = 2
    baseline: bool = False


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str


@dataclass(frozen=True)
class Contract:
    name: str
    baseline: str
    candidate: str
    kind: str = "exact_bytes"


@dataclass
class RunResult:
    prompt_id: str
    profile: str
    suite: str
    rc: int
    cmd: list[str]
    out_path: str
    log_path: str
    stdout_sha256: str | None = None
    stdout_bytes: int = 0
    wall_ms: float = 0.0
    timing: dict[str, Any] = field(default_factory=dict)
    shadow: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    prompt_id: str
    contract: str
    baseline: str
    candidate: str
    kind: str
    passed: bool
    first_diff: int | None = None
    baseline_snippet: str = ""
    candidate_snippet: str = ""
    reason: str = ""


@dataclass
class WeightServerState:
    enabled: bool = False
    owned: bool = False
    bin_path: str = ""
    scope: str = "both"
    backend: str = "ipc"
    manifest_path: str = ""
    log_path: str = ""
    cmd: list[str] = field(default_factory=list)
    pid: int | None = None
    preflight_cmd: list[str] = field(default_factory=list)
    preflight_log_path: str = ""
    preflight_rc: int | None = None
    preflight_wall_ms: float = 0.0
    preflight_telemetry: dict[str, Any] = field(default_factory=dict)
    start_wall_ms: float = 0.0
    ready: bool = False
    cleanup: str = "not_started"
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class WeightServer:
    def __init__(
        self,
        *,
        bin_path: str,
        base_model: str,
        mtp_model: str | None,
        manifest_path: Path,
        log_path: Path,
        ready_timeout_s: float,
        reserve_gb: int,
        span_mb: int | None,
        copy_chunk_mb: int | None,
        extra_args: list[str],
        preflight_timeout_s: float,
        scope: str,
        backend: str,
    ) -> None:
        self.bin_path = bin_path
        self.base_model = base_model
        self.mtp_model = mtp_model
        self.manifest_path = manifest_path
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.reserve_gb = reserve_gb
        self.span_mb = span_mb
        self.copy_chunk_mb = copy_chunk_mb
        self.extra_args = extra_args
        self.preflight_timeout_s = preflight_timeout_s
        self.scope = scope
        self.backend = backend
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_f: Any = None
        self.state = WeightServerState(
            enabled=True,
            owned=True,
            bin_path=bin_path,
            scope=scope,
            backend=backend,
            manifest_path=str(manifest_path),
            log_path=str(log_path),
            preflight_log_path=str(log_path.with_suffix(".preflight.log")),
        )

    def command(self) -> list[str]:
        cmd = [
            self.bin_path,
            "--base",
            self.base_model,
            "--manifest",
            str(self.manifest_path),
            "--scope",
            self.scope,
            "--backend",
            self.backend,
            "--exit-on-parent-pid",
            str(os.getpid()),
            "--reserve-gb",
            str(self.reserve_gb),
        ]
        if self.mtp_model:
            cmd.extend(["--mtp", self.mtp_model])
        if self.span_mb is not None:
            cmd.extend(["--span-mb", str(self.span_mb)])
        if self.copy_chunk_mb is not None:
            cmd.extend(["--copy-chunk-mb", str(self.copy_chunk_mb)])
        cmd.extend(self.extra_args)
        return cmd

    def preflight(self) -> None:
        cmd = [*self.command(), "--dry-run"]
        self.state.preflight_cmd = cmd
        preflight_log_path = Path(self.state.preflight_log_path)
        preflight_log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            with preflight_log_path.open("wb") as log_f:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=self.preflight_timeout_s,
                )
        except subprocess.TimeoutExpired as e:
            self.state.preflight_wall_ms = (time.monotonic() - t0) * 1000.0
            self.state.error = f"ds4_weight_server dry-run timed out after {self.preflight_timeout_s:g}s"
            raise TimeoutError(self.state.error) from e
        self.state.preflight_wall_ms = (time.monotonic() - t0) * 1000.0
        self.state.preflight_rc = proc.returncode
        self.state.preflight_telemetry = self.parse_log_file(preflight_log_path)
        if proc.returncode != 0:
            self.state.error = (
                f"ds4_weight_server dry-run failed rc={proc.returncode}: "
                f"{self.log_tail_from(preflight_log_path)}"
            )
            raise RuntimeError(self.state.error)

    def start(self) -> WeightServerState:
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_f = self.log_path.open("wb")
        cmd = self.command()
        self.state.cmd = cmd
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.state.pid = self.proc.pid
        deadline = t0 + self.ready_timeout_s
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                self.state.start_wall_ms = (time.monotonic() - t0) * 1000.0
                self.state.error = f"ds4_weight_server exited before ready rc={rc}: {self.log_tail()}"
                raise RuntimeError(self.state.error)
            if self.manifest_path.exists():
                validate_weight_manifest(self.manifest_path, self.base_model, self.mtp_model, self.scope)
                if self.log_f:
                    self.log_f.flush()
                self.state.telemetry = self.parse_log_file(self.log_path)
                self.state.start_wall_ms = (time.monotonic() - t0) * 1000.0
                self.state.ready = True
                self.state.cleanup = "pending"
                return self.state
            time.sleep(0.25)
        self.state.start_wall_ms = (time.monotonic() - t0) * 1000.0
        self.state.error = f"timed out waiting for ds4_weight_server manifest: {self.log_tail()}"
        raise TimeoutError(self.state.error)

    def log_tail(self, limit: int = 4000) -> str:
        if self.log_f:
            self.log_f.flush()
        return self.log_tail_from(self.log_path, limit=limit)

    @staticmethod
    def log_tail_from(path: Path, limit: int = 4000) -> str:
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", errors="replace").replace("\n", "\\n")

    @staticmethod
    def parse_log_file(path: Path) -> dict[str, Any]:
        try:
            return parse_weight_server_log(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return {}

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            self.state.cleanup = "not_started"
            if self.log_f:
                self.log_f.close()
                self.log_f = None
            return
        rc = self.proc.poll()
        if rc is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
                self.state.cleanup = "terminated"
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
                self.state.cleanup = "killed"
            except ProcessLookupError:
                self.state.cleanup = "already_exited"
        else:
            self.state.cleanup = f"already_exited_rc={rc}"
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        self.state.telemetry = self.parse_log_file(self.log_path)


def parse_env_assignments(spec: str) -> tuple[str, dict[str, str]]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("expected NAME:KEY=VALUE,...")
    name, env_spec = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("profile name is empty")
    env: dict[str, str] = {}
    for item in re.split(r"[,;]", env_spec):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"environment item lacks '=': {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("environment key is empty")
        env[key] = value
    if not env:
        raise argparse.ArgumentTypeError("profile has no environment flags")
    return name, env


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "on"}


def first_diff(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def snippet(data: bytes, pos: int | None, width: int = 80) -> str:
    if pos is None:
        return ""
    start = max(0, pos - width // 2)
    end = min(len(data), pos + width // 2)
    return data[start:end].decode("utf-8", errors="replace").replace("\n", "\\n")


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def parse_shadow(log_text: str) -> dict[str, Any]:
    checks = 0
    decision_bad = 0
    logit_bad = 0
    max_abs = 0.0
    max_rms = 0.0
    v2_checks = 0
    v2_failed = 0
    v2_accept3 = 0
    v2_accept2 = 0
    v2_accept1 = 0
    v2_max_margin0 = 0.0
    v2_max_margin1 = 0.0
    gen_steps: list[dict[str, Any]] = []
    for m in SHADOW_RE.finditer(log_text):
        checks += 1
        if m.group("agree") != "1":
            decision_bad += 1
        if m.group("logit_agree") != "1":
            logit_bad += 1
        max_abs = max(max_abs, float(m.group("max_abs")))
        max_rms = max(max_rms, float(m.group("rms")))
    for m in VERIFY_V2_SHADOW_RE.finditer(log_text):
        v2_checks += 1
        if m.group("ok") != "1":
            v2_failed += 1
        committed = int(m.group("committed"))
        if committed >= 3:
            v2_accept3 += 1
        elif committed == 2:
            v2_accept2 += 1
        else:
            v2_accept1 += 1
        v2_max_margin0 = max(v2_max_margin0, abs(float(m.group("margin0"))))
        v2_max_margin1 = max(v2_max_margin1, abs(float(m.group("margin1"))))
    for m in GEN_STEP_PROFILE_RE.finditer(log_text):
        gen_steps.append({
            "cycle": int(m.group("cycle")),
            "pos": int(m.group("pos")),
            "mtp": int(m.group("mtp")),
            "accepted": int(m.group("accepted")),
            "eval_ms": float(m.group("eval_ms")),
            "generated_before": int(m.group("generated_before")),
        })

    def gen_step_metric(skip_cycles: int = 0, skip_tokens: int = 0) -> dict[str, Any]:
        accepted = 0
        eval_ms = 0.0
        cycles = 0
        for step in gen_steps:
            if step["cycle"] < skip_cycles:
                continue
            if step["generated_before"] < skip_tokens:
                continue
            accepted += step["accepted"]
            eval_ms += step["eval_ms"]
            cycles += 1
        return {
            "cycles": cycles,
            "accepted": accepted,
            "eval_ms": eval_ms,
            "tps": (accepted * 1000.0 / eval_ms) if eval_ms > 0.0 else 0.0,
            "ms_per_token": (eval_ms / accepted) if accepted > 0 else 0.0,
        }

    gen_step_profile = {
        "steps": len(gen_steps),
        "total": gen_step_metric(),
        "skip_cycles_1": gen_step_metric(skip_cycles=1),
        "skip_cycles_4": gen_step_metric(skip_cycles=4),
        "skip_tokens_32": gen_step_metric(skip_tokens=32),
    }
    return {
        "checks": checks,
        "decision_bad": decision_bad,
        "logit_bad": logit_bad,
        "max_abs": max_abs,
        "max_rms": max_rms,
        "verify_v2_checks": v2_checks,
        "verify_v2_failed": v2_failed,
        "verify_v2_accept1": v2_accept1,
        "verify_v2_accept2": v2_accept2,
        "verify_v2_accept3": v2_accept3,
        "verify_v2_max_margin0": v2_max_margin0,
        "verify_v2_max_margin1": v2_max_margin1,
        "gen_step_profile": gen_step_profile,
    }


def profile_timing_summary(*, wall_ms: float, shadow: dict[str, Any]) -> dict[str, Any]:
    """Return stable timing fields for reports without hiding raw parsed data."""
    gen_step = shadow.get("gen_step_profile", {})
    total = gen_step.get("total", {})
    skip1 = gen_step.get("skip_cycles_1", {})
    skip4 = gen_step.get("skip_cycles_4", {})
    skip32 = gen_step.get("skip_tokens_32", {})
    accepted = int(total.get("accepted", 0) or 0)
    eval_ms = float(total.get("eval_ms", 0.0) or 0.0)
    return {
        "wall_ms": wall_ms,
        "gen_step_samples": int(gen_step.get("steps", 0) or 0),
        "decode_eval_ms": eval_ms,
        "decode_accepted_tokens": accepted,
        "decode_tps": float(total.get("tps", 0.0) or 0.0),
        "decode_ms_per_token": float(total.get("ms_per_token", 0.0) or 0.0),
        "steady_state": {
            "all": total,
            "skip_first_cycle": skip1,
            "skip_first_4_cycles": skip4,
            "skip_first_32_tokens": skip32,
        },
    }


def parse_weight_server_log(log_text: str) -> dict[str, Any]:
    plans: dict[str, Any] = {}
    vmm_plans: dict[str, Any] = {}
    uploads: dict[str, Any] = {}
    memory: dict[str, float] = {}
    ready: dict[str, Any] = {}
    vmm_support: dict[str, Any] = {}
    backend: dict[str, Any] = {}
    broker_served = 0
    broker_requests = 0
    for m in WEIGHT_PLAN_RE.finditer(log_text):
        plans[m.group("model")] = {
            "model_gib": float(m.group("model_gib")),
            "raw_tensor_ranges_gib": float(m.group("raw_gib")),
            "ranges": int(m.group("ranges")),
        }
    mem = WEIGHT_MEMORY_RE.search(log_text)
    if mem:
        memory = {
            "need_gib": float(mem.group("need_gib")),
            "reserve_gib": float(mem.group("reserve_gib")),
            "free_gib": float(mem.group("free_gib")),
            "total_gib": float(mem.group("total_gib")),
        }
    support_m = WEIGHT_VMM_SUPPORT_RE.search(log_text)
    if support_m:
        vmm_support = {
            "vmm": int(support_m.group("vmm")),
            "posix_fd": int(support_m.group("posix_fd")),
            "uva": int(support_m.group("uva")),
            "granularity_min": int(support_m.group("gran_min")),
            "granularity_recommended": int(support_m.group("gran_rec")),
        }
    for m in WEIGHT_VMM_PLAN_RE.finditer(log_text):
        vmm_plans[m.group("model")] = {
            "logical_gib": float(m.group("logical_gib")),
            "allocated_gib": float(m.group("allocated_gib")),
            "granularity": int(m.group("granularity")),
        }
    backend_m = WEIGHT_BACKEND_RE.search(log_text)
    if backend_m:
        backend = {
            "name": backend_m.group("backend"),
            "logical_gib": float(backend_m.group("logical_gib")),
            "allocated_gib": float(backend_m.group("allocated_gib")),
        }
    for m in WEIGHT_UPLOAD_RE.finditer(log_text):
        uploads[m.group("model")] = {
            "uploaded_gib": float(m.group("gib")),
            "ranges": int(m.group("ranges")),
        }
    ready_m = WEIGHT_READY_RE.search(log_text)
    if ready_m:
        ready = {
            "manifest": ready_m.group("manifest"),
            "ranges": int(ready_m.group("ranges")),
        }
    lock_m = WEIGHT_LOCK_RE.search(log_text)
    broker_m = WEIGHT_BROKER_RE.search(log_text)
    for served_m in WEIGHT_BROKER_SERVED_RE.finditer(log_text):
        broker_served += 1
        broker_requests = max(broker_requests, int(served_m.group("requests")))
    shutdown_m = None
    for shutdown_m in WEIGHT_SHUTDOWN_RE.finditer(log_text):
        pass
    if shutdown_m and shutdown_m.group("requests") is not None:
        broker_requests = max(broker_requests, int(shutdown_m.group("requests")))
    return {
        "plans": plans,
        "vmm_plans": vmm_plans,
        "memory": memory,
        "uploads": uploads,
        "ready": ready,
        "vmm_support": vmm_support,
        "backend": backend,
        "broker_path": broker_m.group("path") if broker_m else "",
        "broker_served": broker_served,
        "broker_requests": broker_requests,
        "lock_path": lock_m.group("path") if lock_m else "",
        "shutdown": shutdown_m is not None,
        "lock_busy": "another weight server owns lock" in log_text,
        "refused_upload": "refusing upload" in log_text,
    }


def weight_server_expected_models(scope: str) -> list[str]:
    if scope == "both":
        return ["base", "mtp"]
    return [scope]


def weight_server_validation(
    state: WeightServerState,
    *,
    scope: str,
    backend: str,
    preflight_required: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    if not state.enabled:
        return {"passed": True, "enabled": False, "checks": checks, "reasons": reasons}

    checks["ready"] = state.ready
    checks["manifest_path"] = bool(state.manifest_path)
    checks["scope_matches"] = state.scope == scope
    checks["backend_matches"] = state.backend == backend
    if preflight_required:
        checks["preflight_rc_zero"] = state.preflight_rc == 0
        checks["preflight_not_refused"] = not state.preflight_telemetry.get("refused_upload", False)
        if backend == "vmm":
            support = state.preflight_telemetry.get("vmm_support", {})
            checks["preflight_vmm_supported"] = (
                support.get("vmm") == 1 and support.get("posix_fd") == 1 and support.get("uva") == 1
            )

    if state.owned:
        telemetry = state.telemetry
        cmd = state.cmd
        uploads = telemetry.get("uploads", {})
        ready = telemetry.get("ready", {})
        checks["cleanup_terminated"] = state.cleanup == "terminated"
        checks["shutdown_observed"] = telemetry.get("shutdown", False)
        checks["ready_telemetry"] = bool(ready.get("manifest")) and int(ready.get("ranges", 0)) > 0
        checks["parent_guard"] = "--exit-on-parent-pid" in cmd
        checks["lock_not_busy"] = not telemetry.get("lock_busy", False)
        if "--no-lock" not in cmd:
            checks["lock_recorded"] = bool(telemetry.get("lock_path"))
        if backend == "vmm":
            checks["vmm_backend_telemetry"] = telemetry.get("backend", {}).get("name") == "vmm"
            checks["vmm_broker_listening"] = bool(telemetry.get("broker_path"))
            checks["vmm_broker_requests"] = int(telemetry.get("broker_requests", 0)) > 0
        for model_id in weight_server_expected_models(scope):
            checks[f"uploaded_{model_id}"] = model_id in uploads and int(uploads[model_id].get("ranges", 0)) > 0
            if backend == "vmm":
                checks[f"vmm_plan_{model_id}"] = model_id in telemetry.get("vmm_plans", {})
    else:
        manifest = state.telemetry.get("manifest", {})
        owner = manifest.get("owner", {})
        ranges = manifest.get("ranges", {})
        checks["external_manifest"] = state.cleanup == "external"
        checks["external_owner"] = bool(owner.get("pid")) and owner.get("scope") == scope
        checks["external_backend"] = manifest.get("backend", "ipc") == backend
        if backend == "vmm":
            checks["external_broker"] = bool(manifest.get("broker_path"))
        for model_id in weight_server_expected_models(scope):
            checks[f"manifest_ranges_{model_id}"] = int(ranges.get(model_id, 0)) > 0

    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    return {
        "passed": not reasons,
        "enabled": True,
        "preflight_required": preflight_required,
        "checks": checks,
        "reasons": reasons,
    }


def profile_from_json(raw: dict[str, Any]) -> EngineProfile:
    return EngineProfile(
        name=str(raw["name"]),
        env={str(k): str(v) for k, v in raw.get("env", {}).items()},
        args=[str(v) for v in raw.get("args", [])],
        backend=str(raw.get("backend", "cuda")),
        use_mtp=parse_bool(raw.get("use_mtp", False)),
        mtp_draft=int(raw.get("mtp_draft", 2)),
        baseline=parse_bool(raw.get("baseline", False)),
    )


def prompt_from_json(i: int, raw: Any) -> PromptCase:
    if isinstance(raw, str):
        return PromptCase(f"p{i:02d}", raw)
    return PromptCase(str(raw.get("id", f"p{i:02d}")), str(raw["prompt"]))


def contract_from_json(raw: dict[str, Any]) -> Contract:
    return Contract(
        name=str(raw.get("name", f"{raw['baseline']}_vs_{raw['candidate']}")),
        baseline=str(raw["baseline"]),
        candidate=str(raw["candidate"]),
        kind=str(raw.get("kind", "exact_bytes")),
    )


def default_mtp_profiles() -> list[EngineProfile]:
    return [
        EngineProfile("nomtp", {}, use_mtp=False, baseline=True),
        EngineProfile("mtp-fast", dict(FAST_MTP_ENV), use_mtp=True),
        EngineProfile(
            "mtp-fast-shadow-b",
            {
                **FAST_MTP_ENV,
                "DS4_CUDA_MTP_SHADOW_B_N2_Q8": "1",
                "DS4_MTP_TIMING": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-v2-shadow",
            {
                **FAST_MTP_ENV,
                "DS4_MTP_VERIFY_V2_SHADOW": "1",
                "DS4_MTP_TIMING": "1",
            },
            use_mtp=True,
            mtp_draft=3,
        ),
        EngineProfile(
            "mtp-v2-active",
            {
                **FAST_MTP_ENV,
                "DS4_MTP_VERIFY_V2": "1",
                "DS4_MTP_TIMING": "1",
            },
            use_mtp=True,
            mtp_draft=3,
        ),
        EngineProfile(
            "mtp-batch-first",
            {
                **FAST_MTP_ENV,
                "DS4_MTP_BATCH_FIRST": "1",
                "DS4_MTP_TIMING": "1",
            },
            use_mtp=True,
            mtp_draft=3,
        ),
        EngineProfile(
            "mtp-no-opt-output",
            {
                "DS4_CUDA_MTP_TOP2": "1",
                "DS4_CUDA_MTP_VERIFY_TOP2": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-opt-output",
            {
                **FAST_MTP_ENV,
                "DS4_CUDA_MTP_VERIFY_OPT_OUTPUT": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-no-verify-top2",
            {
                "DS4_CUDA_MTP_TOP2": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-rollback-structural",
            {
                **FAST_MTP_ENV,
                "DS4_CUDA_NO_BATCH_Q8_PAIR": "1",
                "DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6_N2": "1",
                "DS4_CUDA_NO_DECODE_Q8_PAIR": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-exact-replay",
            {
                **FAST_MTP_ENV,
                "DS4_MTP_EXACT_REPLAY": "1",
            },
            use_mtp=True,
        ),
        EngineProfile(
            "mtp-strict",
            {
                **FAST_MTP_ENV,
                "DS4_MTP_STRICT": "1",
            },
            use_mtp=True,
        ),
    ]


def default_engine_profiles() -> list[EngineProfile]:
    return [EngineProfile("baseline", {}, use_mtp=False, baseline=True)]


def default_contracts(profiles: list[EngineProfile], suite: str) -> list[Contract]:
    baseline = next((p.name for p in profiles if p.baseline), profiles[0].name)
    contracts: list[Contract] = []
    seen: set[tuple[str, str]] = set()
    for p in profiles:
        if p.name == baseline:
            continue
        pairs = [(baseline, p.name)]
        if suite == "mtp_speculative" and p.name != "mtp-fast" and any(x.name == "mtp-fast" for x in profiles):
            pairs.append(("mtp-fast", p.name))
        for base_name, cand_name in pairs:
            key = (base_name, cand_name)
            if key in seen:
                continue
            seen.add(key)
            contracts.append(Contract(f"{base_name}_vs_{cand_name}", base_name, cand_name))
    return contracts


def load_plan(path: Path) -> tuple[list[EngineProfile], list[PromptCase], list[Contract], str | None]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    profiles = [profile_from_json(p) for p in raw.get("profiles", [])]
    prompts = [prompt_from_json(i, p) for i, p in enumerate(raw.get("prompts", []))]
    contracts = [contract_from_json(c) for c in raw.get("contracts", [])]
    suite = raw.get("suite")
    return profiles, prompts, contracts, suite


def build_command(
    *,
    bin_path: str,
    base_model: str,
    mtp_model: str | None,
    profile: EngineProfile,
    prompt: str,
    tokens: int,
    temperature: float,
    nothink: bool,
) -> list[str]:
    cmd = [
        bin_path,
        f"--{profile.backend}",
        "-m",
        base_model,
        "--temp",
        f"{temperature:g}",
        "-n",
        str(tokens),
    ]
    if nothink:
        cmd.append("--nothink")
    if profile.use_mtp:
        if not mtp_model:
            raise ValueError(f"profile {profile.name} requires an MTP model")
        cmd[4:4] = ["--mtp", mtp_model, "--mtp-draft", str(profile.mtp_draft)]
    cmd.extend(profile.args)
    cmd.extend(["-p", prompt])
    return cmd


def run_profile(
    *,
    bin_path: str,
    base_model: str,
    mtp_model: str | None,
    suite: str,
    prompt_case: PromptCase,
    tokens: int,
    temperature: float,
    nothink: bool,
    profile: EngineProfile,
    work_dir: Path,
    weight_ipc_manifest: str | None,
    weight_ipc_scope: str,
) -> RunResult:
    safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_case.id)
    safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile.name)
    out_path = work_dir / f"{safe_prompt}_{safe_profile}.out"
    log_path = work_dir / f"{safe_prompt}_{safe_profile}.log"
    cmd = build_command(
        bin_path=bin_path,
        base_model=base_model,
        mtp_model=mtp_model,
        profile=profile,
        prompt=prompt_case.prompt,
        tokens=tokens,
        temperature=temperature,
        nothink=nothink,
    )
    env = os.environ.copy()
    env.update(profile.env)
    if weight_ipc_manifest and profile.backend == "cuda":
        env.setdefault("DS4_CUDA_WEIGHT_IPC_MANIFEST", weight_ipc_manifest)
        env.setdefault("DS4_CUDA_WEIGHT_IPC_SCOPE", weight_ipc_scope)
    t0 = time.monotonic()
    with out_path.open("wb") as out_f, log_path.open("wb") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=out_f, stderr=log_f)
    wall_ms = (time.monotonic() - t0) * 1000.0
    result = RunResult(
        prompt_id=prompt_case.id,
        profile=profile.name,
        suite=suite,
        rc=proc.returncode,
        cmd=cmd,
        out_path=str(out_path),
        log_path=str(log_path),
        wall_ms=wall_ms,
    )
    if out_path.exists():
        result.stdout_sha256, result.stdout_bytes = sha256_file(out_path)
    if log_path.exists():
        result.shadow = parse_shadow(log_path.read_text(errors="replace"))
    result.timing = profile_timing_summary(wall_ms=result.wall_ms, shadow=result.shadow)
    return result


def compare_exact_bytes(
    contract: Contract,
    prompt_id: str,
    baseline: RunResult,
    candidate: RunResult,
) -> ComparisonResult:
    if baseline.rc != 0 or candidate.rc != 0:
        return ComparisonResult(
            prompt_id=prompt_id,
            contract=contract.name,
            baseline=contract.baseline,
            candidate=contract.candidate,
            kind=contract.kind,
            passed=False,
            reason=f"nonzero rc baseline={baseline.rc} candidate={candidate.rc}",
        )
    base_bytes = Path(baseline.out_path).read_bytes()
    cand_bytes = Path(candidate.out_path).read_bytes()
    diff = first_diff(base_bytes, cand_bytes)
    return ComparisonResult(
        prompt_id=prompt_id,
        contract=contract.name,
        baseline=contract.baseline,
        candidate=contract.candidate,
        kind=contract.kind,
        passed=diff is None,
        first_diff=diff,
        baseline_snippet=snippet(base_bytes, diff),
        candidate_snippet=snippet(cand_bytes, diff),
    )


def evaluate_contract(
    contract: Contract,
    prompt_id: str,
    results: dict[tuple[str, str], RunResult],
) -> ComparisonResult:
    baseline = results.get((prompt_id, contract.baseline))
    candidate = results.get((prompt_id, contract.candidate))
    if baseline is None or candidate is None:
        return ComparisonResult(
            prompt_id=prompt_id,
            contract=contract.name,
            baseline=contract.baseline,
            candidate=contract.candidate,
            kind=contract.kind,
            passed=False,
            reason="missing profile result",
        )
    if contract.kind != "exact_bytes":
        return ComparisonResult(
            prompt_id=prompt_id,
            contract=contract.name,
            baseline=contract.baseline,
            candidate=contract.candidate,
            kind=contract.kind,
            passed=False,
            reason=f"unsupported contract kind: {contract.kind}",
        )
    return compare_exact_bytes(contract, prompt_id, baseline, candidate)


def dataclass_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [dataclass_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dataclass_dict(v) for k, v in obj.items()}
    return obj


def validate_weight_manifest(path: Path, base_model: str, mtp_model: str | None, scope: str = "both") -> dict[str, Any]:
    expected: dict[str, int] = {}
    if scope in {"both", "base"}:
        expected["base"] = os.path.getsize(base_model)
    if scope in {"both", "mtp"} and mtp_model:
        expected["mtp"] = os.path.getsize(mtp_model)
    if not expected:
        raise ValueError(f"weight manifest scope {scope!r} has no expected models")
    seen: dict[str, list[tuple[int, int]]] = {k: [] for k in expected}
    saw_header = False
    backend = ""
    broker_path = ""
    owner: dict[str, Any] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"DS4_WEIGHT_SERVER_IPC_V1", "DS4_WEIGHTD_IPC_V1"}:
            saw_header = True
            backend = "ipc"
            continue
        if line == "DS4_WEIGHT_SERVER_VMM_V1":
            saw_header = True
            backend = "vmm"
            continue
        parts = line.split()
        if parts and parts[0] == "broker":
            if backend != "vmm" or len(parts) != 2:
                raise ValueError(f"invalid weight manifest broker line {lineno}: {raw}")
            broker_path = parts[1]
            continue
        if parts and parts[0] == "owner":
            if len(parts) != 5:
                raise ValueError(f"invalid weight manifest owner line {lineno}: {raw}")
            try:
                owner_pid = int(parts[1])
                owner_device = int(parts[2])
            except ValueError as e:
                raise ValueError(f"invalid weight manifest owner pid/device on line {lineno}") from e
            owner_scope = parts[3]
            if owner_scope != scope:
                raise ValueError(f"weight manifest owner scope mismatch: manifest={owner_scope} requested={scope}")
            if owner_pid <= 1:
                raise ValueError(f"invalid weight manifest owner pid on line {lineno}")
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError as e:
                raise ValueError(f"weight manifest owner pid is not running: {owner_pid}") from e
            except PermissionError:
                pass
            owner = {
                "pid": owner_pid,
                "device": owner_device,
                "scope": owner_scope,
                "lock_file": "" if parts[4] == "-" else parts[4],
            }
            continue
        if backend == "vmm":
            if len(parts) != 7 or parts[0] != "alloc":
                raise ValueError(f"invalid VMM weight manifest line {lineno}: {raw}")
            _, alloc_id_s, model_id, model_size_s, off_s, bytes_s, alloc_bytes_s = parts
            if model_id not in expected:
                raise ValueError(f"unexpected weight manifest model id {model_id!r} on line {lineno}")
            try:
                int(alloc_id_s)
                model_size = int(model_size_s)
                off = int(off_s)
                n = int(bytes_s)
                alloc_n = int(alloc_bytes_s)
            except ValueError as e:
                raise ValueError(f"non-integer VMM weight manifest allocation on line {lineno}") from e
            if model_size != expected[model_id]:
                raise ValueError(
                    f"weight manifest size mismatch for {model_id}: manifest={model_size} local={expected[model_id]}"
                )
            if off < 0 or n <= 0 or off > model_size or n > model_size - off or alloc_n < n:
                raise ValueError(f"invalid VMM weight manifest allocation bounds on line {lineno}")
            seen[model_id].append((off, off + n))
            continue
        if len(parts) != 6 or parts[0] != "range":
            raise ValueError(f"invalid weight manifest line {lineno}: {raw}")
        _, model_id, model_size_s, off_s, bytes_s, handle_hex = parts
        if model_id not in expected:
            raise ValueError(f"unexpected weight manifest model id {model_id!r} on line {lineno}")
        try:
            model_size = int(model_size_s)
            off = int(off_s)
            n = int(bytes_s)
        except ValueError as e:
            raise ValueError(f"non-integer weight manifest range on line {lineno}") from e
        if model_size != expected[model_id]:
            raise ValueError(
                f"weight manifest size mismatch for {model_id}: manifest={model_size} local={expected[model_id]}"
            )
        if off < 0 or n <= 0 or off > model_size or n > model_size - off:
            raise ValueError(f"invalid weight manifest range bounds on line {lineno}")
        if len(handle_hex) != 128 or re.search(r"[^0-9A-Fa-f]", handle_hex):
            raise ValueError(f"invalid CUDA IPC handle encoding on line {lineno}")
        seen[model_id].append((off, off + n))
    if not saw_header:
        raise ValueError(f"weight manifest missing DS4_WEIGHT_SERVER_IPC_V1 header: {path}")
    if backend == "vmm" and not broker_path:
        raise ValueError(f"VMM weight manifest missing broker record: {path}")
    for model_id, ranges in seen.items():
        if not ranges:
            raise ValueError(f"weight manifest has no ranges for {model_id}")
        ranges.sort()
        prev_end = 0
        for off, end in ranges:
            if off < prev_end:
                raise ValueError(f"weight manifest has overlapping ranges for {model_id}")
            prev_end = end
    if not owner:
        raise ValueError(f"weight manifest missing live owner record: {path}")
    return {
        "backend": backend or "ipc",
        "owner": owner,
        "broker_path": broker_path,
        "ranges": {k: len(v) for k, v in seen.items()},
    }


def print_run_line(result: RunResult) -> None:
    shadow = result.shadow
    shadow_text = ""
    if shadow.get("checks"):
        shadow_text = (
            f" shadow checks={shadow['checks']} decision_bad={shadow['decision_bad']} "
            f"logit_bad={shadow['logit_bad']} max_abs={shadow['max_abs']:.6g} "
            f"rms={shadow['max_rms']:.6g}"
        )
    if shadow.get("verify_v2_checks"):
        shadow_text += (
            f" v2 checks={shadow['verify_v2_checks']} failed={shadow['verify_v2_failed']} "
            f"a1={shadow['verify_v2_accept1']} a2={shadow['verify_v2_accept2']} "
            f"a3={shadow['verify_v2_accept3']}"
        )
    gen_step = shadow.get("gen_step_profile", {})
    if gen_step.get("steps"):
        total = gen_step["total"]
        skip1 = gen_step["skip_cycles_1"]
        skip32 = gen_step["skip_tokens_32"]
        shadow_text += (
            f" steady total={total['tps']:.2f}t/s"
            f" skip1={skip1['tps']:.2f}t/s"
            f" skip32tok={skip32['tps']:.2f}t/s"
        )
    status = "OK" if result.rc == 0 else f"FAILED rc={result.rc}"
    print(
        f"{result.profile:28s} {status:12s} sha={result.stdout_sha256 or '-'} "
        f"bytes={result.stdout_bytes} wall={result.wall_ms:.0f}ms "
        f"out={result.out_path} log={result.log_path}{shadow_text}"
    )


def print_comparison(comp: ComparisonResult) -> None:
    status = "PASS" if comp.passed else "FAIL"
    diff = "MATCH" if comp.first_diff is None else f"DIFF@{comp.first_diff}"
    reason = f" reason={comp.reason}" if comp.reason else ""
    print(
        f"  [{status}] {comp.contract}: {comp.baseline} vs {comp.candidate} "
        f"{comp.kind} {diff}{reason}"
    )
    if not comp.passed and comp.first_diff is not None:
        print(f"    {comp.baseline}: {comp.baseline_snippet}")
        print(f"    {comp.candidate}: {comp.candidate_snippet}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="mtp_speculative",
                    choices=["argmax_generation", "mtp_speculative"])
    ap.add_argument("--plan", type=Path, help="JSON proof plan with profiles, prompts, and contracts.")
    ap.add_argument("--bin", default=os.environ.get("DS4_PROOF_BIN", "./ds4"))
    ap.add_argument("--base", default=os.environ.get("DS4_PROOF_BASE"))
    ap.add_argument("--mtp", default=os.environ.get("DS4_PROOF_MTP"))
    ap.add_argument("--budget", choices=sorted(BUDGET_PRESETS),
                    help="Named proof budget. --tokens overrides the preset token count.")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--work-dir", default="/tmp/ds4_proof")
    ap.add_argument("--json-report", type=Path)
    ap.add_argument("--weight-ipc-manifest",
                    help="CUDA shared weight manifest from ds4_weight_server.")
    ap.add_argument("--weight-server-scope", choices=["both", "base", "mtp"], default="both",
                    help="Shared weight scope for validation, server startup, and CUDA import. "
                    "Defaults to base when no MTP model is configured.")
    ap.add_argument("--start-weight-server", action="store_true",
                    help="Start ds4_weight_server for this proof run and clean it up on exit.")
    ap.add_argument("--weight-server-bin", default=os.environ.get("DS4_WEIGHT_SERVER_BIN", "./ds4_weight_server"))
    ap.add_argument("--weight-server-backend", choices=["ipc", "vmm"], default="ipc",
                    help="ds4_weight_server sharing backend. Default: ipc.")
    ap.add_argument("--weight-server-manifest", type=Path,
                    help="Manifest path for --start-weight-server. Defaults under --work-dir.")
    ap.add_argument("--weight-server-ready-timeout", type=float, default=1800.0,
                    help="Seconds to wait for ds4_weight_server to create a manifest.")
    ap.add_argument("--weight-server-preflight-timeout", type=float, default=300.0,
                    help="Seconds to wait for ds4_weight_server --dry-run preflight.")
    ap.add_argument("--no-weight-server-preflight", action="store_true",
                    help="Skip the short-lived ds4_weight_server --dry-run before startup.")
    ap.add_argument("--weight-server-reserve-gb", type=int, default=32,
                    help="Free CUDA memory reserve passed to ds4_weight_server.")
    ap.add_argument("--weight-server-span-mb", type=int,
                    help="Raw span size passed to ds4_weight_server.")
    ap.add_argument("--weight-server-copy-chunk-mb", type=int,
                    help="Pinned upload chunk size passed to ds4_weight_server.")
    ap.add_argument("--weight-server-arg", action="append", default=[],
                    help="Extra argument passed to ds4_weight_server. May be repeated.")
    ap.add_argument("--prompt", action="append", dest="prompts")
    ap.add_argument("--prompt-file", action="append", type=Path, dest="prompt_files")
    ap.add_argument("--only", action="append", dest="only_profiles",
                    help="Profile name to run. May be repeated. Defaults to all profiles.")
    ap.add_argument("--custom", action="append", default=[], type=parse_env_assignments,
                    metavar="NAME:KEY=VALUE,...",
                    help="Compatibility alias: add an MTP candidate inheriting mtp-fast flags.")
    ap.add_argument("--custom-profile", action="append", default=[], type=parse_env_assignments,
                    metavar="NAME:KEY=VALUE,...",
                    help="Add a non-MTP engine profile with environment flags.")
    ap.add_argument("--no-nothink", action="store_true",
                    help="Do not add --nothink to generated ds4 commands.")
    args = ap.parse_args(argv)

    plan_profiles: list[EngineProfile] = []
    plan_prompts: list[PromptCase] = []
    plan_contracts: list[Contract] = []
    if args.plan:
        plan_profiles, plan_prompts, plan_contracts, plan_suite = load_plan(args.plan)
        if plan_suite:
            args.suite = plan_suite

    if not args.base:
        ap.error("provide --base or DS4_PROOF_BASE")
    if args.suite == "mtp_speculative" and not args.mtp:
        ap.error("suite mtp_speculative requires --mtp or DS4_PROOF_MTP")
    if args.weight_ipc_manifest and args.start_weight_server:
        ap.error("use either --weight-ipc-manifest or --start-weight-server, not both")
    weight_server_scope = args.weight_server_scope
    if (args.weight_ipc_manifest or args.start_weight_server) and not args.mtp:
        if args.weight_server_scope == "mtp":
            ap.error("--weight-server-scope mtp requires --mtp or DS4_PROOF_MTP")
        if args.weight_server_scope == "both":
            weight_server_scope = "base"

    profiles = plan_profiles or (
        default_mtp_profiles() if args.suite == "mtp_speculative" else default_engine_profiles()
    )
    profiles.extend(
        EngineProfile(name, {**FAST_MTP_ENV, **env}, use_mtp=True)
        for name, env in args.custom
    )
    profiles.extend(
        EngineProfile(name, env, use_mtp=False)
        for name, env in args.custom_profile
    )

    budget_preset = BUDGET_PRESETS.get(args.budget) if args.budget else None
    tokens = args.tokens
    token_source = "cli"
    if tokens is None:
        if budget_preset:
            tokens = budget_preset.tokens
            token_source = f"budget:{budget_preset.name}"
        else:
            tokens = 96
            token_source = "default"

    prompt_source = "builtin-default"
    if args.prompts or args.prompt_files:
        prompts: list[PromptCase] = []
        for i, prompt in enumerate(args.prompts or []):
            prompts.append(PromptCase(f"p{i:02d}", prompt))
        base_i = len(prompts)
        for j, path in enumerate(args.prompt_files or []):
            prompts.append(PromptCase(path.stem or f"p{base_i + j:02d}", path.read_text()))
        prompt_source = "cli"
    elif plan_prompts:
        prompts = plan_prompts
        prompt_source = "plan"
    else:
        default_count = budget_preset.prompt_count if budget_preset else len(DEFAULT_PROMPTS)
        prompts = [
            PromptCase(f"p{i:02d}", p)
            for i, p in enumerate(DEFAULT_PROOF_PROMPTS[:default_count])
        ]
        if budget_preset:
            prompt_source = f"budget:{budget_preset.name}"

    selected = set(args.only_profiles or [p.name for p in profiles])
    known = {p.name for p in profiles}
    unknown = selected - known
    if unknown:
        ap.error(f"unknown profiles: {', '.join(sorted(unknown))}")
    profiles = [p for p in profiles if p.name in selected]
    if not profiles:
        ap.error("no profiles selected")

    contracts = [
        c for c in (plan_contracts or default_contracts(profiles, args.suite))
        if c.baseline in selected and c.candidate in selected
    ]

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    weight_ipc_manifest = args.weight_ipc_manifest
    weight_server: WeightServer | None = None
    weight_server_state = WeightServerState(
        enabled=bool(args.weight_ipc_manifest or args.start_weight_server),
        owned=False,
        scope=weight_server_scope,
        backend=args.weight_server_backend,
        manifest_path=args.weight_ipc_manifest or "",
    )
    if args.weight_ipc_manifest:
        manifest_info = validate_weight_manifest(Path(args.weight_ipc_manifest), args.base, args.mtp, weight_server_scope)
        weight_server_state.ready = True
        weight_server_state.cleanup = "external"
        weight_server_state.telemetry = {"manifest": manifest_info}
    elif args.start_weight_server:
        manifest_path = args.weight_server_manifest or (work_dir / "ds4_weight_server.ipc")
        log_path = work_dir / "ds4_weight_server.log"
        weight_server = WeightServer(
            bin_path=args.weight_server_bin,
            base_model=args.base,
            mtp_model=args.mtp,
            manifest_path=manifest_path,
            log_path=log_path,
            ready_timeout_s=args.weight_server_ready_timeout,
            reserve_gb=args.weight_server_reserve_gb,
            span_mb=args.weight_server_span_mb,
            copy_chunk_mb=args.weight_server_copy_chunk_mb,
            extra_args=args.weight_server_arg,
            preflight_timeout_s=args.weight_server_preflight_timeout,
            scope=weight_server_scope,
            backend=args.weight_server_backend,
        )
        if not args.no_weight_server_preflight:
            print(f"preflighting ds4_weight_server log={weight_server.state.preflight_log_path}")
            try:
                weight_server.preflight()
            except Exception as e:
                print(f"ds4_weight_server preflight FAILED {e}")
                return 1
        print(f"starting ds4_weight_server manifest={manifest_path} log={log_path}")
        try:
            weight_server_state = weight_server.start()
        except Exception as e:
            weight_server.stop()
            print(f"ds4_weight_server FAILED {e}")
            return 1
        weight_ipc_manifest = str(manifest_path)
        print(
            f"ds4_weight_server ready pid={weight_server_state.pid} "
            f"wall={weight_server_state.start_wall_ms:.0f}ms"
        )

    print(
        f"ds4-proof suite={args.suite} profiles={len(profiles)} prompts={len(prompts)} "
        f"tokens={tokens} budget={args.budget or '-'}"
    )
    results: dict[tuple[str, str], RunResult] = {}
    comparisons: list[ComparisonResult] = []
    failures = 0

    try:
        for prompt_case in prompts:
            print(f"\n=== {prompt_case.id} {prompt_case.prompt[:80]!r}")
            for profile in profiles:
                if weight_server and not weight_server.is_running():
                    print(f"{profile.name:28s} FAILED ds4_weight_server exited before profile run")
                    failures += 1
                    continue
                try:
                    result = run_profile(
                        bin_path=args.bin,
                        base_model=args.base,
                        mtp_model=args.mtp,
                        suite=args.suite,
                        prompt_case=prompt_case,
                        tokens=tokens,
                        temperature=args.temperature,
                        nothink=not args.no_nothink,
                        profile=profile,
                        work_dir=work_dir,
                        weight_ipc_manifest=weight_ipc_manifest,
                        weight_ipc_scope=weight_server_scope,
                    )
                except ValueError as e:
                    print(f"{profile.name:28s} FAILED {e}")
                    failures += 1
                    continue
                results[(prompt_case.id, profile.name)] = result
                print_run_line(result)
                if result.rc != 0:
                    failures += 1
                if weight_server and not weight_server.is_running():
                    print(f"{profile.name:28s} FAILED ds4_weight_server exited during profile run")
                    failures += 1

            for contract in contracts:
                comp = evaluate_contract(contract, prompt_case.id, results)
                comparisons.append(comp)
                print_comparison(comp)
                if not comp.passed:
                    failures += 1

            for profile in profiles:
                result = results.get((prompt_case.id, profile.name))
                if not result:
                    continue
                shadow = result.shadow
                if shadow.get("decision_bad") or shadow.get("logit_bad"):
                    failures += 1
                if shadow.get("verify_v2_failed"):
                    failures += 1
    finally:
        if weight_server:
            weight_server.stop()
            weight_server_state = weight_server.state
            print(f"ds4_weight_server cleanup={weight_server_state.cleanup}")

    weight_server_verdict = weight_server_validation(
        weight_server_state,
        scope=weight_server_scope,
        backend=args.weight_server_backend,
        preflight_required=args.start_weight_server and not args.no_weight_server_preflight,
    )
    if weight_server_verdict["enabled"] and not weight_server_verdict["passed"]:
        print(
            "ds4_weight_server validation FAILED "
            f"reasons={','.join(weight_server_verdict['reasons'])}"
        )
        failures += 1

    report = {
        "schema": "ds4-proof-report-v1",
        "suite": args.suite,
        "tokens": tokens,
        "budget": {
            "name": args.budget or "",
            "tokens": tokens,
            "token_source": token_source,
            "prompt_source": prompt_source,
            "prompt_count": len(prompts),
            "preset_prompt_count": budget_preset.prompt_count if budget_preset else 0,
            "description": budget_preset.description if budget_preset else "",
        },
        "temperature": args.temperature,
        "work_dir": str(work_dir),
        "weight_ipc_manifest": weight_ipc_manifest,
        "weight_ipc_scope": weight_server_scope,
        "weight_server_backend": args.weight_server_backend,
        "weight_server": dataclass_dict(weight_server_state),
        "weight_server_validation": weight_server_verdict,
        "profiles": [dataclass_dict(p) for p in profiles],
        "prompts": [dataclass_dict(p) for p in prompts],
        "contracts": [dataclass_dict(c) for c in contracts],
        "results": [dataclass_dict(r) for r in results.values()],
        "comparisons": [dataclass_dict(c) for c in comparisons],
        "failures": failures,
    }
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\njson_report={args.json_report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
