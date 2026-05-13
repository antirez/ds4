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


DEFAULT_PROMPTS = [
    "List 100 prime numbers, comma-separated, just numbers.",
    "Write a concise explanation of how speculative decoding works, then give three caveats.",
]

FAST_MTP_ENV = {
    "DS4_CUDA_MTP_TOP2": "1",
    "DS4_CUDA_MTP_VERIFY_TOP2": "1",
    "DS4_CUDA_MTP_VERIFY_OPT_OUTPUT": "1",
}

SHADOW_RE = re.compile(
    r"mtp shadow b_n2_q8 .*?\sagree=(?P<agree>[01]).*?"
    r"\slogit_agree=(?P<logit_agree>[01]).*?"
    r"logit_max_abs=(?P<max_abs>[-+0-9.eE]+).*?"
    r"logit_rms=(?P<rms>[-+0-9.eE]+)"
)


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
    manifest_path: str = ""
    log_path: str = ""
    cmd: list[str] = field(default_factory=list)
    pid: int | None = None
    preflight_cmd: list[str] = field(default_factory=list)
    preflight_log_path: str = ""
    preflight_rc: int | None = None
    preflight_wall_ms: float = 0.0
    start_wall_ms: float = 0.0
    ready: bool = False
    cleanup: str = "not_started"
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
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_f: Any = None
        self.state = WeightServerState(
            enabled=True,
            owned=True,
            bin_path=bin_path,
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
                validate_weight_manifest(self.manifest_path, self.base_model, self.mtp_model)
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
    for m in SHADOW_RE.finditer(log_text):
        checks += 1
        if m.group("agree") != "1":
            decision_bad += 1
        if m.group("logit_agree") != "1":
            logit_bad += 1
        max_abs = max(max_abs, float(m.group("max_abs")))
        max_rms = max(max_rms, float(m.group("rms")))
    return {
        "checks": checks,
        "decision_bad": decision_bad,
        "logit_bad": logit_bad,
        "max_abs": max_abs,
        "max_rms": max_rms,
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
            "mtp-no-opt-output",
            {
                "DS4_CUDA_MTP_TOP2": "1",
                "DS4_CUDA_MTP_VERIFY_TOP2": "1",
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
    for p in profiles:
        if p.name == baseline:
            continue
        contracts.append(Contract(f"{baseline}_vs_{p.name}", baseline, p.name))
        if suite == "mtp_speculative" and p.name != "mtp-fast" and any(x.name == "mtp-fast" for x in profiles):
            contracts.append(Contract(f"mtp-fast_vs_{p.name}", "mtp-fast", p.name))
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


def validate_weight_manifest(path: Path, base_model: str, mtp_model: str | None) -> None:
    expected = {"base": os.path.getsize(base_model)}
    if mtp_model:
        expected["mtp"] = os.path.getsize(mtp_model)
    seen: dict[str, list[tuple[int, int]]] = {k: [] for k in expected}
    saw_header = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in {"DS4_WEIGHT_SERVER_IPC_V1", "DS4_WEIGHTD_IPC_V1"}:
            saw_header = True
            continue
        parts = line.split()
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
    for model_id, ranges in seen.items():
        if not ranges:
            raise ValueError(f"weight manifest has no ranges for {model_id}")
        ranges.sort()
        prev_end = 0
        for off, end in ranges:
            if off < prev_end:
                raise ValueError(f"weight manifest has overlapping ranges for {model_id}")
            prev_end = end


def print_run_line(result: RunResult) -> None:
    shadow = result.shadow
    shadow_text = ""
    if shadow.get("checks"):
        shadow_text = (
            f" shadow checks={shadow['checks']} decision_bad={shadow['decision_bad']} "
            f"logit_bad={shadow['logit_bad']} max_abs={shadow['max_abs']:.6g} "
            f"rms={shadow['max_rms']:.6g}"
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
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--work-dir", default="/tmp/ds4_proof")
    ap.add_argument("--json-report", type=Path)
    ap.add_argument("--weight-ipc-manifest",
                    help="CUDA shared weight manifest from ds4_weight_server.")
    ap.add_argument("--start-weight-server", action="store_true",
                    help="Start ds4_weight_server for this proof run and clean it up on exit.")
    ap.add_argument("--weight-server-bin", default=os.environ.get("DS4_WEIGHT_SERVER_BIN", "./ds4_weight_server"))
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

    if args.prompts or args.prompt_files:
        prompts: list[PromptCase] = []
        for i, prompt in enumerate(args.prompts or []):
            prompts.append(PromptCase(f"p{i:02d}", prompt))
        base_i = len(prompts)
        for j, path in enumerate(args.prompt_files or []):
            prompts.append(PromptCase(path.stem or f"p{base_i + j:02d}", path.read_text()))
    elif plan_prompts:
        prompts = plan_prompts
    else:
        prompts = [PromptCase(f"p{i:02d}", p) for i, p in enumerate(DEFAULT_PROMPTS)]

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
        manifest_path=args.weight_ipc_manifest or "",
    )
    if args.weight_ipc_manifest:
        validate_weight_manifest(Path(args.weight_ipc_manifest), args.base, args.mtp)
        weight_server_state.ready = True
        weight_server_state.cleanup = "external"
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

    print(f"ds4-proof suite={args.suite} profiles={len(profiles)} prompts={len(prompts)} tokens={args.tokens}")
    results: dict[tuple[str, str], RunResult] = {}
    comparisons: list[ComparisonResult] = []
    failures = 0

    try:
        for prompt_case in prompts:
            print(f"\n=== {prompt_case.id} {prompt_case.prompt[:80]!r}")
            for profile in profiles:
                try:
                    result = run_profile(
                        bin_path=args.bin,
                        base_model=args.base,
                        mtp_model=args.mtp,
                        suite=args.suite,
                        prompt_case=prompt_case,
                        tokens=args.tokens,
                        temperature=args.temperature,
                        nothink=not args.no_nothink,
                        profile=profile,
                        work_dir=work_dir,
                        weight_ipc_manifest=weight_ipc_manifest,
                    )
                except ValueError as e:
                    print(f"{profile.name:28s} FAILED {e}")
                    failures += 1
                    continue
                results[(prompt_case.id, profile.name)] = result
                print_run_line(result)
                if result.rc != 0:
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
    finally:
        if weight_server:
            weight_server.stop()
            weight_server_state = weight_server.state
            print(f"ds4_weight_server cleanup={weight_server_state.cleanup}")

    report = {
        "schema": "ds4-proof-report-v1",
        "suite": args.suite,
        "tokens": args.tokens,
        "temperature": args.temperature,
        "work_dir": str(work_dir),
        "weight_ipc_manifest": weight_ipc_manifest,
        "weight_server": dataclass_dict(weight_server_state),
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
