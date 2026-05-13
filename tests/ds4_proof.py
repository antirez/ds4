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

    print(f"ds4-proof suite={args.suite} profiles={len(profiles)} prompts={len(prompts)} tokens={args.tokens}")
    results: dict[tuple[str, str], RunResult] = {}
    comparisons: list[ComparisonResult] = []
    failures = 0

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

    report = {
        "schema": "ds4-proof-report-v1",
        "suite": args.suite,
        "tokens": args.tokens,
        "temperature": args.temperature,
        "work_dir": str(work_dir),
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
