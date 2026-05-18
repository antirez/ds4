#!/usr/bin/env python3
"""Smoke-test ds4_proof.py weight-server lifecycle validation.

This test is intentionally CUDA-free. It exercises the proof harness contract
with fake model files, a fake engine, and a fake ds4_weight_server process that
writes a structurally valid manifest and lifecycle logs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DS4_PROOF = ROOT / "tests" / "ds4_proof.py"


def write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def write_fake_tools(tmp: Path) -> tuple[Path, Path, Path, Path]:
    base = tmp / "base.gguf"
    mtp = tmp / "mtp.gguf"
    base.write_bytes(b"B")
    mtp.write_bytes(b"M")

    fake_engine = tmp / "fake_engine.py"
    write_executable(
        fake_engine,
        """
        #!/usr/bin/env python3
        import sys
        print("ds4: mtp accept trace path=decode2 start=1 first=10 drafted=2 accepted=3 checkpoint=4 next_top=20 mtp_valid=1 mtp_draft=30 drafts=20,30", file=sys.stderr)
        print("ds4: gen step profile cycle=0 pos=1 mtp=1 accepted=3 eval_ms=30.0 generated_before=0", file=sys.stderr)
        print("ds4: mtp accept trace path=decode2 start=2 first=11 drafted=2 accepted=2 checkpoint=6 next_top=21 mtp_valid=1 mtp_draft=31 drafts=21,31", file=sys.stderr)
        print("ds4: gen step profile cycle=1 pos=2 mtp=1 accepted=2 eval_ms=20.0 generated_before=1", file=sys.stderr)
        print("ds4: mtp accept trace path=first-miss start=4 first=12 drafted=1 accepted=1 checkpoint=7 next_top=22 mtp_valid=0 mtp_draft=-1 drafts=22", file=sys.stderr)
        print("ds4: gen step profile cycle=2 pos=4 mtp=1 accepted=1 eval_ms=10.0 generated_before=32", file=sys.stderr)
        print("fake engine output")
        """,
    )

    fake_weight_server = tmp / "fake_weight_server.py"
    write_executable(
        fake_weight_server,
        """
        #!/usr/bin/env python3
        import argparse
        import os
        import signal
        import sys
        import time

        ap = argparse.ArgumentParser()
        ap.add_argument("--base")
        ap.add_argument("--mtp")
        ap.add_argument("--manifest")
        ap.add_argument("--scope", default="both")
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--exit-on-parent-pid")
        ap.add_argument("--reserve-gb")
        ap.add_argument("--span-mb")
        ap.add_argument("--copy-chunk-mb")
        args, _extra = ap.parse_known_args()

        models = [args.scope] if args.scope != "both" else ["base", "mtp"]
        for model in models:
            print(
                f"ds4_weight_server: {model} plan model=0.00 GiB "
                "raw_tensor_ranges=0.00 GiB ranges=1",
                flush=True,
            )
        print(
            "ds4_weight_server: memory preflight full upload plan need=0.00 GiB "
            "reserve=32.00 GiB free=128.00 GiB total=128.00 GiB",
            flush=True,
        )
        if args.dry_run:
            print(
                "ds4_weight_server: dry-run complete; no allocations or manifest were created",
                flush=True,
            )
            sys.exit(0)

        print("ds4_weight_server: acquired lock /tmp/ds4_weight_server_cuda0.lock", flush=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write("DS4_WEIGHT_SERVER_IPC_V1\\n")
            f.write(f"owner {os.getpid()} 0 {args.scope} /tmp/ds4_weight_server_cuda0.lock\\n")
            if args.scope in ("both", "base"):
                f.write("range base 1 0 1 " + "0" * 128 + "\\n")
            if args.scope in ("both", "mtp"):
                f.write("range mtp 1 0 1 " + "1" * 128 + "\\n")

        for model in models:
            print(f"ds4_weight_server: {model} uploaded 0.00 GiB across 1 ranges", flush=True)
        print(f"ds4_weight_server: ready manifest={args.manifest} ranges={len(models)}", flush=True)

        stopping = False

        def stop(_signum, _frame):
            global stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        while not stopping:
            time.sleep(0.05)
        print("ds4_weight_server: shutting down", flush=True)
        """,
    )
    return base, mtp, fake_engine, fake_weight_server


def run_proof(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    if proc.returncode != 0:
        print(proc.stdout)
        raise AssertionError(f"proof command failed rc={proc.returncode}")
    report_path = Path(cmd[cmd.index("--json-report") + 1])
    return json.loads(report_path.read_text(encoding="utf-8"))


def assert_true(report: dict[str, Any], path: str) -> None:
    cur: Any = report
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    if cur is not True:
        raise AssertionError(f"{path} is not true: {cur!r}")


def assert_equal(report: dict[str, Any], path: str, expected: Any) -> None:
    cur: Any = report
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    if cur != expected:
        raise AssertionError(f"{path} expected {expected!r}: {cur!r}")


def assert_gt(report: dict[str, Any], path: str, minimum: float) -> None:
    cur: Any = report
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    if not cur > minimum:
        raise AssertionError(f"{path} expected > {minimum!r}: {cur!r}")


def run_owned_lifecycle(tmp: Path, base: Path, mtp: Path, fake_engine: Path, fake_weight_server: Path) -> None:
    work = tmp / "owned-work"
    report = run_proof(
        [
            sys.executable,
            str(DS4_PROOF),
            "--bin",
            str(fake_engine),
            "--base",
            str(base),
            "--mtp",
            str(mtp),
            "--tokens",
            "1",
            "--prompt",
            "local",
            "--only",
            "mtp-fast",
            "--start-weight-server",
            "--weight-server-scope",
            "mtp",
            "--weight-server-bin",
            str(fake_weight_server),
            "--weight-server-derive-output-certifier",
            "--weight-server-derive-group-count",
            "8",
            "--weight-server-derive-q8-f16",
            "blk.0.attn_output_a.weight",
            "--weight-server-derive-q8-f32",
            "blk.0.attn_q_b.weight",
            "--weight-server-derive-budget-gb",
            "1",
            "--work-dir",
            str(work),
            "--json-report",
            str(tmp / "owned-report.json"),
        ]
    )
    if report["failures"] != 0:
        raise AssertionError(f"owned proof failures={report['failures']}")
    assert_true(report, "weight_server_validation.passed")
    cmd = report["weight_server"]["cmd"]
    for expected in [
        "--derive-output-certifier",
        "--derive-group-count",
        "8",
        "--derive-q8-f16",
        "blk.0.attn_output_a.weight",
        "--derive-q8-f32",
        "blk.0.attn_q_b.weight",
        "--derive-budget-gb",
        "1",
    ]:
        if expected not in cmd:
            raise AssertionError(f"owned proof did not forward {expected!r}: {cmd}")
    for check in [
        "ready",
        "scope_matches",
        "preflight_rc_zero",
        "preflight_not_refused",
        "cleanup_terminated",
        "shutdown_observed",
        "ready_telemetry",
        "parent_guard",
        "lock_not_busy",
        "lock_recorded",
        "uploaded_mtp",
    ]:
        assert_true(report, f"weight_server_validation.checks.{check}")
    assert_gt(report, "results.0.timing.steady_state.skip_first_32_tokens.tps", 0.0)
    assert_equal(report, "results.0.timing.acceptance.all.alignment", "aligned")
    assert_equal(report, "results.0.timing.acceptance.all.draft_tokens_proposed", 5)
    assert_equal(report, "results.0.timing.acceptance.all.draft_tokens_accepted", 3)
    assert_equal(report, "results.0.timing.acceptance.all.full_accept_cycles", 1)
    assert_equal(report, "results.0.timing.acceptance.all.partial_accept_cycles", 1)
    assert_equal(report, "results.0.timing.acceptance.all.reject_cycles", 1)
    assert_equal(report, "results.0.timing.acceptance.all.by_path.decode2.draft_tokens_proposed", 4)
    assert_equal(report, "results.0.timing.acceptance.all.by_accepted_drafts.draft_accept_0.cycles", 1)


def run_budget_preset(tmp: Path, base: Path, mtp: Path, fake_engine: Path) -> None:
    report = run_proof(
        [
            sys.executable,
            str(DS4_PROOF),
            "--bin",
            str(fake_engine),
            "--base",
            str(base),
            "--mtp",
            str(mtp),
            "--budget",
            "candidate",
            "--only",
            "mtp-fast",
            "--work-dir",
            str(tmp / "budget-work"),
            "--json-report",
            str(tmp / "budget-report.json"),
        ]
    )
    if report["failures"] != 0:
        raise AssertionError(f"budget proof failures={report['failures']}")
    assert_equal(report, "budget.name", "candidate")
    assert_equal(report, "budget.tokens", 512)
    assert_equal(report, "budget.prompt_count", 4)
    assert_equal(report, "tokens", 512)
    assert_equal(report, "results.0.timing.decode_accepted_tokens", 6)
    assert_equal(report, "results.0.shadow.accept_trace.0.path", "decode2")
    assert_equal(report, "results.0.timing.acceptance.all.draft_tokens_accepted", 3)
    assert_gt(report, "results.0.timing.steady_state.skip_first_cycle.tps", 0.0)


def run_external_manifest(tmp: Path, base: Path, mtp: Path, fake_engine: Path) -> None:
    manifest = tmp / "external.ipc"
    manifest.write_text(
        "\n".join(
            [
                "DS4_WEIGHT_SERVER_IPC_V1",
                f"owner {os.getpid()} 0 mtp /tmp/ds4_weight_server_cuda0.lock",
                "range mtp 1 0 1 " + "1" * 128,
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = run_proof(
        [
            sys.executable,
            str(DS4_PROOF),
            "--bin",
            str(fake_engine),
            "--base",
            str(base),
            "--mtp",
            str(mtp),
            "--tokens",
            "1",
            "--prompt",
            "local",
            "--only",
            "mtp-fast",
            "--weight-ipc-manifest",
            str(manifest),
            "--weight-server-scope",
            "mtp",
            "--work-dir",
            str(tmp / "external-work"),
            "--json-report",
            str(tmp / "external-report.json"),
        ]
    )
    if report["failures"] != 0:
        raise AssertionError(f"external proof failures={report['failures']}")
    assert_true(report, "weight_server_validation.passed")
    assert_true(report, "weight_server_validation.checks.external_manifest")
    assert_true(report, "weight_server_validation.checks.external_owner")
    assert_true(report, "weight_server_validation.checks.manifest_ranges_mtp")


def run_base_only_default_scope(tmp: Path, base: Path, fake_engine: Path, fake_weight_server: Path) -> None:
    work = tmp / "base-only-work"
    report = run_proof(
        [
            sys.executable,
            str(DS4_PROOF),
            "--bin",
            str(fake_engine),
            "--suite",
            "argmax_generation",
            "--base",
            str(base),
            "--tokens",
            "1",
            "--prompt",
            "local",
            "--start-weight-server",
            "--weight-server-bin",
            str(fake_weight_server),
            "--work-dir",
            str(work),
            "--json-report",
            str(tmp / "base-only-report.json"),
        ]
    )
    if report["failures"] != 0:
        raise AssertionError(f"base-only proof failures={report['failures']}")
    if report["weight_ipc_scope"] != "base":
        raise AssertionError(f"expected effective base scope, got {report['weight_ipc_scope']!r}")
    assert_true(report, "weight_server_validation.passed")
    assert_true(report, "weight_server_validation.checks.uploaded_base")
    if "uploaded_mtp" in report["weight_server_validation"]["checks"]:
        raise AssertionError("base-only validation unexpectedly required uploaded_mtp")


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    with tempfile.TemporaryDirectory(prefix="ds4-weight-harness-smoke.") as raw_tmp:
        tmp = Path(raw_tmp)
        base, mtp, fake_engine, fake_weight_server = write_fake_tools(tmp)
        run_owned_lifecycle(tmp, base, mtp, fake_engine, fake_weight_server)
        run_budget_preset(tmp, base, mtp, fake_engine)
        run_external_manifest(tmp, base, mtp, fake_engine)
        run_base_only_default_scope(tmp, base, fake_engine, fake_weight_server)
    print("ds4_weight_server_harness_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
