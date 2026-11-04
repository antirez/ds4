"""JJ DAI v0.1 — M1 execution verifier.

Runs a CANDIDATE artifact (what a node/model produced) against an execution
canary inside the sandbox, and compares the output hash to the canary's known
answer. Verification cost = one sandbox exec, far below production cost — the
cost asymmetry from whitepaper §7.1.

Emits a sandbox witness record (flat append for M1; hash-chained log is M2).
"""
from __future__ import annotations

import sandbox
from proto import Canary, Verdict
from witness_stub import emit_sandbox


def verify(candidate_cmd: list[str], canary: Canary,
           jail: str | None = None, min_tier: str = "strong") -> Verdict:
    trace = sandbox.run(candidate_cmd, stdin=canary.stdin, jail=jail,
                        min_tier=min_tier)
    passed = (trace.out_hash == canary.expected_out_hash) and trace.exit_code == 0
    verdict = Verdict(
        canary_id=canary.id,
        passed=passed,
        expected_out_hash=canary.expected_out_hash,
        got_out_hash=trace.out_hash,
        trace=trace,
    )
    emit_sandbox(verdict)            # witness records, never gates (§12.1)
    return verdict


def gauntlet(candidate_cmd: list[str], canaries: list[Canary],
             min_tier: str = "strong") -> list[Verdict]:
    return [verify(candidate_cmd, c, min_tier=min_tier) for c in canaries]
