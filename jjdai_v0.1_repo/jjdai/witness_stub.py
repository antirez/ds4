"""JJ DAI v0.1 — witness adapter (M2).

M1 shipped a flat JSONL stub here. M2 replaces the internals with the
hash-chained WitnessLog (witness.py) while keeping the emit_sandbox() surface,
so verifier.py is unchanged. Also exposes emit_infer() and the scheduler hook.
"""
from __future__ import annotations
import os

from witness import WitnessLog
from proto import Verdict

WITNESS_LOG = os.environ.get("JJDAI_WITNESS_LOG", "witness_chain.jsonl")

_log: WitnessLog | None = None
_scheduler = None   # optional AnchorScheduler, set via set_scheduler()


def _get() -> WitnessLog:
    global _log
    if _log is None or _log.path != WITNESS_LOG:
        _log = WitnessLog(WITNESS_LOG)
    return _log


def set_scheduler(sched) -> None:
    global _scheduler
    _scheduler = sched


def emit_sandbox(verdict: Verdict) -> str:
    h = _get().append("SANDBOX", verdict)
    if _scheduler:
        _scheduler.notify_append()
    return h


def emit_infer(provenance: dict) -> str:
    h = _get().append("INFER", provenance)
    if _scheduler:
        _scheduler.notify_append()
    return h
