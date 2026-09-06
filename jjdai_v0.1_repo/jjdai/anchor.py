"""JJ DAI v0.1 — M2 external anchor.

Anchoring makes the witness log's state at time T provable to OUTSIDERS: the
head hash is committed to something the node cannot rewrite. Per the v0.1
decision this is a PUBLIC TIMESTAMP.

Design: one port, pluggable backends (the hot path never depends on network):

  * LocalAnchor       — dev/test: writes head to a separate append-only file.
                        Proves the MECHANISM (anchor -> verify -> catch rewrite)
                        but NOT external trust: same operator controls both files.
  * OpenTimestampsAnchor — production seam: calls the `ots` client
                        (`pip install opentimestamps-client`); commits the head
                        into Bitcoin via public calendar servers, free.
                        Verification: `ots verify <file>.ots`.
  * Rfc3161Anchor     — production seam: POSTs a TimeStampQuery to a TSA
                        (e.g. freetsa.org); returns a signed TimeStampToken.

Anchoring is PERIODIC (every N records or M seconds), not per-record: one
anchor covers the whole prefix, because the head hash transitively commits to
every record before it.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time

from proto import sha256_text


class AnchorError(RuntimeError):
    pass


class LocalAnchor:
    """Dev/test backend. External-trust value: NONE (same-operator files).
    Mechanism value: full — verify() catches any rewrite of the anchored prefix."""

    def __init__(self, path: str):
        self.path = path

    def commit(self, head_hash: str, seq: int) -> dict:
        rec = {"ts": time.time(), "seq": seq, "head": head_hash,
               "backend": "local"}
        rec["proof"] = sha256_text(json.dumps(rec, sort_keys=True))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def latest(self) -> dict | None:
        if not os.path.exists(self.path):
            return None
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
        return last


class OpenTimestampsAnchor:
    """Production backend via the `ots` CLI. Requires network + installed client.

    commit(): writes head to a small file and runs `ots stamp` -> .ots proof.
    Later `ots upgrade` / `ots verify` complete and check the Bitcoin proof.
    """

    def __init__(self, workdir: str):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)

    def available(self) -> bool:
        return shutil.which("ots") is not None

    def commit(self, head_hash: str, seq: int) -> dict:
        if not self.available():
            raise AnchorError("`ots` client not installed "
                              "(pip install opentimestamps-client)")
        fname = os.path.join(self.workdir, f"head_{seq}.txt")
        with open(fname, "w") as f:
            f.write(head_hash + "\n")
        subprocess.run(["ots", "stamp", fname], check=True,
                       capture_output=True, timeout=60)
        return {"ts": time.time(), "seq": seq, "head": head_hash,
                "backend": "opentimestamps", "proof_file": fname + ".ots"}


class AnchorScheduler:
    """Anchors the witness head every `every_n` records (checked on notify)."""

    def __init__(self, witness_log, backend, every_n: int = 100):
        self.log = witness_log
        self.backend = backend
        self.every_n = every_n
        self._since = 0

    def notify_append(self) -> dict | None:
        self._since += 1
        if self._since >= self.every_n:
            self._since = 0
            return self.backend.commit(self.log.head(), self.log._seq - 1)
        return None


def verify_against_anchor(log_path: str, anchor_rec: dict) -> bool:
    """Auditor check: walk the chain up to the anchored seq and compare heads.

    True  -> the log prefix matches what was anchored at time T.
    False -> the prefix was rewritten after anchoring (caught).
    """
    from witness import WitnessLog, ChainError, GENESIS, _rec_hash
    prev = GENESIS
    try:
        for rec in WitnessLog.read_all(log_path):
            body = rec["body"]
            if body["prev_hash"] != prev or _rec_hash(body) != rec["h"]:
                return False
            prev = rec["h"]
            if body["seq"] == anchor_rec["seq"]:
                return prev == anchor_rec["head"]
    except (ChainError, KeyError, json.JSONDecodeError):
        return False
    return False  # anchored seq missing => truncated log
