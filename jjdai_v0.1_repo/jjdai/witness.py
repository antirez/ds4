"""JJ DAI v0.1 — M2 witness log.

Append-only, hash-chained record log (whitepaper §13 / §7.4). Replaces the M1
flat stub. Properties:

  * every record carries prev_hash -> tampering with ANY record breaks the
    chain from that point forward (tamper-EVIDENT, not tamper-proof);
  * two record kinds, interleaved in ONE chain per node:
      INFER   — provenance attestation of the config that produced an answer
                (trust via replication);
      SANDBOX — replayable execution proof (trust via replay);
  * the log head can be anchored externally (see anchor.py) so the state at
    time T cannot be silently rewritten later;
  * the witness OBSERVES and never gates (§12.1): appends are fire-and-forget
    for the hot path; verification is offline.

Format: JSONL. Each line: {"h": sha256(canonical(body)), "body": {...}}.
body = {seq, ts, kind, prev_hash, payload}. Canonicalization = sorted-keys
compact JSON of body — stable across writers.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, is_dataclass

from proto import sha256_text

GENESIS = "0" * 64


def _canon(body: dict) -> str:
    return json.dumps(body, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _rec_hash(body: dict) -> str:
    return sha256_text(_canon(body))


class ChainError(RuntimeError):
    """Raised by verify_chain on any break: tampered, reordered, or truncated-in-middle."""


class WitnessLog:
    def __init__(self, path: str):
        self.path = path
        self._seq, self._head = self._load_tip()

    # ---------- internal ----------
    def _load_tip(self) -> tuple[int, str]:
        if not os.path.exists(self.path):
            return 0, GENESIS
        seq, head = 0, GENESIS
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seq = rec["body"]["seq"] + 1
                head = rec["h"]
        return seq, head

    # ---------- write path ----------
    def append(self, kind: str, payload) -> str:
        """Append a record; returns its hash (the new head)."""
        if is_dataclass(payload):
            payload = asdict(payload)
        body = {
            "seq": self._seq,
            "ts": time.time(),
            "kind": kind,                 # "INFER" | "SANDBOX" | "ANCHOR"
            "prev_hash": self._head,
            "payload": payload,
        }
        h = _rec_hash(body)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"h": h, "body": body}, ensure_ascii=False,
                               sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self._head = h
        return h

    def emit_infer(self, provenance: dict) -> str:
        return self.append("INFER", provenance)

    def emit_sandbox(self, verdict) -> str:
        return self.append("SANDBOX", verdict)

    def head(self) -> str:
        return self._head

    # ---------- read/verify path (offline) ----------
    @staticmethod
    def read_all(path: str) -> list[dict]:
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def verify_chain(path: str) -> tuple[int, str]:
        """Re-hash every record and walk the chain.

        Returns (n_records, head_hash) if intact; raises ChainError on the
        first break. This is what an auditor runs against a copied log.
        """
        prev = GENESIS
        n = 0
        for rec in WitnessLog.read_all(path):
            body = rec["body"]
            if body["seq"] != n:
                raise ChainError(f"seq gap at record {n}: got {body['seq']}")
            if body["prev_hash"] != prev:
                raise ChainError(f"chain break at seq {n}: prev_hash mismatch")
            h = _rec_hash(body)
            if h != rec["h"]:
                raise ChainError(f"tampered record at seq {n}: hash mismatch")
            prev = h
            n += 1
        return n, prev

    @staticmethod
    def replay(path: str, seq: int) -> dict:
        """Return the verified record at seq (verifies chain up to it)."""
        prev = GENESIS
        for rec in WitnessLog.read_all(path):
            body = rec["body"]
            if body["prev_hash"] != prev or _rec_hash(body) != rec["h"]:
                raise ChainError(f"chain invalid at seq {body['seq']}")
            prev = rec["h"]
            if body["seq"] == seq:
                return rec
        raise KeyError(f"seq {seq} not found")
