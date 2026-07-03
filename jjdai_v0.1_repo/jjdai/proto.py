"""JJ DAI v0.1 — M1 shared schemas (proto).

Minimal dataclasses for the sandbox+canary loop. Full WitnessRecord chaining
lands in M2; here we keep a flat sandbox trace + verdict.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json
import time


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8"))


@dataclass
class SandboxTrace:
    """A replayable proof of one execution (see whitepaper §7.4)."""
    env_fp: str          # sha256 of runtime/lib versions + limits + seed
    in_hash: str         # sha256 of (cmd + stdin)
    out_hash: str        # sha256 of stdout bytes
    err_hash: str        # sha256 of stderr bytes
    exit_code: int
    wall_ms: int
    isolation: str       # which isolation backend actually applied
    timed_out: bool = False
    tier: str = "weak"   # achieved isolation tier: strong|medium|weak
    truncated: bool = False  # output exceeded cap and was cut (bomb contained)

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class Canary:
    """An execution canary: a task with a known-correct output hash.

    Lives in the EVAL store, never in a RAG namespace the model reads from.
    """
    id: str
    stdin: str
    expected_out_hash: str
    source: str = "reference"
    rotation_epoch: int = 0


@dataclass
class Verdict:
    canary_id: str
    passed: bool
    expected_out_hash: str
    got_out_hash: str
    trace: SandboxTrace
    ts: float = field(default_factory=time.time)

    def json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

@dataclass
class ProvenanceRecord:
    """Attestation of the config that produced an answer (M0/M3, §7.2 of spec).

    Travels in the response metadata of /v1/messages and is emitted as an
    INFER witness record. Trust model: replication, not replay (whitepaper §7.5).
    """
    node_id: str
    model_id: str
    quant_bits: str
    base_hash: str
    adapter_set: list
    route_policy_ver: str
    seed: int

    def dict(self) -> dict:
        return asdict(self)


@dataclass
class ChampionManifest:
    """A champion offered to nodes for pull-by-choice upgrade (M4).

    v0.1 delivery: ADAPTERS ONLY (decision #2) — instant, DIIP Class 1.
    Provenance travels INSIDE the package: the artifact hash binds the manifest
    to the exact delivered bytes, and the public gauntlet result rides along.
    """
    champion_id: str
    topic: str
    artifact_path: str
    artifact_hash: str          # sha256 of the delivered artifact bytes
    producer_node: str
    gauntlet: dict              # public gauntlet result {canary_id: bool}
    delivery_class: str = "adapter"

    def dict(self) -> dict:
        return asdict(self)
