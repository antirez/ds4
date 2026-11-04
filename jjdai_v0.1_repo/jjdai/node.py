"""JJ DAI v0.1 — M3 node daemon (folds in M0).

A node = Engine (the "muscle") + NodeDaemon (the JJ DAI wrapper). The daemon:

  * exposes a /v1/messages-SHAPED handle(request) -> response (Anthropic wire
    format), transport-agnostic: in-process for tests, an HTTP server is a thin
    wrapper around handle() on a live node;
  * injects a ProvenanceRecord into response metadata — the M0 acceptance;
  * emits an INFER witness record into the node's own hash chain (M2).

ENGINE SEAM
-----------
`Engine` is the narrow-waist boundary. In production it is DwarfStar (or any
runtime) behind /v1/messages. In this container there is no live engine, so
MockSpecialistEngine stands in: deterministic, honest about being a mock.
Everything above the seam (daemon, router, registry, witness) is the real logic.
"""
from __future__ import annotations
import time
from typing import Protocol

from proto import ProvenanceRecord
from witness import WitnessLog


class Engine(Protocol):
    model_id: str
    quant_bits: str
    base_hash: str

    def generate(self, messages: list[dict], system: str = "") -> str: ...


class MockSpecialistEngine:
    """Stands in for DwarfStar behind the seam. Deterministic by construction."""

    def __init__(self, specialty: str, model_id: str, quant_bits: str,
                 base_hash: str):
        self.specialty = specialty
        self.model_id = model_id
        self.quant_bits = quant_bits
        self.base_hash = base_hash

    def generate(self, messages: list[dict], system: str = "") -> str:
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, list):  # content blocks
            last = " ".join(b.get("text", "") for b in last
                            if isinstance(b, dict))
        return (f"[{self.specialty}-specialist/{self.model_id}] "
                f"answer to: {last[:80]}")


class NodeDaemon:
    def __init__(self, node_id: str, engine: Engine, witness_path: str,
                 adapter_set: list[str] | None = None,
                 route_policy_ver: str = "policy-v0", seed: int = 1):
        self.node_id = node_id
        self.engine = engine
        self.witness = WitnessLog(witness_path)
        self.adapter_set = adapter_set or []
        self.route_policy_ver = route_policy_ver
        self.seed = seed

    def provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            node_id=self.node_id,
            model_id=self.engine.model_id,
            quant_bits=self.engine.quant_bits,
            base_hash=self.engine.base_hash,
            adapter_set=list(self.adapter_set),
            route_policy_ver=self.route_policy_ver,
            seed=self.seed,
        )

    def handle(self, request: dict) -> dict:
        """Anthropic /v1/messages-shaped request -> response.

        Minimal fields honored: model, system, messages. The response carries
        metadata.provenance (M0) and its hash chain gets an INFER record (M2).
        """
        t0 = time.monotonic()
        text = self.engine.generate(request.get("messages", []),
                                    system=request.get("system", ""))
        prov = self.provenance()
        witness_head = self.witness.emit_infer(prov.dict())
        return {
            "id": f"msg_{self.node_id}_{self.witness._seq}",
            "type": "message",
            "role": "assistant",
            "model": self.engine.model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"latency_ms": int((time.monotonic() - t0) * 1000)},
            "metadata": {
                "provenance": prov.dict(),
                "witness_head": witness_head,   # auditable pointer into the chain
            },
        }
