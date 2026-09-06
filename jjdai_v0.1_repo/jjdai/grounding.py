"""JJ DAI v0.1 — M5 Plane H: the grounding gate.

Wraps (fails) any answer on a GROUNDED topic whose memory citations do not
verify. "Verify" is cryptographic, not stylistic — every citation must:

  1. exist: cited chunk_id is present in the RAG namespace;
  2. bind:  the hash the harness recorded when SERVING the chunk matches the
            hash stored for it (a citation to a never-served chunk has no hash);
  3. include: the chunk's Merkle inclusion proof verifies against the CURRENT
            namespace root — the memory the answer leans on is exactly the
            memory the namespace commits to.

Non-grounded topics pass without citations: the gate is scoped, not global.
Every gate decision is a GROUNDING record in the witness chain (observes §12.1;
here the verifier, not the witness, is what gates).
"""
from __future__ import annotations

from rag_store import RagStore
from witness import WitnessLog

GROUNDED_TOPICS = {"legal", "energy"}   # policy: topics that require memory


def verify_grounding(response: dict, store: RagStore, ns: str, topic: str,
                     witness: WitnessLog | None = None,
                     grounded_topics: set[str] = GROUNDED_TOPICS
                     ) -> tuple[bool, str]:
    def _emit(passed: bool, reason: str):
        if witness:
            witness.append("GROUNDING", {
                "topic": topic, "passed": passed, "reason": reason,
                "citations": response.get("metadata", {}).get("citations", [])})
        return passed, reason

    if topic not in grounded_topics:
        return _emit(True, f"topic {topic!r} not grounded; gate not applicable")

    cites = response.get("metadata", {}).get("citations", [])
    if not cites:
        return _emit(False, "grounded topic answered with NO memory citations")

    root = store.merkle_root(ns)
    for c in cites:
        cid = c["chunk_id"]
        chunk = store.get(cid)
        if chunk is None:
            return _emit(False, f"citation to nonexistent chunk {cid}")
        if c["hash"] is None:
            return _emit(False, f"chunk {cid} cited but never served this turn")
        if c["hash"] != chunk["hash"]:
            return _emit(False, f"chunk {cid} hash mismatch (memory tampered "
                                "between serving and verification)")
        proof = store.prove(ns, cid)
        if proof is None or not RagStore.verify_proof(proof) \
                or proof["root"] != root:
            return _emit(False, f"chunk {cid} Merkle inclusion failed against "
                                "namespace root")
    return _emit(True, f"{len(cites)} citation(s) verified against root "
                       f"{root[:12]}…")
