"""JJ DAI v0.1 — M5 acceptance tests: RAG-as-tool + Plane H grounding gate.

ТЗ criterion: an answer on a grounded topic WITHOUT valid memory citations
fails verification. Proven against three engine behaviors (diligent, lazy,
fabricating) plus a store-tamper attack on the Merkle commitment.

Run:  python3 test_grounding.py
"""
from __future__ import annotations
import os
import tempfile

from rag_store import RagStore
from canary_store import EvalRagOverlapError
from harness import Harness, GroundedMockEngine, LazyMockEngine, FabricatingMockEngine
from grounding import verify_grounding
from witness import WitnessLog

WORK = tempfile.mkdtemp(prefix="jjdai_m5_")
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---- store with legal memory ---------------------------------------------------
rag = RagStore(os.path.join(WORK, "rag.db"))
NS = "legal"
rag.add(NS, "sha-v4.3", "Реєстрація змін до статуту здійснюється протягом 30 днів.")
rag.add(NS, "sha-v4.3", "Міноритарний акціонер має право вето за переліком із 13 пунктів.")
rag.add(NS, "aoa-v3.7", "Кворум загальних зборів становить 51 відсоток голосів.")
root0 = rag.merkle_root(NS)

WLOG = os.path.join(WORK, "grounding_chain.jsonl")
wl = WitnessLog(WLOG)
REQ = {"messages": [{"role": "user",
                     "content": "Який кворум зборів і права міноритарного акціонера?"}]}

# (a) diligent engine: retrieves, cites served chunks -> gate passes,
#     proofs verify against the namespace root
resp = Harness(GroundedMockEngine(), rag, NS).run(REQ)
ok, why = verify_grounding(resp, rag, NS, "legal", witness=wl)
check("(a) grounded answer with valid citations passes", ok, why)
check("(a2) citations carry real Merkle proofs",
      all(RagStore.verify_proof(rag.prove(NS, c["chunk_id"]))
          for c in resp["metadata"]["citations"]))

# (b) lazy engine: grounded topic, no retrieve, no citations -> FAIL (ТЗ)
resp = Harness(LazyMockEngine(), rag, NS).run(REQ)
ok, why = verify_grounding(resp, rag, NS, "legal", witness=wl)
check("(b) grounded topic without citations FAILS", not ok, why)

# (c) fabricating engine: retrieved, but cites a chunk never served -> FAIL
resp = Harness(FabricatingMockEngine(), rag, NS).run(REQ)
ok, why = verify_grounding(resp, rag, NS, "legal", witness=wl)
check("(c) fabricated citation FAILS", not ok, why)

# (d) store tamper: mutate a chunk AFTER the answer cited it -> hash/Merkle break
resp = Harness(GroundedMockEngine(), rag, NS).run(REQ)
cited = resp["metadata"]["citations"][0]["chunk_id"]
rag.db.execute("UPDATE chunks SET text=?, hash=hash WHERE id=?",
               ("Кворум становить 10 відсотків (підроблено).", cited))
rag.db.commit()
# text changed but stored hash left stale -> recompute root over stored hashes
# stays the same; the REAL check is hash-of-text. Emulate an auditor recompute:
from proto import sha256_text
row = rag.get(cited)
check("(d) tampered memory detected (stored hash no longer matches text)",
      sha256_text(row["text"]) != row["hash"],
      "auditor re-hash catches silent edit")

# and a full-tamper (text+hash rewritten) shifts the Merkle root -> old proofs die
rag.db.execute("UPDATE chunks SET hash=? WHERE id=?",
               (sha256_text(row["text"]), cited))
rag.db.commit()
check("(d2) consistent tamper shifts the Merkle root (old commitments die)",
      rag.merkle_root(NS) != root0,
      f"{root0[:10]}… -> {rag.merkle_root(NS)[:10]}…")

# (e) gate is scoped: non-grounded topic passes without citations
resp = Harness(LazyMockEngine(), rag, NS).run(
    {"messages": [{"role": "user", "content": "Расскажи анекдот"}]})
ok, why = verify_grounding(resp, rag, NS, "general", witness=wl)
check("(e) non-grounded topic passes without citations", ok, why)

# (f) eval/RAG separation guard applies to the RAG db location too
try:
    RagStore(os.path.join(WORK, "eval", "rag.db"),
             eval_dir=os.path.join(WORK, "eval"))
    check("(f) RAG-inside-eval rejected", False)
except EvalRagOverlapError:
    check("(f) RAG-inside-eval rejected (no open-book)", True)

# (g) every gate decision is witnessed; chain intact
n, _ = WitnessLog.verify_chain(WLOG)
kinds = [r["body"]["payload"]["passed"] for r in WitnessLog.read_all(WLOG)
         if r["body"]["kind"] == "GROUNDING"]
check("(g) witness records all gate decisions, chain intact",
      n >= 4 and kinds.count(False) == 2, f"{n} records, outcomes={kinds}")

failed = sum(1 for _, ok in results if not ok)
print(f"\nsummary: {len(results) - failed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
