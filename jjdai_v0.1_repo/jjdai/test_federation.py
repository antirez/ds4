"""JJ DAI v0.1 — M3 acceptance tests: two-node federation + router.

ТЗ criteria:
  (a) a legal query routes to the legal node; a code query routes to the code node;
  (b) every response carries a valid ProvenanceRecord + witness pointer (M0);
  (c) champion is PER-TOPIC (personal), not global — the same registry must pick
      DIFFERENT champions for different topics;
  (d) requester-choice override is honored and recorded;
  (e) code-topic scores come from REAL sandbox verdicts (M1 verifier), not fiction;
  (f) each node's witness chain verifies after traffic.

Run:  python3 test_federation.py
"""
from __future__ import annotations
import os
import tempfile

from node import NodeDaemon, MockSpecialistEngine
from registry import ChampionRegistry
from router import Router
from witness import WitnessLog
from verifier import verify
from proto import Canary
import sandbox

WORK = tempfile.mkdtemp(prefix="jjdai_m3_")
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---- two HETEROGENEOUS nodes (different model/quant — federation, not clones) --
n_legal = NodeDaemon(
    "n-legal",
    MockSpecialistEngine("legal", "v4-flash-q2-legal-adapter", "q2", "base-aaa"),
    os.path.join(WORK, "n_legal_chain.jsonl"), adapter_set=["legal-lora-v3"])
n_code = NodeDaemon(
    "n-code",
    MockSpecialistEngine("code", "v4-pro-q4-code-adapter", "q4", "base-bbb"),
    os.path.join(WORK, "n_code_chain.jsonl"), adapter_set=["code-lora-v7"])
nodes = {"n-legal": n_legal, "n-code": n_code}

# ---- registry seeded by REAL execution verdicts for the code topic ------------
reg = ChampionRegistry(os.path.join(WORK, "registry.jsonl"))

good = os.path.join(WORK, "good.py")
open(good, "w").write("import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n")
bad = os.path.join(WORK, "bad.py")
open(bad, "w").write("import sys\na,b=map(int,sys.stdin.read().split())\nprint(a*b)\n")
ref = sandbox.run(["python3", good], stdin="2 3\n", min_tier="weak")
canary = Canary(id="add-2-3", stdin="2 3\n", expected_out_hash=ref.out_hash)

# n-code's artifact passes the canary; n-legal's attempt at code fails it.
v1 = verify(["python3", good], canary, min_tier="weak")
reg.record("n-code", "code", v1.passed, kind="execution")
v2 = verify(["python3", bad], canary, min_tier="weak")
reg.record("n-legal", "code", v2.passed, kind="execution")
check("(e) code scores from real sandbox verdicts",
      v1.passed and not v2.passed, "green feeds champion, red feeds rival")

# legal topic: NL domain -> replication verdicts (honestly labeled as such)
for _ in range(3):
    reg.record("n-legal", "legal", True, kind="replication")
reg.record("n-code", "legal", False, kind="replication")

# ---- router over the federation -----------------------------------------------
router = Router(reg, nodes, default_node="n-code")

# (a) topic routing
resp_l, d_l = router.dispatch(
    {"messages": [{"role": "user",
                   "content": "Составь договор аренды с учётом закона Украины"}]})
check("(a) legal query -> legal node", d_l.node_id == "n-legal",
      f"topic={d_l.topic}, {d_l.reason}")

resp_c, d_c = router.dispatch(
    {"messages": [{"role": "user",
                   "content": "Напиши функцию сортировки на python и найди баг"}]})
check("(a) code query -> code node", d_c.node_id == "n-code",
      f"topic={d_c.topic}, {d_c.reason}")

# (b) provenance + witness pointer in the response
prov = resp_l["metadata"]["provenance"]
check("(b) provenance carried and correct",
      prov["node_id"] == "n-legal" and prov["adapter_set"] == ["legal-lora-v3"]
      and len(resp_l["metadata"]["witness_head"]) == 64)

# (c) PER-TOPIC champion: same registry, different champions per topic
check("(c) personal champion is per-topic, not global",
      reg.champion("legal") == "n-legal" and reg.champion("code") == "n-code",
      f"legal→{reg.champion('legal')}, code→{reg.champion('code')}")

# (d) requester override honored and recorded
resp_o, d_o = router.dispatch(
    {"messages": [{"role": "user", "content": "Напиши функцию сортировки"}]},
    override_node="n-legal")
check("(d) requester-choice override honored",
      d_o.node_id == "n-legal" and d_o.overridden
      and resp_o["metadata"]["routing"]["overridden"] is True)

# unknown topic -> explicit fallback with reason (no silent guess)
_, d_u = router.dispatch(
    {"messages": [{"role": "user", "content": "Расскажи о философии Руми"}]})
check("unknown topic -> explicit fallback", d_u.topic == "general"
      and "fallback" in d_u.reason, d_u.reason)

# (f) each node's witness chain verifies after traffic
for name, daemon in nodes.items():
    n, head = WitnessLog.verify_chain(daemon.witness.path)
    check(f"(f) {name} witness chain intact", n >= 1,
          f"{n} INFER records, head={head[:12]}…")

failed = sum(1 for _, ok in results if not ok)
print(f"\nsummary: {len(results) - failed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
