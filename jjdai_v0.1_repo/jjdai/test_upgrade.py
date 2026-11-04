"""JJ DAI v0.1 — M4 acceptance tests: pull-by-choice upgrade.

ТЗ criterion (the hard one): a POISONED candidate with VALID provenance must
fail local acceptance and never be installed.

The poisoned candidate here is the literal §7.5 Goodhart attack: it hard-codes
the answer to the PUBLIC gauntlet canary (which the attacker can see) and is
wrong everywhere else. Its manifest is honest (hash matches its bytes), it
passes the public gauntlet — and the node's PRIVATE held-out canaries catch it.

Run:  python3 test_upgrade.py
"""
from __future__ import annotations
import os
import tempfile

import sandbox
from proto import Canary
from updater import NodeUpdater, UpgradeRejected, make_manifest
from witness import WitnessLog

WORK = tempfile.mkdtemp(prefix="jjdai_m4_")
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def art(name, body):
    p = os.path.join(WORK, name)
    open(p, "w").write(body)
    return p


# ---- adapters (task: print the SUM of integers on stdin) ----------------------
V1 = art("v1.py",   # current: handles exactly two numbers, fails on three
         "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n")
V2 = art("v2.py",   # good candidate: general sum
         "import sys\nprint(sum(map(int,sys.stdin.read().split())))\n")
V3 = art("v3.py",   # POISONED: hard-codes the public canary, wrong otherwise
         "import sys\nd=sys.stdin.read().split()\n"
         "print(5) if d==['2','3'] else print(eval('*'.join(d)))\n")

# ---- canaries ------------------------------------------------------------------
def canary(cid, stdin, ref_artifact):
    ref = sandbox.run(["python3", ref_artifact], stdin=stdin, min_tier="weak")
    return Canary(id=cid, stdin=stdin, expected_out_hash=ref.out_hash)

# PUBLIC gauntlet canary — the attacker can see this one ("2 3" -> 5).
pub = canary("pub-2-3", "2 3\n", V2)
# PRIVATE held-out canaries — the node made these itself, never published.
priv = [canary("priv-10-7", "10 7\n", V2),      # sum=17 (poisoned prints 70)
        canary("priv-1-2-4", "1 2 4\n", V2)]    # sum=7  (v1 crashes, poisoned prints 8)

# sanity: poisoned really does pass the public canary
t = sandbox.run(["python3", V3], stdin=pub.stdin, min_tier="weak")
check("poisoned candidate passes PUBLIC gauntlet (attack is real)",
      t.out_hash == pub.expected_out_hash)

# ---- node with v1 active -------------------------------------------------------
WLOG = os.path.join(WORK, "upgrade_chain.jsonl")
cur = make_manifest("champ-v1", "code", V1, "n-code", {"pub-2-3": True})
upd = NodeUpdater("n-code", priv, WLOG, current=cur, min_tier="weak")

# (a) tampered package: manifest signed over different bytes -> fail-closed,
#     rejected BEFORE any execution
m_bad = make_manifest("champ-vX", "code", V2, "n-evil", {"pub-2-3": True})
open(V2, "a").write("# tampered after manifest\n")
try:
    upd.try_upgrade(m_bad)
    check("(a) tampered package rejected fail-closed", False)
except UpgradeRejected as e:
    check("(a) tampered package rejected fail-closed", "hash mismatch" in str(e))
# restore v2 and re-manifest honestly
open(V2, "w").write("import sys\nprint(sum(map(int,sys.stdin.read().split())))\n")
m_good = make_manifest("champ-v2", "code", V2, "n-code2", {"pub-2-3": True})

# (b) POISONED candidate with VALID provenance -> rejected by PRIVATE canaries
m_poison = make_manifest("champ-v3-poison", "code", V3, "n-evil",
                         {"pub-2-3": True})   # provenance is genuinely valid!
try:
    upd.try_upgrade(m_poison)
    check("(b) poisoned candidate NOT installed", False, "was accepted!")
except UpgradeRejected as e:
    check("(b) poisoned candidate rejected by private held-out canaries",
          upd.active.champion_id == "champ-v1" and "regressions" in str(e),
          str(e)[:90])

# (c) good candidate: delta computed and shown, accepted in one press
delta = upd.try_upgrade(m_good)
check("(c) good candidate accepted with delta",
      upd.active.champion_id == "champ-v2"
      and delta.improvements == ["priv-1-2-4"] and not delta.regressions,
      delta.summary())

# (d) rollback is one press
back = upd.rollback()
check("(d) one-press rollback to previous", back.champion_id == "champ-v1")

# (e) every decision is in the witness chain, chain intact
n, _ = WitnessLog.verify_chain(WLOG)
kinds = [r["body"]["payload"].get("action") for r in WitnessLog.read_all(WLOG)
         if r["body"]["kind"] == "UPGRADE"]
check("(e) witness records all decisions, chain intact",
      n >= 4 and kinds.count("rejected") == 2 and "accepted" in kinds
      and "rollback" in kinds, f"{n} records, actions={kinds}")

failed = sum(1 for _, ok in results if not ok)
print(f"\nsummary: {len(results) - failed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
