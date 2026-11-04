"""JJ DAI v0.1 — M2 acceptance tests (adversarial, like M1's).

Proves the ТЗ criteria:
  (a) any record replays from the log;
  (b) tampering with ANY record breaks the chain (also: reorder, truncate);
  (c) the anchored head catches a full-prefix rewrite that a plain chain
      check alone cannot (operator rewrote the WHOLE log consistently).

Run:  python3 test_witness.py
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile

from witness import WitnessLog, ChainError
from anchor import LocalAnchor, AnchorScheduler, verify_against_anchor

WORK = tempfile.mkdtemp(prefix="jjdai_m2_")
LOG = os.path.join(WORK, "witness_chain.jsonl")
ANCHORS = os.path.join(WORK, "anchors.jsonl")
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---- build a realistic mixed chain: INFER ⊕ SANDBOX, anchored every 3 ----
log = WitnessLog(LOG)
sched = AnchorScheduler(log, LocalAnchor(ANCHORS), every_n=3)
for i in range(7):
    if i % 2 == 0:
        log.emit_infer({"node": "n1", "model": "v4-flash-q2", "seed": 1, "i": i})
    else:
        log.emit_sandbox({"canary_id": f"c{i}", "passed": True,
                          "out_hash": "ab" * 32, "i": i})
    sched.notify_append()

n, head = WitnessLog.verify_chain(LOG)
check("chain verifies intact", n == 7, f"{n} records, head={head[:12]}…")

anchor_rec = LocalAnchor(ANCHORS).latest()
check("periodic anchor emitted", anchor_rec is not None and
      anchor_rec["seq"] == 5, f"anchored seq={anchor_rec['seq']}")

# ---- (a) replay any record ----
r = WitnessLog.replay(LOG, 3)
check("replay returns verified record", r["body"]["kind"] == "SANDBOX"
      and r["body"]["payload"]["canary_id"] == "c3")

# ---- (b1) tamper with a payload field -> chain breaks ----
def mutate(src, dst, fn):
    recs = [json.loads(l) for l in open(src) if l.strip()]
    fn(recs)
    with open(dst, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

T1 = os.path.join(WORK, "tampered.jsonl")
mutate(LOG, T1, lambda rs: rs[2]["body"]["payload"].update({"model": "EVIL"}))
try:
    WitnessLog.verify_chain(T1)
    check("tampered record detected", False, "chain verified but should not")
except ChainError as e:
    check("tampered record detected", True, str(e))

# ---- (b2) reorder two records -> chain breaks ----
T2 = os.path.join(WORK, "reordered.jsonl")
mutate(LOG, T2, lambda rs: rs.__setitem__(slice(1, 3), [rs[2], rs[1]]))
try:
    WitnessLog.verify_chain(T2)
    check("reorder detected", False)
except ChainError as e:
    check("reorder detected", True, str(e))

# ---- (b3) truncate the tail: plain chain check CANNOT see it… ----
T3 = os.path.join(WORK, "truncated.jsonl")
mutate(LOG, T3, lambda rs: rs.__delitem__(slice(4, None)))
n3, _ = WitnessLog.verify_chain(T3)  # verifies! (prefix is honest)
check("truncation invisible to bare chain (expected)", n3 == 4,
      "why the anchor exists")
# …but the ANCHOR catches it: anchored seq 5 is gone.
check("truncation caught by anchor",
      not verify_against_anchor(T3, anchor_rec))

# ---- (c) full consistent rewrite -> bare chain passes, anchor catches ----
T4 = os.path.join(WORK, "rewritten.jsonl")
lie = WitnessLog(T4)
for i in range(7):
    lie.append("INFER", {"history": "rewritten", "i": i})
n4, _ = WitnessLog.verify_chain(T4)
check("consistent rewrite passes bare chain (expected)", n4 == 7,
      "internally consistent lie")
check("consistent rewrite caught by anchor",
      not verify_against_anchor(T4, anchor_rec))

# ---- honest log still matches its anchor ----
check("honest log matches anchor", verify_against_anchor(LOG, anchor_rec))

failed = sum(1 for _, ok in results if not ok)
print(f"\nsummary: {len(results) - failed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
