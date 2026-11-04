"""JJ DAI v0.1 — M1 demo. Proves the acceptance criteria:

  (a) a good artifact passes the canary (green),
  (b) a broken ("кривой") artifact reddens the canary,
  (c) the eval store is structurally separated from the RAG store.

Run:  python3 demo.py
"""
from __future__ import annotations
import os
import stat
import tempfile

import sandbox
from proto import Canary
from canary_store import CanaryStore, EvalRagOverlapError
from verifier import verify

WORK = tempfile.mkdtemp(prefix="jjdai_m1_")

# --- two candidate artifacts: one correct, one wrong --------------------------
GOOD = os.path.join(WORK, "good.py")
BAD = os.path.join(WORK, "bad.py")

with open(GOOD, "w") as f:
    f.write("import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a + b)\n")
with open(BAD, "w") as f:
    f.write("import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a * b)   # wrong op\n")
for p in (GOOD, BAD):
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)

# --- build a canary: its expected answer comes from a trusted reference run ---
STDIN = "2 3\n"
ref = sandbox.run(["python3", GOOD], stdin=STDIN, min_tier="weak")
canary = Canary(id="add-2-3", stdin=STDIN, expected_out_hash=ref.out_hash)

# --- (c) eval / RAG separation is a hard guard --------------------------------
eval_dir = os.path.join(WORK, "eval_store")
rag_dir = os.path.join(WORK, "rag_store")
store = CanaryStore(eval_dir=eval_dir, rag_dir=rag_dir)
store.add(canary)
print(f"[c] eval store {eval_dir!r} separated from RAG store {rag_dir!r}: OK")

try:
    CanaryStore(eval_dir=os.path.join(rag_dir, "canaries"), rag_dir=rag_dir)
    print("[c] FAIL: overlap was not caught")
except EvalRagOverlapError:
    print("[c] overlapping eval-inside-RAG correctly REJECTED (no open-book)")

# --- (a) good candidate -------------------------------------------------------
vg = verify(["python3", GOOD], canary, min_tier="weak")  # dev box
print(f"\n[a] good.py -> passed={vg.passed}  (isolation: {vg.trace.isolation})")

# --- (b) broken candidate -----------------------------------------------------
vb = verify(["python3", BAD], canary, min_tier="weak")   # dev box
print(f"[b] bad.py  -> passed={vb.passed}  "
      f"expected={vb.expected_out_hash[:12]}…  got={vb.got_out_hash[:12]}…")

print("\nResult:", "OK — loop works"
      if (vg.passed and not vb.passed) else "UNEXPECTED")
print("witness log:", os.path.abspath(
    os.environ.get("JJDAI_WITNESS_LOG", "witness_sandbox.jsonl")))
