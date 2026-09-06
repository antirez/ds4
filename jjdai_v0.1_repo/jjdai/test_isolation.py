"""JJ DAI v0.1 — M1 isolation escape tests (adversarial).

A harness is only as honest as its hardest canary. These tests try to BREAK the
sandbox and assert it contains each attack. Tests that require the STRONG tier
(network egress, host-fs read, fork-bomb via cgroup) auto-skip on a weak host and
say so — skipping is reported, never silently passed.

Run:  python3 test_isolation.py
"""
from __future__ import annotations
import os
import tempfile

import sandbox
from sandbox import SandboxTooWeak, detect_tier

WORK = tempfile.mkdtemp(prefix="jjdai_iso_")
TIER = detect_tier()
results = []


def _art(name, body):
    p = os.path.join(WORK, name)
    open(p, "w").write(body)
    return ["python3", p]


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, why):
    results.append((name, None, why))
    print(f"  [SKIP] {name} — {why}")


print(f"host isolation tier detected: {TIER!r}\n")

# 1. fail-closed: asking for strong on a weak host must RAISE, not run.
if TIER == "weak":
    try:
        sandbox.run(["python3", "-c", "print(1)"], min_tier="strong")
        check("fail-closed refuses weak host", False, "did NOT raise")
    except SandboxTooWeak:
        check("fail-closed refuses weak host", True, "raised SandboxTooWeak")
else:
    skip("fail-closed refuses weak host", f"host is {TIER}, not weak")

# 2. memory bomb -> RLIMIT_AS kills it before the sentinel prints.
t = sandbox.run(_art("mem.py",
    "x=bytearray(2_000_000_000)\nprint('ALLOCATED')\n"), min_tier=TIER)
check("memory bomb contained", t.exit_code != 0 and t.out_hash ==
      sandbox.sha256(b""), f"exit={t.exit_code}")

# 3. wall-clock timeout -> flagged and process killed.
t = sandbox.run(_art("slow.py", "import time\ntime.sleep(30)\nprint('DONE')\n"),
                limits={"wall_seconds": 2}, min_tier=TIER)
check("wall-clock timeout enforced", t.timed_out, f"wall={t.wall_ms}ms")

# 4. output bomb -> capped + truncated flag, parent not OOM'd.
t = sandbox.run(_art("outbomb.py",
    "import sys\n"
    "while True: sys.stdout.write('A'*65536)\n"),
    limits={"out_cap_bytes": 256*1024}, min_tier=TIER)
check("output bomb contained", t.truncated, "truncated + killed")

# 5. file-size bomb -> RLIMIT_FSIZE stops the write (no WROTE sentinel).
t = sandbox.run(_art("fbomb.py",
    "open('big.bin','wb').write(b'x'*100_000_000)\nprint('WROTE')\n"),
    min_tier=TIER)
check("file-size bomb contained", t.exit_code != 0, f"exit={t.exit_code}")

# 6. determinism: same artifact + input -> identical out_hash (canary precondition).
a = sandbox.run(_art("det.py", "print(2+2)\n"), min_tier=TIER)
b = sandbox.run(["python3", os.path.join(WORK, "det.py")], min_tier=TIER)
check("deterministic out_hash", a.out_hash == b.out_hash, a.out_hash[:12] + "…")

# 7. network egress (STRONG only).
if TIER == "strong":
    t = sandbox.run(_art("net.py",
        "import socket\n"
        "socket.create_connection(('1.1.1.1',53),timeout=3)\nprint('NET_OK')\n"),
        min_tier="strong")
    check("network egress blocked", t.exit_code != 0)
else:
    skip("network egress blocked", f"needs strong tier (have {TIER})")

# 8. host filesystem read (STRONG only).
if TIER == "strong":
    t = sandbox.run(_art("host.py",
        "print(open('/etc/hostname').read())\n"), min_tier="strong")
    check("host filesystem hidden", t.exit_code != 0)
else:
    skip("host filesystem hidden", f"needs strong tier (have {TIER})")

# 9. fork bomb (STRONG only; needs pids cgroup).
if TIER == "strong":
    t = sandbox.run(_art("fork.py",
        "import os\nwhile True: os.fork()\n"),
        limits={"wall_seconds": 3}, min_tier="strong")
    check("fork bomb contained", t.timed_out or t.exit_code != 0)
else:
    skip("fork bomb contained", f"needs strong tier / pids cgroup (have {TIER})")

# ---- summary ----
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
print(f"\nsummary: {passed} passed, {failed} failed, {skipped} skipped (tier={TIER})")
print("note: SKIPs require a STRONG host (Linux+bubblewrap+libseccomp or macOS).")
