#!/usr/bin/env python3
"""Speed driver.  usage: speed.py decode|prefill [runs] [--tag T] [--tokens N]
decode: 400 greedy tokens after a short prompt, `runs` times; prefill: ~5000-token prompt (prefix of $IK_TEXT_DIR/alice.txt,
PREFILL_CHARS characters, default 21000), 1 generated token.
Reads ds4's own timing line ("prefill: X t/s, generation: Y t/s") and prints per-run figures and a median.
Environment: IK_MODEL, IK_TEXT_DIR, IK_WORK (default ./ik-work, stderr of each run goes there), DS4_DIR (default .), DS4_LOCK_FILE."""
import json, os, re, subprocess, sys, statistics
MODEL = os.environ["IK_MODEL"]
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))
os.makedirs(WORK, exist_ok=True)
mode = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "x"
ntok = sys.argv[sys.argv.index("--tokens") + 1] if "--tokens" in sys.argv else ("400" if mode == "decode" else "1")
if mode == "decode": prompt = "The history of the Roman Empire begins with"; ctx = "2048"
else: prompt = open(os.path.join(os.environ["IK_TEXT_DIR"], "alice.txt")).read()[:int(os.environ.get("PREFILL_CHARS", "21000"))]; ctx = "8192"
env = dict(os.environ, DS4_LOCK_FILE=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"))
res = []
for i in range(runs):
    out = "%s/speed_%s_%s_%d" % (WORK, mode, tag, i)
    cmd = ["./ds4", "-m", MODEL, "--metal", "--raw-prompt", "--temp", "0", "--ctx", ctx, "-t", "8", "-n", ntok, "-p", prompt]
    p = subprocess.run(cmd, cwd=DS4_DIR, env=env, capture_output=True, text=True)
    open(out + ".err", "w").write(p.stderr)
    m = re.search(r"prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s", p.stderr)
    r = {"run": i, "rc": p.returncode, "prefill_tps": float(m.group(1)) if m else None, "decode_tps": float(m.group(2)) if m else None}
    res.append(r); print(json.dumps(r), flush=True)
key = "decode_tps" if mode == "decode" else "prefill_tps"
v = [r[key] for r in res if r[key]]
if v: print(json.dumps({"mode": mode, "tag": tag, "n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}))
