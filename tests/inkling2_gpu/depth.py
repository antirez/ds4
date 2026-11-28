#!/usr/bin/env python3
"""Deep-context decode speed through the CLI.  usage: depth.py DEPTH[,DEPTH...] [--tag T] [--tokens N] [--env K=V ...]
DEPTH = approximate prompt length in tokens (text from the plain-text books in $IK_TEXT_DIR: moby.txt, twocities.txt, pride.txt,
holmes.txt; about 3.25 characters per token; the exact count is read from ds4's log).  Greedy, N generated tokens (default 200).
Prints per depth: prompt tokens, prefill t/s, decode t/s.  The decode figure is ds4's own "generation" figure and covers only the
generated tokens.
Environment: IK_MODEL, IK_TEXT_DIR, IK_WORK (default ./ik-work, stderr of each run goes there), DS4_DIR (default .; runs ./ds4
from another checkout), DS4_LOCK_FILE."""
import json, os, re, subprocess, sys
MODEL = os.environ["IK_MODEL"]
SRC = os.environ["IK_TEXT_DIR"]
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))
os.makedirs(WORK, exist_ok=True)
depths = [int(x) for x in sys.argv[1].split(",")]
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "d"
ntok = sys.argv[sys.argv.index("--tokens") + 1] if "--tokens" in sys.argv else "200"
extra_env = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--env") + 1:]) if "--env" in sys.argv else {}
text = "".join(open(os.path.join(SRC, f)).read() for f in ("moby.txt", "twocities.txt", "pride.txt", "holmes.txt"))
env = dict(os.environ, DS4_LOCK_FILE=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"), **extra_env)
for d in depths:
    if d <= 100:
        prompt = "The history of the Roman Empire begins with"
    else:
        prompt = text[:int(d * 3.25)]
    cmd = ["./ds4", "-m", MODEL, "--metal", "--raw-prompt", "--temp", "0",
           "--ctx", str(int(d * 1.4) + int(ntok) + 400), "-t", "8", "-n", ntok, "-p", prompt]
    p = subprocess.run(cmd, cwd=DS4_DIR, env=env, capture_output=True, text=True)
    open("%s/depth_%s_%d.err" % (WORK, tag, d), "w").write(p.stderr)
    m = re.search(r"prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s", p.stderr)
    n = re.search(r"processing (\d+) input tokens", p.stderr)
    print(json.dumps({"tag": tag, "depth": d, "prompt_tokens": int(n.group(1)) if n else None, "rc": p.returncode,
                      "prefill_tps": float(m.group(1)) if m else None, "decode_tps": float(m.group(2)) if m else None}), flush=True)
