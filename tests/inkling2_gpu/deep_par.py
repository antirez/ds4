#!/usr/bin/env python3
"""Deep-context parity: GPU (chunk 512) on a long non-repetitive prompt dumps its router picks, the CPU oracle replays them,
the top-20 log-prob gap at the last NPOS positions is reported.  usage: deep_par.py NTOK_APPROX [NPOS]
The prompt is a prefix of the books in $IK_TEXT_DIR (moby.txt, twocities.txt, pride.txt, holmes.txt), about 3.25 characters per
token.  Takes about 30 minutes at 32k tokens (the CPU oracle runs ~17 tokens/s).
Environment: IK_MODEL, IK_TEXT_DIR, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]
SRC = os.environ["IK_TEXT_DIR"]
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))
D = os.path.abspath(os.environ.get("DS4_DIR", "."))
os.makedirs(WORK, exist_ok=True)
approx = int(sys.argv[1]); npos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
text = "".join(open(os.path.join(SRC, f)).read() for f in ("moby.txt", "twocities.txt", "pride.txt", "holmes.txt"))[: int(approx * 3.25)]
env = dict(os.environ, DS4_LOCK_FILE=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"))
tf = WORK + "/tok_in.txt"
open(tf, "w").write("0" + json.dumps(text) + "\n")
p = subprocess.run(["./ds4", "-m", MODEL, "--metal", "--first-token-test", "--raw-prompt", "-p", "x"], cwd=D, env=dict(env, DS4_INKLING2_TOK_FILE=tf),
                   capture_output=True, text=True)
ids = json.loads(p.stdout.strip().splitlines()[-1]); n = len(ids)
open(WORK + "/ids.txt", "w").write(",".join(map(str, ids)))
pos = ",".join(str(n - npos + k) for k in range(npos))
print("tokens", n, "positions", pos, flush=True)


def run(backend, out, extra, chunk):
    e = dict(env, DS4_INKLING2_FT_TOKENS=WORK + "/ids.txt", DS4_INKLING2_FT_POS=pos, DS4_INKLING2_CHUNK=str(chunk),
             DS4_INKLING2_FT_TOPK="200", DS4_INKLING2_FT_OUT=out, **extra)
    cmd = ["./ds4", "-m", MODEL, backend, "--first-token-test", "--raw-prompt", "-p", "x", "-t", "8"]
    t0 = time.time()
    with open(out + ".log", "w") as lf:
        rc = subprocess.call(cmd, cwd=D, env=e, stdout=lf, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


g, gp = WORK + "/gpu.jsonl", WORK + "/gpu.picks"
rc, dt = run("--metal", g, {"DS4_INKLING2_PICKS_OUT": gp}, 512)
print("gpu rc", rc, round(dt, 1), "s", flush=True)
if rc: sys.exit(1)
c = WORK + "/cpu_forced.jsonl"
rc, dt = run("--cpu", c, {"DS4_INKLING2_FT_PICKS": gp, "DS4_INKLING2_FT_PICKLOG": WORK + "/pickdiff.txt"}, 256)
print("cpu rc", rc, round(dt, 1), "s", flush=True)
if rc: sys.exit(1)


def load(path):
    d = {}
    for l in open(path):
        j = json.loads(l); d[j["pos"]] = {i: lp for i, lp in j["top"]}
    return d


r, m = load(c), load(g)
worst = 0.0; top1 = 0; per = []
for q in sorted(r):
    rt = sorted(r[q].items(), key=lambda x: -x[1])
    d = max(abs(m[q].get(i, -99.0) - lp) for i, lp in rt[:20])
    top1 += max(m[q], key=m[q].get) == rt[0][0]; worst = max(worst, d); per.append(round(d, 5))
nd = sum(1 for _ in open(WORK + "/pickdiff.txt")) if os.path.exists(WORK + "/pickdiff.txt") else 0
res = {"ntok": n, "npos": len(r), "top1": f"{top1}/{len(r)}", "worst_top20": round(worst, 5), "per_pos": per,
       "forced_pickdiff_decisions": nd, "decisions": n * 48}
print(json.dumps(res)); open(WORK + "/result.json", "w").write(json.dumps(res))
