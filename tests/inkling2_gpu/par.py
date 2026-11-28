#!/usr/bin/env python3
"""Parity driver: GPU (Metal) against the CPU oracle.  usage:
  par.py LABEL NAME CHUNK [--free] [--npos 9] [--cpuchunk 256] [--tag T] [--nocpu]
LABEL (for example bf16 or nvfp4) only names the output files; the model is $IK_MODEL.  Needs $IK_VECTORS/logits_bf16.jsonl.
Runs the GPU first-token oracle on prompt NAME (+8 forced ids from the llama.cpp vectors), dumps its router picks, replays them
in the CPU oracle (picks forced), compares top-20 logprobs.
--free also runs the CPU oracle unforced and reports the share of router decisions that differ from the GPU's.
Environment: IK_MODEL, IK_VECTORS, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import array
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]                                              # the Inkling 2 Mini GGUF
VEC = os.path.join(os.environ["IK_VECTORS"], "logits_bf16.jsonl")           # llama.cpp reference vectors
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))              # scratch output directory
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))                   # directory holding ./ds4
LOCK = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
os.makedirs(WORK, exist_ok=True)
args = sys.argv[1:]
alias, name, chunk = args[0], args[1], int(args[2])
npos = int(args[args.index("--npos") + 1]) if "--npos" in args else 9
cpuchunk = args[args.index("--cpuchunk") + 1] if "--cpuchunk" in args else "256"
tag = args[args.index("--tag") + 1] if "--tag" in args else "p"
free = "--free" in args
skipcpu = "--nocpu" in args
row = [json.loads(l) for l in open(VEC) if json.loads(l)["name"] == name][0]
pl = len(row["prompt_ids"]); forced = row["positions"][npos - 1]["forced_ids"]
toks = row["prompt_ids"] + forced; ntok = len(toks)
positions = [pl - 1 + k for k in range(npos)]
T = "%s_%s_%s" % (alias, name, tag)

def run(backend, out, extra, chunk_, logtag):
    env = dict(os.environ, DS4_LOCK_FILE=LOCK,
               DS4_INKLING2_FT_TOKENS=",".join(map(str, toks)), DS4_INKLING2_FT_POS=",".join(map(str, positions)),
               DS4_INKLING2_CHUNK=str(chunk_), DS4_INKLING2_FT_TOPK="200", DS4_INKLING2_FT_OUT=out)
    env.update(extra)
    cmd = ["./ds4", "-m", MODEL, backend, "--first-token-test", "--raw-prompt", "-p", "x", "-t", "8"]
    t0 = time.time()
    with open(WORK + "/%s_%s.log" % (T, logtag), "w") as lf:
        rc = subprocess.call(cmd, cwd=DS4_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return rc, time.time() - t0

def load(path):
    d = {}
    for l in open(path):
        j = json.loads(l); d[j["pos"]] = {i: lp for i, lp in j["top"]}
    return d

def cmp(ref, mine):
    r, m = load(ref), load(mine)
    worst = 0.0; top1 = 0; per = []
    for p in positions:
        rt = sorted(r[p].items(), key=lambda x: -x[1])
        d = max(abs(m[p].get(i, -99.0) - lp) for i, lp in rt[:20])
        ok = max(m[p], key=m[p].get) == rt[0][0]
        top1 += ok; worst = max(worst, d); per.append(round(d, 5))
    return top1, worst, per

def picks(path):
    a = array.array("i"); a.frombytes(open(path, "rb").read()); return a

res = {"model": alias, "prompt": name, "ntok": ntok, "chunk": chunk, "npos": npos}
g = WORK + "/%s_gpu.jsonl" % T; gp = WORK + "/%s_gpu.picks" % T
rc, dt = run("--metal", g, {"DS4_INKLING2_PICKS_OUT": gp}, chunk, "gpu")
res["gpu_rc"] = rc; res["gpu_s"] = round(dt, 1)
if rc: print(json.dumps(res)); sys.exit(1)
if not skipcpu:
    c = WORK + "/%s_cpuforced.jsonl" % T
    rc, dt = run("--cpu", c, {"DS4_INKLING2_FT_PICKS": gp, "DS4_INKLING2_FT_PICKLOG": WORK + "/%s_pickdiff.txt" % T}, cpuchunk, "cpuforced")
    res["cpu_forced_rc"] = rc; res["cpu_forced_s"] = round(dt, 1)
    if rc == 0:
        t1, w, per = cmp(c, g)
        res.update(top1="%d/%d" % (t1, npos), worst_top20=round(w, 5), per_pos=per)
        nd = sum(1 for _ in open(WORK + "/%s_pickdiff.txt" % T)) if os.path.exists(WORK + "/%s_pickdiff.txt" % T) else 0
        res["forced_pickdiff_decisions"] = nd
        res["decisions"] = ntok * 48
if free and not skipcpu:
    c2 = WORK + "/%s_cpufree.jsonl" % T; cp = WORK + "/%s_cpufree.picks" % T
    rc, dt = run("--cpu", c2, {"DS4_INKLING2_PICKS_OUT": cp}, cpuchunk, "cpufree")
    res["cpu_free_rc"] = rc; res["cpu_free_s"] = round(dt, 1)
    if rc == 0:
        a, b = picks(gp), picks(cp)
        # compare as sets per (slot, pos): a decision "differs" if the sets of 8 ids differ
        dif = 0; tot = 0
        for i in range(0, len(a), 8):
            tot += 1
            if set(a[i:i + 8]) != set(b[i:i + 8]): dif += 1
        res["free_decisions"] = tot; res["free_pick_set_diff"] = dif; res["free_diff_pct"] = round(100.0 * dif / tot, 3)
        t1, w, per = cmp(c2, g)
        res.update(free_top1="%d/%d" % (t1, npos), free_worst_top20=round(w, 5))
print(json.dumps(res))
open(WORK + "/results.jsonl", "a").write(json.dumps(res) + "\n")
