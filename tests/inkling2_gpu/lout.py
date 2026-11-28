#!/usr/bin/env python3
"""Per-slot l_out comparison, GPU vs CPU oracle with the GPU's picks forced.  usage: lout.py LABEL NAME
LABEL only names the output files; the model is $IK_MODEL.  Needs $IK_VECTORS/logits_bf16.jsonl.
Runs both with DS4_INKLING2_FT_VISITS and compares 'lout' lines by (slot, pos): per slot the median and max of
|gpu - cpu| over tokens, and the same relative to the CPU row-sum magnitude.
Environment: IK_MODEL, IK_VECTORS, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import re, statistics
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]                                              # the Inkling 2 Mini GGUF
VEC = os.path.join(os.environ["IK_VECTORS"], "logits_bf16.jsonl")           # llama.cpp reference vectors
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))              # scratch output directory
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))                   # directory holding ./ds4
LOCK = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
os.makedirs(WORK, exist_ok=True)
alias, name = sys.argv[1], sys.argv[2]
row = [json.loads(l) for l in open(VEC) if json.loads(l)["name"] == name][0]
pl = len(row["prompt_ids"]); toks = row["prompt_ids"] + row["positions"][8]["forced_ids"]
def run(backend, tag, extra):
    out = WORK + "/lout_%s_%s_%s" % (alias, name, tag)
    env = dict(os.environ, DS4_LOCK_FILE=LOCK, DS4_INKLING2_FT_TOKENS=",".join(map(str, toks)),
               DS4_INKLING2_FT_POS=str(len(toks) - 1), DS4_INKLING2_CHUNK="512", DS4_INKLING2_FT_OUT=out + ".jsonl",
               DS4_INKLING2_FT_VISITS=out + ".visits")
    env.update(extra)
    cmd = ["./ds4", "-m", MODEL, backend, "--first-token-test", "--raw-prompt", "-p", "x", "-t", "8"]
    with open(out + ".log", "w") as lf:
        rc = subprocess.call(cmd, cwd=DS4_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
    assert rc == 0, out
    return out + ".visits"
gv = run("--metal", "gpu", {"DS4_INKLING2_PICKS_OUT": WORK + "/lout_%s_%s.picks" % (alias, name)})
cv = run("--cpu", "cpu", {"DS4_INKLING2_FT_PICKS": WORK + "/lout_%s_%s.picks" % (alias, name)})
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r"lout visit=(\d+) slot=(\d+) pos=(\d+) (\S+)", l)
        if m: d[(int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    return d
g, c = load(gv), load(cv)
assert set(g) == set(c), (len(g), len(c))
res = []
for slot in range(48):
    diffs = [abs(g[(slot, p)] - c[(slot, p)]) for (s, p) in c if s == slot]
    mags = [abs(c[(slot, p)]) for (s, p) in c if s == slot]
    res.append((slot, statistics.median(diffs), max(diffs), statistics.median(mags)))
worst_rel = max(r[2] / max(r[3], 1e-9) for r in res)
print(json.dumps({"model": alias, "prompt": name, "ntok": len(toks), "slots": 48,
                  "max_abs_diff_any_slot": max(r[2] for r in res), "worst_max_over_median_mag": worst_rel,
                  "per_slot": [[r[0], round(r[1], 4), round(r[2], 4), round(r[3], 2)] for r in res]}))
