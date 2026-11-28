#!/usr/bin/env python3
"""Chunk-schedule invariance (GPU only): same tokens through the GPU graph with different prefill chunk schedules; compares raw
f32 logits of the 9 test positions against the first schedule, free-running and with the router picks of the first
schedule forced.  usage: chunk.py LABEL NAME 512,100,1 [--tail K] [--nofree]   (needs numpy)
LABEL only names the output files; the model is $IK_MODEL.  Needs $IK_VECTORS/logits_bf16.jsonl.
Environment: IK_MODEL, IK_VECTORS, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import numpy as np
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]                                              # the Inkling 2 Mini GGUF
VEC = os.path.join(os.environ["IK_VECTORS"], "logits_bf16.jsonl")           # llama.cpp reference vectors
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))              # scratch output directory
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))                   # directory holding ./ds4
LOCK = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
os.makedirs(WORK, exist_ok=True)
alias, name = sys.argv[1], sys.argv[2]
# schedule entries: "512" (all chunks of 512), "512t8" (chunks of 512, the last 8 tokens one at a time)
sched = []
for x in sys.argv[3].split(","):
    c, _, tl = x.partition("t")
    sched.append((int(c), int(tl) if tl else 0))
gtail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 0
nforce = "--nofree" not in sys.argv
row = [json.loads(l) for l in open(VEC) if json.loads(l)["name"] == name][0]
pl = len(row["prompt_ids"]); toks = row["prompt_ids"] + row["positions"][8]["forced_ids"]
positions = [pl - 1 + k for k in range(9)]
V = 200058

def run(tag, chunk, tail, extra):
    out = WORK + "/chunk_%s_%s_%s" % (alias, name, tag)
    env = dict(os.environ, DS4_LOCK_FILE=LOCK, DS4_INKLING2_FT_TOKENS=",".join(map(str, toks)),
               DS4_INKLING2_FT_POS=",".join(map(str, positions)), DS4_INKLING2_CHUNK=str(chunk), DS4_INKLING2_FT_TOPK="50",
               DS4_INKLING2_FT_OUT=out + ".jsonl", DS4_INKLING2_FT_LOGITS=out + ".f32", DS4_INKLING2_PICKS_OUT=out + ".picks")
    if tail: env["DS4_INKLING2_DECODE_TAIL"] = str(tail)
    env.update(extra)
    cmd = ["./ds4", "-m", MODEL, "--metal", "--first-token-test", "--raw-prompt", "-p", "x", "-t", "8"]
    t0 = time.time()
    with open(out + ".log", "w") as lf:
        rc = subprocess.call(cmd, cwd=DS4_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    lg = np.fromfile(out + ".f32", dtype=np.float32).reshape(9, V) if rc == 0 else None
    pk = np.fromfile(out + ".picks", dtype=np.int32) if rc == 0 else None
    return rc, dt, lg, pk, out + ".picks"

def lp(l):
    m = l.max(axis=1, keepdims=True)
    return l - (m + np.log(np.exp((l - m).astype(np.float64)).sum(axis=1, keepdims=True)))

res = {"model": alias, "prompt": name, "ntok": len(toks), "sched": sys.argv[3], "runs": []}
ref = None
for ch, tl in sched:
    tl = tl or gtail
    for mode in (["free", "forced"] if nforce else ["free"]):
        if ref is None and mode == "forced": continue
        extra = {"DS4_INKLING2_FT_PICKS": ref[3]} if mode == "forced" else {}
        rc, dt, lg, pk, pkpath = run("c%dt%d_%s" % (ch, tl, mode), ch, tl, extra)
        r = {"chunk": ch, "tail": tl, "mode": mode, "rc": rc, "s": round(dt, 1)}
        if rc == 0:
            if ref is None:
                ref = (lg, pk, ch, pkpath); r["ref"] = True
            else:
                same = np.array_equal(lg, ref[0])
                r["logits_identical_to_first"] = bool(same)
                a, b = lp(lg), lp(ref[0])
                top = np.argsort(-b, axis=1)[:, :20]
                r["worst_top20_logprob_diff"] = float(np.abs(np.take_along_axis(a, top, 1) - np.take_along_axis(b, top, 1)).max())
                r["max_abs_logit_diff"] = float(np.abs(lg - ref[0]).max())
                r["top1_equal"] = int((lg.argmax(1) == ref[0].argmax(1)).sum())
                r["pick_sets_differ"] = int(sum(set(pk[i:i + 8]) != set(ref[1][i:i + 8]) for i in range(0, len(pk), 8))) if mode == "free" else 0
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
open(WORK + "/results_chunk.jsonl", "a").write(json.dumps(res) + "\n")
