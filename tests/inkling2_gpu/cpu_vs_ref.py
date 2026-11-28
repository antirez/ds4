#!/usr/bin/env python3
"""CPU oracle on prompt + llama.cpp's own 64 greedy tokens (teacher forced), per-step top-20 logprob difference vs the
reference completion_probabilities, next to the GPU's (from the gen_cmp.py --dump-logprobs run).  usage: cpu_vs_ref.py LABEL NAME
Needs $IK_VECTORS/logits_bf16.jsonl and $IK_VECTORS/LABEL_logprobs.json (llama.cpp /completion answers per prompt name, as used by
gen_cmp.py).  Environment: IK_MODEL, IK_VECTORS, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]                                              # the Inkling 2 Mini GGUF
VEC = os.path.join(os.environ["IK_VECTORS"], "logits_bf16.jsonl")           # llama.cpp reference vectors
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))              # scratch output directory
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))                   # directory holding ./ds4
LOCK = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
os.makedirs(WORK, exist_ok=True)
alias, name = sys.argv[1], sys.argv[2]
ref = json.load(open(os.path.join(os.environ["IK_VECTORS"], "%s_logprobs.json" % alias)))[name]
vec = [json.loads(l) for l in open(VEC) if json.loads(l)["name"] == name][0]
pids = vec["prompt_ids"]; cp = ref["resp"]["completion_probabilities"]; rids = [c["id"] for c in cp]
toks = pids + rids[:-1]; pos = [len(pids) - 1 + i for i in range(len(rids))]
out = "%s/cpuref_%s_%s.jsonl" % (WORK, alias, name)
env = dict(os.environ, DS4_LOCK_FILE=LOCK, DS4_INKLING2_FT_TOKENS=",".join(map(str, toks)), DS4_INKLING2_FT_POS=",".join(map(str, pos)),
           DS4_INKLING2_CHUNK="256", DS4_INKLING2_FT_TOPK="50", DS4_INKLING2_FT_OUT=out)
if not os.path.exists(out): subprocess.check_call(["./ds4", "-m", MODEL, "--cpu", "--first-token-test", "--raw-prompt", "-p", "x", "-t", "8"],
                      cwd=DS4_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
mine = {}
for l in open(out):
    j = json.loads(l); mine[j["pos"]] = {i: lp for i, lp in j["top"]}
gpu = {i: {t["token"]["id"]: t["logprob"] for t in s["top_logprobs"]} for i, s in enumerate(json.load(open("%s/gen_%s_%s_lp.json" % (WORK, alias, name)))["steps"])}
rows = []
for i, c in enumerate(cp):
    m = mine[pos[i]]; g = gpu[i]
    dc = max(abs(m.get(t["id"], -99) - t["logprob"]) for t in c["top_logprobs"][:20])
    dg = max([abs(g[t["id"]] - t["logprob"]) for t in c["top_logprobs"][:20] if t["id"] in g] or [0])
    rows.append((i, round(dc, 4), round(dg, 4)))
print(json.dumps({"model": alias, "prompt": name, "cpu_worst": max(r[1] for r in rows), "gpu_worst": max(r[2] for r in rows),
                  "worst_step_cpu": max(rows, key=lambda r: r[1])[0], "worst_step_gpu": max(rows, key=lambda r: r[2])[0],
                  "steps_over_0.1_cpu": sum(r[1] > 0.1 for r in rows), "steps_over_0.1_gpu": sum(r[2] > 0.1 for r in rows),
                  "first_step_over_0.1_cpu": next((r[0] for r in rows if r[1] > 0.1), None), "first_step_over_0.1_gpu": next((r[0] for r in rows if r[2] > 0.1), None)}))
