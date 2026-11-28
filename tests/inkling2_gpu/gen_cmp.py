#!/usr/bin/env python3
"""Greedy generation through the normal CLI session path vs llama.cpp's reference (64 tokens, 3 prompts).
usage: gen_cmp.py LABEL [ctx]   -> prints one JSON line per prompt.
LABEL only names the output files and picks the reference file $IK_VECTORS/LABEL_logprobs.json (llama.cpp /completion answers
per prompt name: {"capital"|"count"|"code": {"prompt": ..., "resp": {...completion_probabilities, content}}}).
Environment: IK_MODEL, IK_VECTORS, IK_WORK (default ./ik-work), DS4_DIR (default .), DS4_LOCK_FILE."""
import json, os, subprocess, sys, time
MODEL = os.environ["IK_MODEL"]                                              # the Inkling 2 Mini GGUF
WORK = os.path.abspath(os.environ.get("IK_WORK", "./ik-work"))              # scratch output directory
DS4_DIR = os.path.abspath(os.environ.get("DS4_DIR", "."))                   # directory holding ./ds4
LOCK = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
os.makedirs(WORK, exist_ok=True)
alias = sys.argv[1]; ctx = sys.argv[2] if len(sys.argv) > 2 else "4096"
ref = json.load(open(os.path.join(os.environ["IK_VECTORS"], "%s_logprobs.json" % alias)))
env = dict(os.environ, DS4_LOCK_FILE=LOCK)
def run(args, out):
    cmd = ["./ds4", "-m", MODEL, "--metal", "--raw-prompt", "--temp", "0", "--ctx", ctx, "-t", "8"] + args
    t0 = time.time()
    with open(out + ".out", "w") as o, open(out + ".err", "w") as e:
        rc = subprocess.call(cmd, cwd=DS4_DIR, env=env, stdout=o, stderr=e)
    return rc, time.time() - t0
for name in ["capital", "count", "code"]:
    prompt = ref[name]["prompt"]; r = ref[name]["resp"]
    rids = [c["id"] for c in r["completion_probabilities"]]
    rtop = [[t["logprob"] for t in c["top_logprobs"]] for c in r["completion_probabilities"]]
    base = "%s/gen_%s_%s" % (WORK, alias, name)
    rc, dt = run(["-p", prompt, "-n", "64"], base + "_text")
    text = open(base + "_text.out").read()
    rc2, dt2 = run(["-p", prompt, "-n", "64", "--dump-logprobs", base + "_lp.json", "--logprobs-top-k", "20"], base + "_lp")
    res = {"model": alias, "prompt": name, "text_rc": rc, "lp_rc": rc2, "text_equals_ref_content": text.rstrip("\n") == r["content"].rstrip("\n")}
    if rc2 == 0:
        d = json.load(open(base + "_lp.json"))["steps"]
        mine = [s["selected"]["id"] for s in d]
        res["n"] = len(mine)
        k = next((i for i in range(min(len(mine), len(rids))) if mine[i] != rids[i]), None)
        res["first_divergence"] = k
        if k is None:
            res["identical_64"] = len(mine) >= len(rids) and mine[:len(rids)] == rids
        else:
            my_lp = [t["logprob"] for t in d[k]["top_logprobs"]]
            res["ref_top2_gap"] = round(rtop[k][0] - rtop[k][1], 5)
            res["ds4_top2_gap"] = round(my_lp[0] - my_lp[1], 5)
            res["ref_token"] = rids[k]; res["ds4_token"] = mine[k]
            ids20 = [t["token"]["id"] for t in d[k]["top_logprobs"]]
            res["ds4_has_ref_token_at_rank"] = ids20.index(rids[k]) if rids[k] in ids20 else None
            mylp = {t["token"]["id"]: t["logprob"] for t in d[k]["top_logprobs"]}
            reflp = {t["id"]: t["logprob"] for t in r["completion_probabilities"][k]["top_logprobs"]}
            res["logprob_of_ref_token_ds4_vs_ref"] = [round(mylp.get(rids[k], -99), 4), round(reflp[rids[k]], 4)]
            res["logprob_of_ds4_token_ds4_vs_ref"] = [round(mylp[mine[k]], 4), round(reflp.get(mine[k], -99), 4)]
        # per-step gap diagnostics over the matching prefix
        end = k if k is not None else min(len(mine), len(rids))
        worst = 0.0
        for i in range(end):
            mylp = {t["token"]["id"]: t["logprob"] for t in d[i]["top_logprobs"]}
            for t in r["completion_probabilities"][i]["top_logprobs"][:20]:
                if t["id"] in mylp: worst = max(worst, abs(mylp[t["id"]] - t["logprob"]))
        res["max_logprob_diff_top20_over_the_steps"] = round(worst, 4)
        res["tokens_per_s_note"] = "see .err timing line"
    print(json.dumps(res), flush=True)
