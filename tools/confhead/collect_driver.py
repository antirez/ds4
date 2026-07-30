#!/usr/bin/env python3
"""Confidence-head retrain, stage 1 driver: the labeled-position request suite.

Runs a temp x workload x depth mixture against a live ds4-server whose fused
DSpark loop is dumping lean confidence records (PULSAR_DSPARK_DUMP_LEAN=1) with
conf-sched OFF (unbiased, fully-verified labels).  The dump supplies features;
the fused stats log supplies labels (committed counts = the engine's actual
p/q sampled-acceptance outcomes — or greedy argmax-equality on the temp-0
legs, which stay in the mix so the one head keeps serving both regimes).

Mixture: temperature {0, 0.7, 0.95} x workload {prose, code, structured} x
depth {~1k, ~8k, ~28k tokens} x 2 prompt variants = 54 requests, 512 tokens
each.  The depth axis is the point: the head was never trained past ~2k ctx
and Item 1's acceptance gain concentrates at depth.

Writes a manifest (one JSON object per line, in request order) that
retrain.py aligns with the dump's request segmentation — the n-th generating
request in the manifest is the n-th n_batch==1 segment in the stats log.
Requests are strictly sequential; a failed request is recorded and counted as
a segment only if it produced completion tokens.
"""
import argparse, glob, json, os, random, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# chars-per-token estimates by workload (validated against usage in the manifest;
# run1 measured: prose 3.7, code 3.7, structured 2.0 -- log lines full of IPs and
# numbers tokenize ~2 chars/token, the original 2.9 blew the 30k cell past -c)
CPT = {"prose": 3.6, "code": 3.4, "structured": 2.0}
DEPTHS = {"1k": 1000, "8k": 8000, "30k": 28000}   # 28k keeps headroom under -c
GEN_TOKENS = 512

def prose_prompt(depth_tok, variant):
    docs = sorted(glob.glob(os.path.join(ROOT, "temp/pilot_corpus2/doc_*.txt")))
    assert docs, "temp/pilot_corpus2 missing"
    budget = int(depth_tok * CPT["prose"])
    start = 300 + variant * 60 + (0 if depth_tok < 2000 else 8 if depth_tok < 16000 else 20)
    buf, i = [], start
    total = 0
    while total < budget and i < len(docs):
        t = open(docs[i], encoding="utf-8", errors="ignore").read()
        buf.append(t)
        total += len(t)
        i += 1
    text = "\n\n".join(buf)[:budget]
    return text + "\n\nContinue this document in the same style and register, developing the ideas above in depth.\n"

CODE_SETS = [
    ["src/engine/session.c"],
    ["src/engine/gpu_decode.c", "src/cuda/ds4_cuda_dspark.cu", "src/engine/gpu_prefill.c",
     "src/cuda/ds4_cuda_moe.cu", "src/engine/imatrix.c", "src/server/generate.c"],
]
def code_prompt(depth_tok, variant):
    budget = int(depth_tok * CPT["code"])
    buf, total = [], 0
    for rel in CODE_SETS[variant % len(CODE_SETS)]:
        t = open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore").read()
        buf.append(f"/* ===== {rel} ===== */\n" + t)
        total += len(t)
        if total >= budget:
            break
    text = "\n\n".join(buf)[:budget]
    return text + "\n\n/* Continue this codebase: implement the next helper functions in the same style, with comments explaining memory policy and cache lifetime. */\n"

HOSTS = ["10.0.%d.%d" % (a, b) for a in range(4) for b in range(1, 60)]
PATHS = ["/api/v1/items", "/api/v1/users", "/static/app.js", "/health", "/api/v2/search",
         "/login", "/logout", "/api/v1/orders", "/metrics", "/favicon.ico"]
def structured_prompt(depth_tok, variant):
    rng = random.Random(1234 + variant)
    budget = int(depth_tok * CPT["structured"])
    rows = []
    total = 0
    t0 = 1760600000
    while total < budget:
        line = ('%s - - [%d] "GET %s HTTP/1.1" %d %d %dms\n' %
                (rng.choice(HOSTS), t0 + rng.randint(0, 86400), rng.choice(PATHS),
                 rng.choice([200, 200, 200, 200, 301, 404, 500, 503]),
                 rng.randint(80, 50000), rng.randint(1, 900)))
        rows.append(line)
        total += len(line)
    text = "".join(rows)[:budget]
    return ("Server access log:\n\n" + text +
            "\n\nProduce a detailed JSON report aggregating the records above: request counts per "
            "status code, the top 10 hosts by request count, the top 10 paths, total bytes served, "
            "and mean latency per path. Output valid JSON only.\n")

BUILDERS = {"prose": prose_prompt, "code": code_prompt, "structured": structured_prompt}

def request(port, prompt, temperature, max_tokens, timeout, think):
    # ds4-server's /v1/completions renders a chat wrapper and defaults
    # thinking ON; with thinking enabled the server FORCES temp>0 requests to
    # the thinking defaults (temp 1.0, top_p 1.0 -- generate.c). "think":
    # false makes the requested temperature real.
    body = {"model": "d", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature}
    if not think:
        body["think"] = False
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}),
        timeout=timeout)
    return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--dry-run", action="store_true", help="build prompts, print sizes, no requests")
    ap.add_argument("--think-off", action="store_true",
                    help="send think:false so requested temperatures are honored")
    ap.add_argument("--temps", default="0,0.7,0.95",
                    help="comma-separated temperature legs")
    ap.add_argument("--only", default="",
                    help="restrict to workload:depth cells, e.g. structured:30k (comma-separated)")
    args = ap.parse_args()
    temps = [float(t) for t in args.temps.split(",")]
    only = set(tuple(c.split(":")) for c in args.only.split(",") if c)

    entries = []
    idx = 0
    # depth-major outer order keeps same-prompt temps adjacent (KV prefix reuse)
    for workload in ("prose", "code", "structured"):
        for dname, dtok in DEPTHS.items():
            if only and (workload, dname) not in only:
                continue
            for variant in (0, 1):
                prompt = BUILDERS[workload](dtok, variant)
                if args.dry_run:
                    print(f"[dry] {workload}/{dname}/v{variant}: {len(prompt)} chars "
                          f"(~{len(prompt)/CPT[workload]:.0f} tok est)", flush=True)
                    continue
                for temp in temps:
                    tag = f"{workload}/{dname}/v{variant}/t{temp}"
                    ent = {"idx": idx, "tag": tag, "workload": workload,
                           "depth": dname, "variant": variant, "temperature": temp,
                           "think": "off" if args.think_off else "on",
                           "status": "pending", "prompt_tokens": 0, "completion_tokens": 0}
                    t0 = time.time()
                    try:
                        resp = request(args.port, prompt, temp, GEN_TOKENS,
                                       args.timeout, not args.think_off)
                        u = resp["usage"]
                        ent.update(status="ok", prompt_tokens=u["prompt_tokens"],
                                   completion_tokens=u["completion_tokens"])
                        print(f"[{idx:2d}] {tag}: prompt={u['prompt_tokens']} "
                              f"gen={u['completion_tokens']} in {time.time()-t0:.0f}s", flush=True)
                    except Exception as e:
                        ent.update(status=f"fail: {e}")
                        print(f"[{idx:2d}] {tag}: FAILED {e}", flush=True)
                    entries.append(ent)
                    idx += 1
                    with open(args.manifest, "w") as f:
                        for e2 in entries:
                            f.write(json.dumps(e2) + "\n")
    if not args.dry_run:
        ok = sum(1 for e in entries if e["status"] == "ok")
        print(f"DONE: {ok}/{len(entries)} requests ok", flush=True)

if __name__ == "__main__":
    main()
