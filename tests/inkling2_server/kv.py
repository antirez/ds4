#!/usr/bin/env python3
"""KV disk cache server check.  usage: kv.py [--port 8000] phase1|phase2 [PROMPT_CHARS]   (text from $IK_TEXT_DIR/pride.txt)
phase1: a long first turn (about 6k tokens) then a follow-up, prints prompt/cached tokens of each; saves the conversation.
phase2: (after a server restart on the same --kv-disk-dir) the same conversation plus one more question: prompt/cached tokens."""
import json, os, sys, time, urllib.request
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
phase = sys.argv[1]; chars = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
SRC = os.environ["IK_TEXT_DIR"] + "/"
HIST = "/tmp/inkling_kv_hist.json"


def post(msgs, mt=30):
    body = json.dumps({"model": "inkling-2-mini", "messages": msgs, "temperature": 0, "max_tokens": mt}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1200).read())
    return r, time.time() - t0


def cached(u):
    return u.get("prompt_tokens_details", {}).get("cached_tokens", u.get("cached_tokens"))


def asst(r):
    m = r["choices"][0]["message"]
    a = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        a["reasoning_content"] = m["reasoning_content"]
    return a


if phase == "phase1":
    text = open(SRC + "pride.txt").read()[5000:5000 + chars]
    h = [{"role": "user", "content": "Read this passage and then answer briefly.\n\n" + text + "\n\nQuestion: who is the main character?"}]
    r, dt = post(h)
    print(f"turn 1: prompt={r['usage']['prompt_tokens']} cached={cached(r['usage'])} {dt:.1f}s")
    h.append(asst(r))
    h.append({"role": "user", "content": "And which family does she belong to?"})
    r, dt = post(h)
    print(f"turn 2: prompt={r['usage']['prompt_tokens']} cached={cached(r['usage'])} {dt:.1f}s")
    h.append(asst(r))
    json.dump(h, open(HIST, "w"))
else:
    h = json.load(open(HIST))
    h.append({"role": "user", "content": "Name one more character."})
    r, dt = post(h)
    print(f"turn 3 (after restart): prompt={r['usage']['prompt_tokens']} cached={cached(r['usage'])} {dt:.1f}s")
