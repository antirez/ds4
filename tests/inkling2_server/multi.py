#!/usr/bin/env python3
"""Concurrent sessions give the same answer as the same request alone; alternating conversations keep their cache.
usage: multi.py [--port 8000] [N] [MAX_TOKENS]"""
import json, sys, time, threading, urllib.request
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; MT = int(sys.argv[2]) if len(sys.argv) > 2 else 60
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
TOPICS = ["the Roman aqueducts", "how tides work", "the history of the printing press", "photosynthesis", "the Silk Road",
          "how a transistor works", "the French Revolution", "plate tectonics"]


def post(msgs, mt=MT):
    body = json.dumps({"model": "inkling-2-mini", "messages": msgs, "temperature": 0, "max_tokens": mt, "stream": False}).encode()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900)
    return json.loads(r.read())


def msg(i):
    return [{"role": "user", "content": f"Write two sentences about {TOPICS[i % len(TOPICS)]}."}]


def text(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")


def cached(u):
    return u.get("prompt_tokens_details", {}).get("cached_tokens", u.get("cached_tokens"))


alone = [text(post(msg(i))) for i in range(N)]
res = [None] * N
t0 = time.time()


def w(i):
    res[i] = post(msg(i))


th = [threading.Thread(target=w, args=(i,)) for i in range(N)]
[t.start() for t in th]
[t.join() for t in th]
dt = time.time() - t0
bad = 0
for i in range(N):
    same = text(res[i]) == alone[i]
    bad += not same
    print(f"stream {i}: same_as_alone={same} completion_tokens={res[i]['usage'].get('completion_tokens')}")
    if not same:
        print("  alone :", alone[i][:300].replace("\n", " "))
        print("  conc. :", text(res[i])[:300].replace("\n", " "))
print(f"concurrent N={N}: {dt:.1f}s, {bad} differ")


def turn(hist, q):
    hist = hist + [{"role": "user", "content": q}]
    r = post(hist, 40)
    m = r["choices"][0]["message"]
    a = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        a["reasoning_content"] = m["reasoning_content"]
    return hist + [a], r["usage"]


A, B = [], []
for k, (qa, qb) in enumerate([("Name a river in Italy.", "Name a mountain in Peru."), ("And another one?", "And another one?"), ("One more?", "One more?")]):
    A, ua = turn(A, qa)
    B, ub = turn(B, qb)
    print(f"round {k+1}: A prompt={ua.get('prompt_tokens')} cached={cached(ua)} | B prompt={ub.get('prompt_tokens')} cached={cached(ub)}")
sys.exit(1 if bad else 0)
