#!/usr/bin/env python3
"""Cancel under batching.  N streaming requests at temperature 0; stream CANCEL is dropped after 1 s; the others must finish with the
text they give when run alone, and the cancelled one's slot must serve a later request (same text as alone).  usage: cancel_check.py [--port 8000] [N=4] [MAXTOK=120]"""
import json, sys, threading, time, urllib.request
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; MT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
TOP = ["the Roman aqueducts", "how tides work", "the printing press", "photosynthesis", "the Silk Road", "transistors", "plate tectonics", "the telegraph"]


def body(i, stream):
    return json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": f"Write a paragraph about {TOP[i % 8]}."}],
                       "temperature": 0, "max_tokens": MT, "stream": stream}).encode()


def alone(i):
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, body(i, False), {"Content-Type": "application/json"}), timeout=600).read())
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def streamed(i, out, cancel_after=None):
    resp = urllib.request.urlopen(urllib.request.Request(URL, body(i, True), {"Content-Type": "application/json"}), timeout=600)
    t0 = time.time(); txt = ""
    for line in resp:
        if cancel_after and time.time() - t0 > cancel_after:
            resp.close(); out[i] = None; return
        if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
            try:
                d = json.loads(line[6:])["choices"][0].get("delta") or {}
            except (ValueError, IndexError, KeyError):
                continue
            txt += (d.get("reasoning_content") or "") + (d.get("content") or "")
    out[i] = txt


ref = [alone(i) for i in range(N)]
out = [None] * N
CANCEL = 1
th = [threading.Thread(target=streamed, args=(i, out, 1.0 if i == CANCEL else None)) for i in range(N)]
[t.start() for t in th]; [t.join() for t in th]
bad = 0
for i in range(N):
    if i == CANCEL:
        print(f"stream {i}: cancelled after 1 s"); continue
    ok = out[i] == ref[i]; bad += not ok
    print(f"stream {i}: same as alone = {ok} ({len(out[i] or '')} chars)")
again = alone(CANCEL)
ok = again == ref[CANCEL]; bad += not ok
print(f"cancelled stream's request run again: same as alone = {ok}")
sys.exit(1 if bad else 0)
