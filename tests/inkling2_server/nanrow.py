#!/usr/bin/env python3
"""HTTP check: a server started with DS4_INKLING2_TEST_POISON_ROW=1 --batched-session 4 poisons row 1 of every batched decode step.
Three concurrent requests: with the fix only the request in the poisoned row fails, the others finish; before, the whole step failed.
usage: nanrow.py [--port 8000]  -> prints the status of each request; exit 0 when at least one succeeded and at least one failed."""
import http.client, json, sys, threading
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
res = {}
def go(i):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "inkling-2-mini", "max_tokens": 60, "temperature": 0, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "Count from %d to %d, one number per line." % (i * 10, i * 10 + 40)}]}), {"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read()
    try: j = json.loads(d)
    except Exception: j = {"raw": d[:200].decode("utf8", "replace")}
    res[i] = (r.status, (j.get("usage") or {}).get("completion_tokens"), (j.get("choices") or [{}])[0].get("finish_reason"), str(j.get("error"))[:160])
th = [threading.Thread(target=go, args=(i,)) for i in range(3)]
[t.start() for t in th]; [t.join() for t in th]
for i in sorted(res): print("request", i, res[i])
ok = sum(1 for v in res.values() if v[0] == 200 and v[2] in ("stop", "length")); bad = 3 - ok
print("succeeded %d, failed %d" % (ok, bad))
sys.exit(0 if ok >= 1 and bad >= 1 else 1)
