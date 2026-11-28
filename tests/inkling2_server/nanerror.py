#!/usr/bin/env python3
"""Check: a request failed by the batch NaN check must be answered as an error, not as an empty success.
Server: DS4_INKLING2_TEST_POISON_ROW=1 ... --batched-session 4.  Four concurrent requests (OpenAI chat, OpenAI chat stream, Anthropic,
Anthropic stream) per round, the order rotates so every kind meets the poisoned row.  A failed request must be HTTP 500 with an error
message (non-stream) or carry an `error` event (stream); an HTTP 200 whose finish reason is "error" or whose content is empty is a failure.
usage: nanerror.py [--port 8000] [ROUNDS]  -> exit 0 when every kind failed at least once in the right shape and nothing failed in the old shape"""
import http.client, json, sys, threading, time
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
KINDS = ["chat", "chat_stream", "anth", "anth_stream"]
def req(kind, i):
    msg = "Count from %d to %d, one number per line." % (i * 10, i * 10 + 40)
    if kind.startswith("chat"):
        path = "/v1/chat/completions"; body = {"model": "inkling-2-mini", "max_tokens": 60, "temperature": 0, "reasoning_effort": "none", "stream": kind.endswith("stream"),
                                              "messages": [{"role": "user", "content": msg}]}
    else:
        path = "/v1/messages"; body = {"model": "inkling-2-mini", "max_tokens": 60, "temperature": 0, "stream": kind.endswith("stream"),
                                       "messages": [{"role": "user", "content": msg}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read().decode("utf8", "replace")
    return r.status, d
def verdict(kind, status, d):
    stream = kind.endswith("stream")
    if stream:
        if "event: error" in d:
            return "error-event" if '"message":"' in d.split("event: error", 1)[1] else "bad:error event without message"
        return "ok" if status == 200 and ('"finish_reason":"stop"' in d or '"finish_reason":"length"' in d or "message_stop" in d) and '"finish_reason":"error"' not in d else "bad:" + d[-200:]
    try: j = json.loads(d)
    except Exception: return "bad:not json"
    if status == 500: return "http500" if (j.get("error") or {}).get("message") else "bad:500 without message"
    if status != 200: return "bad:status %d" % status
    fin = (j.get("choices") or [{}])[0].get("finish_reason") or j.get("stop_reason")
    text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "".join(b.get("text", "") for b in j.get("content") or [])
    return "bad:200 finish=%s empty" % fin if fin in ("error",) or (not text and fin not in ("length", "max_tokens")) else "ok"
seen = {k: set() for k in KINDS}; bad = []
for rd in range(rounds):
    order = KINDS[rd % 4:] + KINDS[:rd % 4]; res = {}
    def go(i, k):
        time.sleep(0.15 * i); res[k] = req(k, i)
    th = [threading.Thread(target=go, args=(i, k)) for i, k in enumerate(order)]
    [t.start() for t in th]; [t.join() for t in th]
    for k in KINDS:
        v = verdict(k, *res[k]); seen[k].add(v)
        if v.startswith("bad"): bad.append((rd, k, v))
    print("round", rd, {k: verdict(k, *res[k]) for k in KINDS}, flush=True)
failed = {k: bool(seen[k] & {"error-event", "http500"}) for k in KINDS}
print("failed at least once:", failed, "bad:", bad)
sys.exit(0 if all(failed.values()) and not bad else 1)
