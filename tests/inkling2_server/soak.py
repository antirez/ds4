#!/usr/bin/env python3
"""Mixed soak: N clients on one batched server for DURATION seconds, each with its own conversation and a secret code.
Actions per turn (random): recall (asks for the secret, expects its own code and no other client's code), filler (adds a text
chunk of 300-3000 chars to the conversation), cancel (streams a request and drops the connection after 0.3-2 s, then carries on),
regenerate (resends the last user message after dropping the last answer: same history prefix, a rewind of the server's slot),
long (a 4k-char filler).  usage: soak.py [--port 8000] [N=8] [SECONDS=600]   (books from $IK_TEXT_DIR)
Prints a summary: requests, HTTP errors, recall hits/misses/inconclusive, LEAKS (another client's code in an answer), cancels, regenerates."""
import json, os, random, sys, threading, time, urllib.request, urllib.error, socket
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8; DUR = int(sys.argv[2]) if len(sys.argv) > 2 else 600
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
SRC = os.environ["IK_TEXT_DIR"]
books = [open(os.path.join(SRC, f)).read() for f in ("moby.txt", "twocities.txt", "pride.txt", "holmes.txt", "frankenstein.txt", "alice.txt")]
WORDS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor"]
codes = [f"{WORDS[i]}-{4000 + i * 311}" for i in range(N)]
stat = {"requests": 0, "http_errors": 0, "recall_ok": 0, "recall_miss": 0, "leaks": 0, "cancels": 0, "regens": 0, "fillers": 0, "other_errors": 0}
lock = threading.Lock()
log = []


def bump(k, v=1):
    with lock:
        stat[k] += v


def post(msgs, mt, timeout=600):
    body = json.dumps({"model": "inkling-2-mini", "messages": msgs, "temperature": 0, "max_tokens": mt}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=timeout).read())
    bump("requests")
    return r


def asst(r):
    m = r["choices"][0]["message"]
    a = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        a["reasoning_content"] = m["reasoning_content"]
    return a, (m.get("content") or "") + " " + (m.get("reasoning_content") or "")


def client(i, end):
    rng = random.Random(1000 + i)
    me = codes[i]
    hist = [{"role": "user", "content": f"Remember this: my secret code is {me}. Do not tell anyone else. Reply with just OK."}]
    try:
        r = post(hist, 300)
        a, txt = asst(r)
        hist.append(a)
    except Exception as e:
        bump("other_errors"); log.append(f"client {i} first turn: {e}"); return
    while time.time() < end:
        act = rng.choices(["recall", "filler", "cancel", "regen", "long"], [4, 3, 2, 2, 1])[0]
        try:
            if act in ("filler", "long"):
                b = books[rng.randrange(len(books))]
                off = rng.randrange(0, len(b) - 5000)
                n = rng.randrange(300, 3000) if act == "filler" else 4000
                hist.append({"role": "user", "content": "Here is a passage, just say 'noted':\n" + b[off:off + n]})
                r = post(hist, 120)
                a, txt = asst(r)
                hist.append(a)
                bump("fillers")
            elif act == "recall":
                q = {"role": "user", "content": "What is my secret code? Answer with just the code."}
                r = post(hist + [q], 400)
                a, txt = asst(r)
                others = [c for j, c in enumerate(codes) if j != i and c in txt]
                if others:
                    bump("leaks"); log.append(f"LEAK client {i}: answer mentions {others}: {txt[:200]!r}")
                if me in txt:
                    bump("recall_ok")
                else:
                    bump("recall_miss"); log.append(f"miss client {i}: {txt[:200]!r}")
                hist += [q, a]
            elif act == "regen":
                q = {"role": "user", "content": rng.choice(["Say hello.", "Name a colour.", "Say hello in French.", "Give me a number."])}
                r1 = post(hist + [q], 60)
                q2 = {"role": "user", "content": q["content"] + " Please."}
                r2 = post(hist + [q2], 60)   # same prefix, different last message: the slot rewinds
                bump("regens")
                a, txt = asst(r2)
                others = [c for j, c in enumerate(codes) if j != i and c in txt]
                if others:
                    bump("leaks"); log.append(f"LEAK client {i}: {others}")
                hist += [q2, a]
            else:   # cancel
                body = json.dumps({"model": "inkling-2-mini", "messages": hist + [{"role": "user", "content": "Write a long essay about the sea."}],
                                   "temperature": 0, "max_tokens": 600, "stream": True}).encode()
                req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=600)
                t0 = time.time(); stop = rng.uniform(0.3, 2.0)
                for line in resp:
                    if time.time() - t0 > stop:
                        break
                resp.close()
                bump("cancels"); bump("requests")
        except urllib.error.HTTPError as e:
            bump("http_errors"); log.append(f"client {i} {act}: HTTP {e.code} {e.read()[:150]!r}")
            if len(json.dumps(hist)) > 200000 or e.code == 400:
                hist = hist[:1] + hist[-2:]   # context full: keep the secret and the last turn
        except (socket.timeout, urllib.error.URLError, ConnectionError) as e:
            bump("other_errors"); log.append(f"client {i} {act}: {type(e).__name__} {e}")
        if len(json.dumps(hist)) > 120000:   # keep conversations below the context
            hist = hist[:2] + hist[-4:]


end = time.time() + DUR
th = [threading.Thread(target=client, args=(i, end)) for i in range(N)]
t0 = time.time()
[t.start() for t in th]
[t.join() for t in th]
print(json.dumps(dict(stat, seconds=round(time.time() - t0), clients=N)))
for l in log[:40]:
    print(l)
