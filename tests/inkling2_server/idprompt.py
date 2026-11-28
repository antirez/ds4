#!/usr/bin/env python3
"""Check: a /v1/completions request that sends token ids must reach the model with exactly those ids, also when a live session or a
disk KV file holds the same text under other (canonical) token ids.  The reuse path matched the text the ids spell and swapped in the
stored tokens, so the model saw other ids than the ones sent.
Probe: usage.prompt_tokens.  The ids sent are the canonical ids of text T with one token split in two (same text, one id more); a swapped
prompt has the stored canonical count.
usage: idprompt.py BUILD_DIR MODEL.gguf [--port 8000]  (own temp KV dirs, starts and stops its own ds4-server; lock file $DS4_LOCK_FILE)
Phase live: T as text, then the split ids.  Phase disk: T as text, another prompt (evicts the live state), then the split ids.
Phase same: like disk, but the canonical ids: the file must still be used (a hit), the request's ids equal its tokens."""
import socket, http.client, json, os, random, re, signal, subprocess, sys, tempfile, time
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
build, model = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp(prefix="ds4-idprompt-"); lock = os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock")
W = "river stone cloud engine garden market winter copper violin harbor ladder window pencil forest anchor bridge candle".split()
def text(seed, n=1500):
    r = random.Random(seed); return " ".join(r.choice(W) for _ in range(n))
def tokens(s):
    """canonical ids of s as the server's completions path tokenizes it: ds4 --dump-tokens wraps it in a chat turn, strip the wrapper"""
    out = subprocess.run(["./ds4", "-m", model, "--metal", "--dump-tokens", "-p", s], cwd=build, env=dict(os.environ, DS4_LOCK_FILE=lock + "-cli"),
                         capture_output=True, text=True).stdout.splitlines()
    rows = [(int(m.group(1)), m.group(2)) for m in (re.match(r"\s*(\d+)  (.*)$", l) for l in out) if m]
    a = max(i for i, (_, t) in enumerate(rows) if t == "<|content_text|>")
    z = max(i for i, (_, t) in enumerate(rows) if t == "<|end_message|>")
    return rows[a + 1:z]
T = text(11)
canon = tokens(T); ids_c = [i for i, _ in canon]
space = tokens(" ")[0][0]; k = next(i for i in range(20, 200) if canon[i][1].startswith("Ġ") and canon[i][1][1:] in W)
word = tokens(canon[k][1][1:].replace("Ġ", ""))[0][0]
split = ids_c[:k] + [space, word] + ids_c[k + 1:]
assert split != ids_c and len(split) == len(ids_c) + 1
print("canonical %d ids, split ids %d, split at %d (%s)" % (len(ids_c), len(split), k, canon[k][1]))
def start(kv):
    log = os.path.join(tmp, "srv.log")
    p = subprocess.Popen(["./ds4-server", "-m", model, "--metal",
        "--host", "127.0.0.1", "--port", str(port), "--ctx", "8192", "--kv-disk-dir", kv, "--kv-disk-space-mb", "3000"],
        cwd=build, env=dict(os.environ, DS4_LOCK_FILE=lock), stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        time.sleep(1)
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/v1/models"); c.getresponse().read(); return p, log
        except Exception:
            if p.poll() is not None: sys.exit("server died: " + open(log).read()[-500:])
    sys.exit("server start timeout")
def stop(p):
    p.send_signal(signal.SIGTERM)
    try: p.wait(120)
    except Exception: p.kill()
    for _ in range(120):
        try: socket.create_connection(("127.0.0.1", port), 1).close()
        except OSError: break
        time.sleep(1)
    else: sys.exit("server did not stop")
    time.sleep(2)
def ask(prompt):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/completions", json.dumps({"model": "inkling-2-mini", "prompt": prompt, "max_tokens": 2, "temperature": 0}), {"Content-Type": "application/json"})
    r = c.getresponse(); d = json.loads(r.read()); assert r.status == 200, d
    return d["usage"]["prompt_tokens"]
tail = tokens(" river stone cloud")[:3]; tail_ids = [i for i, _ in tail]
bad = []
for phase in ("live", "disk", "same"):
    kv = os.path.join(tmp, "kv-" + phase); os.mkdir(kv)
    p, log = start(kv)
    n_a = ask(T)
    if phase != "live":
        ask(text(12)); time.sleep(1)
    sent = (ids_c if phase == "same" else split) + tail_ids
    got = ask(sent)
    stop(p)
    srv = open(log).read()
    print("phase %s: text prompt %d tokens; ids sent %d, model saw %d  (disk hits in log: %d, refused: %d)" % (phase, n_a, len(sent), got, srv.count("kv cache hit text"), srv.count("refused a file")))
    if got != len(sent): bad.append(phase)
    if phase == "same" and (srv.count("refused a file") or srv.count("kv cache hit text") < 1): bad.append("same-ids hit lost")
print("PASS" if not bad else "FAIL (ids swapped in phase %s)" % ",".join(bad)); print("logs in", tmp)
sys.exit(1 if bad else 0)
