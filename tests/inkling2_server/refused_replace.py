#!/usr/bin/env python3
"""Check: a KV file the loader refused (written under another --ctx) is replaced by the next store, and the replacement is then
loaded.  Before the fix the refusal was remembered for the life of the process: the replacement was skipped by every lookup and the
whole payload was rewritten on every store.
usage: refused_replace.py BUILD_DIR MODEL.gguf [--port 8000]   (starts and stops its own ds4-server, own temp KV dir; lock file $DS4_LOCK_FILE)
Sequence: server A (--ctx 8192) stores P.  Server B (--ctx 16384): P (load refused, file replaced), Q, P again.
Passes when the last P is a disk hit and the file was replaced exactly once."""
import socket, http.client, json, os, random, subprocess, sys, tempfile, time, signal
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
build, model = sys.argv[1], sys.argv[2]
W = "river stone cloud engine garden market winter copper violin harbor ladder window pencil forest anchor bridge candle".split()
def prompt(seed):
    r = random.Random(seed); return " ".join(r.choice(W) for _ in range(1500)) + "\nSay ok."
tmp = tempfile.mkdtemp(prefix="ds4-refused-replace-"); kv = os.path.join(tmp, "kv"); os.mkdir(kv)
def start(ctx, tag):
    log = os.path.join(tmp, tag + ".log")
    env = dict(os.environ, DS4_LOCK_FILE=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"))
    p = subprocess.Popen(["./ds4-server", "-m", model, "--metal",
        "--host", "127.0.0.1", "--port", str(port), "--ctx", str(ctx), "--kv-disk-dir", kv, "--kv-disk-space-mb", "2000"],
        cwd=build, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        time.sleep(1)
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2); c.request("GET", "/v1/models"); c.getresponse().read(); return p, log
        except Exception:
            if p.poll() is not None: sys.exit("server died: " + open(log).read()[-500:])
    sys.exit("server start timeout")
def stop(p):
    # the server may still be saving its resident cache after SIGTERM: wait until the port is closed
    p.send_signal(signal.SIGTERM)
    try: p.wait(120)
    except Exception: p.kill()
    for _ in range(120):
        try: socket.create_connection(("127.0.0.1", port), 1).close()
        except OSError: break
        time.sleep(1)
    else: sys.exit("server did not stop")
    time.sleep(2)
def ask(text):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "inkling-2-mini", "max_tokens": 4, "temperature": 0, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": text}]}), {"Content-Type": "application/json"})
    r = c.getresponse(); r.read(); assert r.status == 200, r.status
P, Q = prompt(77), prompt(78)
a, la = start(8192, "a"); ask(P); stop(a)
b, lb = start(16384, "b")
ask(P); ask(Q); ask(P)
stop(b)
log = open(lb).read()
hits = log.count("kv cache hit"); replaced = log.count("replaced incompatible file")
print("server B: disk hits %d, replaced incompatible file %d" % (hits, replaced))
ok = hits >= 1 and replaced == 1
print("PASS" if ok else "FAIL"); print("logs in", tmp)
sys.exit(0 if ok else 1)
