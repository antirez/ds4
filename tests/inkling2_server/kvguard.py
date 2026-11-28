#!/usr/bin/env python3
"""KV directory owner guard, HTTP check.

Run against a RUNNING Inkling server whose --kv-disk-dir is DIR.  DIR must be a directory of your own (copies of files, never a
live server's directory).  The script fingerprints every file of another model that DIR holds before the first request, sends
--n distinct long prompts (each makes the server store a checkpoint, which also evicts down to its cap), then checks that every
foreign file is still there with the same bytes.  Exit 0 = guard holds.

usage: kvguard.py [--port 8000] --dir DIR [--n 3]
"""
import argparse, glob, hashlib, http.client, json, os, random, sys, time

WORDS = ("river stone cloud engine garden market winter copper violin harbor ladder window pencil forest anchor bridge candle "
         "dragon feather glacier hammer island jungle kettle lantern meadow needle orchard palace quartz ribbon saddle tunnel").split()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--own-id", type=int, default=64, help="model id of the server under test (header byte 7)")
    a = ap.parse_args()
    foreign = {}
    for f in glob.glob(os.path.join(a.dir, "*.kv")):
        if open(f, "rb").read(48)[7] != a.own_id:
            foreign[f] = md5(f)
    print("foreign files before: %d" % len(foreign))
    if not foreign:
        print("nothing to protect: put copies of another model's .kv files into the directory first")
        return 2
    for i in range(a.n):
        r = random.Random(1000 + i)
        text = " ".join(r.choice(WORDS) for _ in range(1400))
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
        c.request("POST", "/v1/chat/completions", json.dumps({"model": "inkling-2-mini", "max_tokens": 8, "temperature": 0, "reasoning_effort": "none",
                                                              "messages": [{"role": "user", "content": "Conversation %d.\n%s\nSay ok." % (i, text)}]}),
                  {"Content-Type": "application/json"})
        resp = c.getresponse()
        resp.read()
        print("request %d: status %d" % (i, resp.status))
        time.sleep(2)
    own = [f for f in glob.glob(os.path.join(a.dir, "*.kv")) if open(f, "rb").read(48)[7] == a.own_id]
    print("own files now: %d (%d MiB)" % (len(own), sum(os.path.getsize(f) for f in own) >> 20))
    lost = [f for f, h in foreign.items() if not os.path.exists(f) or md5(f) != h]
    print("foreign files lost or changed: %d of %d" % (len(lost), len(foreign)))
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
