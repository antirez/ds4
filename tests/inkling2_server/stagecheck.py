#!/usr/bin/env python3
"""Where does a KV save stage its payload?

Against a RUNNING Inkling server with --kv-disk-dir DIR.  Sends --n distinct 1,500-word prompts (each stores a checkpoint) and
polls every 2 ms for (a) /tmp/ds4-session-payload.* (the old staging area) and (b) DIR/.ds4-stage-* (the new one).
Exit 0 when nothing was ever seen in /tmp and at least one staged file was seen in DIR (so the poll can see the stage at all).

usage: stagecheck.py [--port 8000] --dir DIR [--n 4]
"""
import argparse, glob, http.client, json, os, random, sys, threading, time

WORDS = ("river stone cloud engine garden market winter copper violin harbor ladder window pencil forest anchor bridge candle "
         "dragon feather glacier hammer island jungle kettle lantern meadow needle orchard palace quartz ribbon saddle tunnel").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    seen_tmp, seen_dir, stop = set(), set(), [False]
    tmp_before = set(glob.glob("/tmp/ds4-session-payload.*"))

    def poll():
        while not stop[0]:
            for f in glob.glob("/tmp/ds4-session-payload.*"):
                if f not in tmp_before:
                    seen_tmp.add(f)
            for f in glob.glob(os.path.join(a.dir, ".ds4-stage-*")):
                seen_dir.add(os.path.basename(f))
            time.sleep(0.002)

    th = threading.Thread(target=poll)
    th.start()
    try:
        for i in range(a.n):
            r = random.Random(500 + i)
            text = " ".join(r.choice(WORDS) for _ in range(1500))
            c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
            c.request("POST", "/v1/chat/completions", json.dumps({"model": "inkling-2-mini", "max_tokens": 8, "temperature": 0, "reasoning_effort": "none",
                                                                  "messages": [{"role": "user", "content": "Stage %d.\n%s\nSay ok." % (i, text)}]}),
                      {"Content-Type": "application/json"})
            resp = c.getresponse()
            resp.read()
            print("request %d: status %d" % (i, resp.status))
            time.sleep(1.5)
    finally:
        stop[0] = True
        th.join()
    print("staged files seen in /tmp: %d, inside the kv dir: %d" % (len(seen_tmp), len(seen_dir)))
    left = glob.glob(os.path.join(a.dir, ".ds4-stage-*"))
    print("left behind in the kv dir: %d" % len(left))
    return 0 if not seen_tmp and seen_dir and not left else 1


if __name__ == "__main__":
    sys.exit(main())
