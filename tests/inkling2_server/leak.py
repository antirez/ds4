#!/usr/bin/env python3
"""Cross-conversation leak test for the kept prompt state.

Against a RUNNING Inkling server (any --batched-session setting).
Conversation A holds a secret; its follow-up must answer A's secret.  Before each follow-up another client (C, a different
long conversation with another secret) breaks its prefill, then A's follow-up is asked again.  The answer must still be A's
secret, never C's.

  --mode cancel : C is a streaming request and the client drops the connection during prefill (after --drop seconds)
  --mode fail   : C's prefill FAILS inside the server (NaN check).  Needs a server started with
                  DS4_INKLING2_TEST_POISON_LOGITS=/path/to/flagfile ; the script creates and removes that file.

usage: leak.py [--port 8000] [--trials 10] [--mode cancel|fail] [--poison-file FILE]
Exit code 0 when 0 leaks.
"""
import argparse, http.client, json, os, random, socket, sys, time

WORDS = ("river stone cloud engine garden market winter copper violin harbor ladder window pencil forest anchor bridge candle "
         "dragon feather glacier hammer island jungle kettle lantern meadow needle orchard palace quartz ribbon saddle tunnel "
         "umbrella valley whistle yellow zephyr basket castle desert emerald fabric golden helmet ivory jacket kitten lemon").split()


def filler(seed, nwords):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(nwords)) + "."


def post(port, body, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    try:
        return r.status, json.loads(d)
    except Exception:
        return r.status, {"raw": d[:300].decode("utf8", "replace")}


def chat(port, msgs, max_tokens=300):
    st, j = post(port, {"model": "inkling-2-mini", "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                        "reasoning_effort": "none"})
    if st != 200:
        return st, "", j
    m = j["choices"][0]["message"]
    return st, (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), j


def stream_and_drop(port, msgs, after_s):
    b = {"model": "inkling-2-mini", "messages": msgs, "max_tokens": 200, "temperature": 0, "stream": True, "reasoning_effort": "none"}
    s = socket.create_connection(("127.0.0.1", port))
    data = json.dumps(b).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(data) + data)
    s.settimeout(after_s)
    t0 = time.time()
    try:
        while time.time() - t0 < after_s:
            if not s.recv(65536):
                break
    except socket.timeout:
        pass
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--mode", choices=["cancel", "fail"], default="cancel")
    ap.add_argument("--poison-file", default="/tmp/ds4-leak-poison.flag")
    ap.add_argument("--words-c", type=int, default=5000, help="size of conversation C (about 1.3 tokens per word)")
    a = ap.parse_args()
    A = [{"role": "user", "content": "Remember this: the vault code is PURPLE-HORSE-42.\n\n" + filler(1, 900) + "\n\nSummarise the text above in one sentence."}]
    st, ans, j = chat(a.port, A)
    assert st == 200, j
    ans_c = j["choices"][0]["message"].get("content") or ""
    F = A + [{"role": "assistant", "content": ans_c}, {"role": "user", "content": "What is the vault code? Answer with the code only."}]
    st, ref, j = chat(a.port, F, 400)
    assert st == 200 and "PURPLE-HORSE-42" in ref, ("reference follow-up does not know its own secret", ref)
    print("reference ok: A tokens %s, answer %r" % (j["usage"]["prompt_tokens"], ref[-40:]))
    leaks = bad = 0
    rng = random.Random(7)
    for t in range(a.trials):
        chat(a.port, A)    # A replayed: the kept state is A's prompt end
        C = [{"role": "user", "content": "Remember this: the vault code is GREEN-TIGER-17.\n\n" + filler(100 + t, a.words_c) + "\n\nSummarise the text above."}]
        if a.mode == "cancel":
            stream_and_drop(a.port, C, rng.choice([0.5, 1.0, 1.5, 3.0]))
            time.sleep(3)
        else:
            open(a.poison_file, "w").close()
            try:
                st, jc = post(a.port, {"model": "inkling-2-mini", "messages": C, "max_tokens": 50, "temperature": 0, "reasoning_effort": "none"})
            finally:
                os.unlink(a.poison_file)
            if st == 200:
                print("trial %d: WARNING the poisoned prefill did not fail (server without the poison hook?)" % t)
        st, ans, j = chat(a.port, F, 400)
        leak = st != 200 or "GREEN" in ans.upper() or "TIGER" in ans.upper() or "PURPLE-HORSE-42" not in ans
        leaks += leak
        print("trial %d: status %d cached %s answer %r %s" % (t, st, (j.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"), ans[-40:], "LEAK" if leak else "ok"))
    print("leaks: %d of %d trials (mode %s)" % (leaks, a.trials, a.mode))
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
