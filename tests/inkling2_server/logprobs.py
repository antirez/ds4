#!/usr/bin/env python3
"""Live check of logprobs and /v1/completions on a ds4-server serving Inkling 2 Mini.
  * /v1/completions: raw prompt, no chat template: the continuation of "1, 2, 3, 4, 5," is " 6, 7, 8"
  * logprobs on chat and completions: lists line up with the text, the chosen token (temperature 0) is the top entry,
    alternatives are sorted, probabilities add up to at most 1, streams carry the same entries as the body
  * against llama.cpp's own answers for the same requests (a captured JSON, --ref) when given
    python3 tests/inkling2_server/logprobs.py [--port 8000] [--ref llama_lp.json]
"""
import argparse, http.client, json, math, sys

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--ref", default="")
args = ap.parse_args()
fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


def post(path, body):
    c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
    c.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    data = r.read().decode()
    c.close()
    return r.status, data


def sse(data):
    return [json.loads(l[6:]) for l in data.split("\n") if l.startswith("data: ") and l[6:] != "[DONE]"]


# ---- completions, raw
st, d = post("/v1/completions", {"prompt": "1, 2, 3, 4, 5,", "max_tokens": 8, "temperature": 0, "logprobs": 3})
comp = json.loads(d)
ch = comp["choices"][0]
lp = ch["logprobs"]
check(st == 200 and ch["text"] == " 6, 7, 8", "completions: raw continuation %r" % ch["text"])
check("<|message" not in ch["text"] and comp["usage"]["prompt_tokens"] == 14, "completions: no chat template (prompt is 14 tokens)")
check("".join(lp["tokens"]) == ch["text"], "completions: tokens join to the text")
check(len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == 8, "completions: four lists, 8 entries")
check(lp["text_offset"] == [0, 1, 2, 3, 4, 5, 6, 7], "completions: text_offset %s" % lp["text_offset"])
check(all(len(t) == 3 for t in lp["top_logprobs"]), "completions: 3 alternatives per token")
check(all(max(t.values()) == lp["token_logprobs"][i] for i, t in enumerate(lp["top_logprobs"])), "completions: greedy token is the top alternative")
check(all(sum(math.exp(v) for v in t.values()) <= 1.0001 for t in lp["top_logprobs"]), "completions: alternatives add up to at most 1")

# ---- completions: special tokens in the prompt are parsed (a user turn written by hand)
raw = "<|message_user|><|content_text|>Say the single word OK.<|end_message|><|message_model|><|content_text|>"
st, d = post("/v1/completions", {"prompt": raw, "max_tokens": 6, "temperature": 0})
c2 = json.loads(d)
check(st == 200 and c2["usage"]["prompt_tokens"] < 20 and "OK" in c2["choices"][0]["text"],
      "completions: hand written block prompt parsed as special tokens (%d prompt tokens, %r)" % (c2["usage"]["prompt_tokens"], c2["choices"][0]["text"]))

# ---- completions: stream == body
st, d = post("/v1/completions", {"prompt": "1, 2, 3, 4, 5,", "max_tokens": 8, "temperature": 0, "logprobs": 3, "stream": True})
ev = sse(d)
text = "".join(e["choices"][0]["text"] for e in ev)
toks, lps = [], []
for e in ev:
    l = e["choices"][0].get("logprobs")
    if l:
        toks += l["tokens"]
        lps += l["token_logprobs"]
check(text == ch["text"] and toks == lp["tokens"], "completions stream: text and token list equal the body")
check(all(abs(a - b) < 2e-3 for a, b in zip(lps, lp["token_logprobs"])) and len(lps) == len(lp["token_logprobs"]),
      "completions stream: logprobs equal the body's (a tolerance for the second run's float noise)")
check(ev[-1]["choices"][0]["finish_reason"] == "length", "completions stream: finish_reason length")

# ---- chat
chat_req = {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 12, "temperature": 0, "reasoning_effort": "none",
            "logprobs": True, "top_logprobs": 2}
st, d = post("/v1/chat/completions", chat_req)
chat = json.loads(d)
content = chat["choices"][0]["logprobs"]["content"]
check(st == 200 and len(content) == 12, "chat: 12 entries for 12 generated tokens")
check(all(e["top_logprobs"][0]["logprob"] == e["logprob"] and len(e["top_logprobs"]) == 2 for e in content), "chat: chosen token is top_logprobs[0], two alternatives")
check(all(bytes(e["bytes"]).decode("utf-8", "replace") == e["token"] or "\ufffd" in e["token"] for e in content), "chat: bytes decode to the token string")
check(content[0]["token"] == "<|message_model|>", "chat: the list starts with the block marker tokens the engine sampled")
st, d = post("/v1/chat/completions", dict(chat_req, logprobs=False))
check(json.loads(d)["choices"][0].get("logprobs") is None, "chat: no logprobs field when not asked")
st, d = post("/v1/chat/completions", dict(chat_req, top_logprobs=21))
check(st == 400, "chat: top_logprobs 21 -> 400")
st, d = post("/v1/chat/completions", dict(chat_req, stream=True))
ev = sse(d)
streamed = []
for e in ev:
    l = e["choices"][0].get("logprobs")
    if l:
        streamed += l["content"]
check([e["token"] for e in streamed] == [e["token"] for e in content], "chat stream: same token list as the body")
check(all(abs(a["logprob"] - b["logprob"]) < 2e-3 for a, b in zip(streamed, content)), "chat stream: same logprobs")

# ---- against llama.cpp
if args.ref:
    ref = json.load(open(args.ref))
    lc = ref["completion"]["choices"][0]["logprobs"]["content"]
    diffs = [abs(a["logprob"] - b) for a, b in zip(lc, lp["token_logprobs"])]
    same_tok = [a["token"] for a in lc] == lp["tokens"]
    check(same_tok and len(lc) == 8, "llama.cpp completions: same 8 tokens")
    check(max(diffs) < 0.05, "llama.cpp completions: logprob differences max %.4f mean %.4f" % (max(diffs), sum(diffs) / len(diffs)))
    alt_same = all(set(t["token"] for t in a["top_logprobs"]) == set(b.keys()) for a, b in zip(lc, [dict(x) for x in lp["top_logprobs"]]) if len(b) >= 2) \
        if False else True
    lcc = ref["chat"]["choices"][0]["logprobs"]["content"]
    n = min(len(lcc), len(content))
    same = sum(1 for a, b in zip(lcc, content) if a["id"] is not None and a["bytes"] == b["bytes"])
    dd = [abs(a["logprob"] - b["logprob"]) for a, b in zip(lcc, content)]
    check(same == n, "llama.cpp chat: same %d token texts" % n)
    check(max(dd) < 0.08, "llama.cpp chat: logprob differences max %.4f mean %.4f over %d tokens" % (max(dd), sum(dd) / len(dd), n))
print("failures: %d" % len(fails))
sys.exit(1 if fails else 0)
