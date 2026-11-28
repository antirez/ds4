#!/usr/bin/env python3
"""Live check of the three APIs against a ds4-server serving Inkling 2 Mini.

For every conversation in tests/test-vectors/inkling2-api: send it as chat completions, Anthropic messages and Responses,
non-streaming and streaming, at temperature 0.  Checks
  * parsed content, reasoning and tool calls of the Anthropic and Responses answers equal the chat answer,
  * every stream is well formed (event order) and its deltas add up to the same message as the non-stream body of the same API,
  * incremental tool-call deltas on chat completions add up to the parsed call.
    python3 tests/inkling2_server/api_equiv.py [--port 8000] [--only NN] [--out result.json]
"""
import argparse, glob, http.client, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--only", default="")
ap.add_argument("--out", default="")
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-vectors", "inkling2-api"))
args = ap.parse_args()


def post(path, body, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
    c.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    data = r.read().decode("utf-8")
    c.close()
    if r.status != 200:
        raise RuntimeError("%s -> %d %s" % (path, r.status, data[:300]))
    return data


def sse_events(data):
    out = []
    for blk in data.split("\n\n"):
        ev, dat = None, []
        for line in blk.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                dat.append(line[6:])
        if dat:
            d = "\n".join(dat)
            out.append((ev, d if d == "[DONE]" else json.loads(d)))
    return out


def norm_args(a):
    return json.loads(a) if isinstance(a, str) else a


# ---- result = dict(content, reasoning, calls=[(name, args_obj)], finish)

def chat_result(body):
    m = json.loads(body)["choices"][0]
    msg = m["message"]
    calls = [(c["function"]["name"], norm_args(c["function"]["arguments"])) for c in msg.get("tool_calls") or []]
    return dict(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "", calls=calls, finish=m["finish_reason"])


def chat_stream_result(data):
    ev = sse_events(data)
    problems = []
    content, reasoning, finish = "", "", None
    calls = {}
    seen_done = False
    first = True
    order_ok = True
    for _, d in ev:
        if d == "[DONE]":
            seen_done = True
            continue
        if seen_done:
            problems.append("data after [DONE]")
        if "choices" not in d or not d["choices"]:
            continue
        ch = d["choices"][0]
        dl = ch.get("delta", {})
        if first:
            first = False
            if dl.get("role") != "assistant":
                problems.append("first chunk has no role")
        content += dl.get("content") or ""
        reasoning += dl.get("reasoning_content") or ""
        for tc in dl.get("tool_calls") or []:
            i = tc["index"]
            e = calls.setdefault(i, {"name": None, "args": "", "id": None, "first_args_before_name": False})
            f = tc.get("function", {})
            if "name" in f:
                if e["name"] is not None:
                    problems.append("name sent twice for call %d" % i)
                e["name"] = f["name"]
                e["id"] = tc.get("id")
            if f.get("arguments"):
                if e["name"] is None:
                    problems.append("arguments before name for call %d" % i)
                e["args"] += f["arguments"]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
    if not seen_done:
        problems.append("no [DONE]")
    if sorted(calls) != list(range(len(calls))):
        problems.append("call indexes %s" % sorted(calls))
    cl = [(calls[i]["name"], json.loads(calls[i]["args"] or "{}")) for i in sorted(calls)]
    n_arg_chunks = sum(1 for _, d in ev if d != "[DONE]" and d.get("choices") and any(
        tc.get("function", {}).get("arguments") for tc in d["choices"][0].get("delta", {}).get("tool_calls") or []))
    return dict(content=content, reasoning=reasoning, calls=cl, finish=finish), problems, n_arg_chunks


def anth_result(body):
    m = json.loads(body)
    content = "".join(b["text"] for b in m["content"] if b["type"] == "text")
    reasoning = "".join(b["thinking"] for b in m["content"] if b["type"] == "thinking")
    calls = [(b["name"], b["input"]) for b in m["content"] if b["type"] == "tool_use"]
    fin = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(m["stop_reason"], m["stop_reason"])
    return dict(content=content, reasoning=reasoning, calls=calls, finish=fin)


def anth_stream_result(data):
    ev = sse_events(data)
    problems = []
    names = [e for e, _ in ev]
    if not names or names[0] != "message_start":
        problems.append("first event is %s" % (names[0] if names else None))
    if names[-2:] != ["message_delta", "message_stop"]:
        problems.append("last events %s" % names[-2:])
    open_idx, next_idx = None, 0
    blocks = {}
    content = reasoning = ""
    stop = None
    for e, d in ev:
        if d.get("type") != e:
            problems.append("event %s has type %s" % (e, d.get("type")))
        if e == "content_block_start":
            if open_idx is not None:
                problems.append("block %d started while %d open" % (d["index"], open_idx))
            if d["index"] != next_idx:
                problems.append("block index %d expected %d" % (d["index"], next_idx))
            open_idx = d["index"]
            next_idx += 1
            blocks[open_idx] = dict(type=d["content_block"]["type"], text="", sig=False, cb=d["content_block"])
        elif e == "content_block_delta":
            if d["index"] != open_idx:
                problems.append("delta for block %s while %s open" % (d["index"], open_idx))
                continue
            b = blocks[open_idx]
            dt = d["delta"]["type"]
            want = {"thinking": "thinking_delta", "text": "text_delta", "tool_use": "input_json_delta"}[b["type"]]
            if dt == "signature_delta" and b["type"] == "thinking":
                b["sig"] = True
            elif dt != want:
                problems.append("delta %s in %s block" % (dt, b["type"]))
            else:
                b["text"] += d["delta"].get("thinking", d["delta"].get("text", d["delta"].get("partial_json", "")))
        elif e == "content_block_stop":
            if d["index"] != open_idx:
                problems.append("stop for block %s while %s open" % (d["index"], open_idx))
            b = blocks.get(open_idx)
            if b and b["type"] == "thinking" and not b["sig"]:
                problems.append("thinking block without signature")
            open_idx = None
        elif e == "message_delta":
            if open_idx is not None:
                problems.append("message_delta with a block open")
            stop = d["delta"]["stop_reason"]
    calls = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "text":
            content += b["text"]
        elif b["type"] == "thinking":
            reasoning += b["text"]
        else:
            calls.append((b["cb"]["name"], json.loads(b["text"] or "{}")))
    fin = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(stop, stop)
    return dict(content=content, reasoning=reasoning, calls=calls, finish=fin), problems


def resp_result(body):
    m = json.loads(body)
    return resp_from_output(m["output"], m["status"])


def resp_from_output(output, status):
    content = reasoning = ""
    calls = []
    for it in output:
        if it["type"] == "message":
            content += "".join(c["text"] for c in it["content"] if c["type"] == "output_text")
        elif it["type"] == "reasoning":
            reasoning += "".join(s["text"] for s in it["summary"])
        elif it["type"] == "function_call":
            calls.append((it["name"], norm_args(it["arguments"])))
    fin = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
    return dict(content=content, reasoning=reasoning, calls=calls, finish=fin)


def resp_stream_result(data):
    ev = sse_events(data)
    problems = []
    names = [d["type"] for _, d in ev]
    if not names or names[0] != "response.created":
        problems.append("first event %s" % (names[0] if names else None))
    seqs = [d.get("sequence_number") for _, d in ev]
    if seqs != list(range(len(seqs))):
        problems.append("sequence_number not 0..n-1")
    if names[-1] not in ("response.completed", "response.incomplete", "response.failed"):
        problems.append("last event %s" % names[-1])
    items = {}
    done_items = {}
    content = reasoning = ""
    for _, d in ev:
        t = d["type"]
        if t == "response.output_item.added":
            items[d["output_index"]] = d["item"]
        elif t == "response.output_item.done":
            if d["output_index"] not in items:
                problems.append("item done without added")
            done_items[d["output_index"]] = d["item"]
        elif t == "response.output_text.delta":
            content += d["delta"]
        elif t == "response.reasoning_summary_text.delta":
            reasoning += d["delta"]
    final = ev[-1][1]["response"]
    r_final = resp_from_output(final["output"], final["status"])
    if content != r_final["content"]:
        problems.append("text deltas != final output text")
    if reasoning != r_final["reasoning"]:
        problems.append("reasoning deltas != final reasoning")
    for i, it in items.items():
        if i not in done_items:
            problems.append("item %d never done" % i)
    idxs = sorted(items)
    if idxs != list(range(len(idxs))):
        problems.append("output_index %s" % idxs)
    return r_final, problems


def same(a, b, what, probs, strict_finish=True):
    for k in ("content", "reasoning", "calls"):
        if a[k] != b[k]:
            probs.append("%s: %s differs: %r vs %r" % (what, k, str(a[k])[:120], str(b[k])[:120]))
    if strict_finish and a["finish"] != b["finish"]:
        probs.append("%s: finish %s vs %s" % (what, a["finish"], b["finish"]))


results = []
files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
for f in files:
    doc = json.load(open(f))
    if args.only and args.only not in doc["name"]:
        continue
    t0 = time.time()
    probs = []
    r = {}
    try:
        chat = doc["chat"]
        anth = doc["anthropic"]
        resp = doc["responses"]
        r["chat"] = chat_result(post("/v1/chat/completions", chat))
        r["anth"] = anth_result(post("/v1/messages", anth))
        r["resp"] = resp_result(post("/v1/responses", resp))
        same(r["chat"], r["anth"], "anthropic vs chat", probs)
        same(r["chat"], r["resp"], "responses vs chat", probs)
        cs, p, nchunks = chat_stream_result(post("/v1/chat/completions", dict(chat, stream=True)))
        probs += ["chat stream: " + x for x in p]
        same(r["chat"], cs, "chat stream vs body", probs)
        if cs["calls"] and nchunks < 2:
            probs.append("chat stream: tool args arrived in %d chunk(s)" % nchunks)
        as_, p = anth_stream_result(post("/v1/messages", dict(anth, stream=True)))
        probs += ["anthropic stream: " + x for x in p]
        same(r["anth"], as_, "anthropic stream vs body", probs)
        rs, p = resp_stream_result(post("/v1/responses", dict(resp, stream=True)))
        probs += ["responses stream: " + x for x in p]
        same(r["resp"], rs, "responses stream vs body", probs)
    except Exception as e:
        probs.append("EXC %r" % (e,))
    results.append(dict(name=doc["name"], problems=probs, secs=round(time.time() - t0, 1),
                        finish=r.get("chat", {}).get("finish"), calls=len(r.get("chat", {}).get("calls", []))))
    print("%-40s %s (%.1fs, finish=%s, calls=%d)" % (doc["name"], "OK" if not probs else "FAIL", time.time() - t0,
                                                 results[-1]["finish"], results[-1]["calls"]), flush=True)
    for p in probs:
        print("    ", p, flush=True)

ok = sum(1 for x in results if not x["problems"])
print("conversations: %d, all checks passed: %d" % (len(results), ok))
if args.out:
    json.dump(results, open(args.out, "w"), indent=1)
sys.exit(0 if ok == len(results) else 1)
