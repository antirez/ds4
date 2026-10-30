#!/usr/bin/env python3
"""Live vision-server checks for live-state slot routing and the tool-output
continuation contract.

Requires an idle, image-capable server: a vision model plus its encoder, with
at least four resident sessions for the eviction section:

    ./ds4-server -m <vision-model>.gguf --vision <encoder>.gguf \
        --batched-session 4 --trace /tmp/ds4-live-state-trace.log \
        > /tmp/ds4-live-state.log 2>&1

    python3 tests/test_server_vision_live_state.py --url http://127.0.0.1:8080 \
        --model deepseek-v4-flash --section eviction --log /tmp/ds4-live-state.log

Sections:
  eviction  Conversations that own a checkpoint keep their slots: a request
            that reuses nothing takes an empty slot while one exists, and once
            every slot is occupied the shortest fresh checkpoint is evicted
            (reported in the server log) -- never the long image-conditioned
            conversation, and neither by an unrelated request nor by a new
            image conversation.
  responses-tool-users  Qwen tool outputs followed by user messages retain the
                        validated live frontier; checks 0/1/9 users and guards.
  tool-409  A tool-result-only continuation has no replayable history, so when
            its live frontier cannot be rebuilt the server must answer 409
            instead of generating from a context-free prompt.

These sections are real-model checks; they assert on usage counters (cached
tokens), the server log, and HTTP status, not on generated text quality.
"""

import argparse
import base64
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

FIXTURES = Path(__file__).resolve().parent / "vision-fixtures/glm53"


def image_part(name, api):
    data = base64.b64encode((FIXTURES / name).read_bytes()).decode()
    if api == "anthropic":
        return {"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": data}}
    if api == "responses":
        return {"type": "input_image", "image_url": "data:image/png;base64," + data}
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def post(url, path, body, timeout=300):
    """POST JSON, returning (status, parsed_body_or_raw_text)."""
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw}


def words(count, prefix="word"):
    return " ".join("%s%d" % (prefix, i) for i in range(count))


def archive(lines):
    return "\n".join("Archive record %d: the build passed with no warnings." % i
                     for i in range(lines))


def chat(args, messages, label, max_tokens=16):
    """Run one chat turn and print its cache accounting."""
    status, reply = post(args.url, "/v1/chat/completions", {
        "model": args.model, "messages": messages, "temperature": 0,
        "reasoning_effort": "none", "max_tokens": max_tokens})
    assert status == 200, (label, status, reply)
    usage = reply["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    print("  %-24s prompt=%-6d cached=%-6d" % (
        label, usage["prompt_tokens"], cached), flush=True)
    return reply["choices"][0]["message"], cached, usage


def continue_chat(args, messages, answer, label, prompt, frontier):
    """Append an assistant turn plus a user turn; require the previous live
    frontier to remain fully reusable."""
    messages = messages + [answer, {"role": "user", "content": prompt}]
    answer, cached, usage = chat(args, messages, label)
    assert cached >= frontier, \
        "%s: lost part of the live prefix (cached=%d, frontier=%d)" % (
            label, cached, frontier)
    return answer, cached, usage


def logged_evictions(path):
    """Token counts of checkpoints the router logged as evicted."""
    if not path:
        return []
    log = Path(path).read_text(errors="replace")
    return [int(m.group(1)) for m in re.finditer(
        r"reuses nothing; evicting checkpoint \((\d+) tokens", log)]


def eviction_section(args):
    """Image/text conversations must keep their resident checkpoints."""
    # A: long image conversation -- the expensive checkpoint that must survive.
    history = [{"role": "system", "content": "Answer briefly."},
               {"role": "user", "content": [
                   {"type": "text", "text": archive(600) + "\nReply with exactly READY."},
                   image_part("text.png", "chat")]}]
    a1, cached, usage = chat(args, history, "A1-image")
    assert cached == 0, "A1: a fresh image conversation cannot reuse a checkpoint"
    a2, cached, usage = continue_chat(args, history, a1, "A2-image",
                                      "Reply with exactly AGAIN.",
                                      usage["total_tokens"])
    a2_frontier = usage["total_tokens"]
    a2_messages = history + [a1, {"role": "user", "content": "Reply with exactly AGAIN."}]

    # One unrelated conversation while three slots are still empty: it must take
    # an empty slot instead of evicting the image conversation.
    alien = [{"role": "user", "content": "Reply with exactly ALIEN."}]
    alien1, cached, usage = chat(args, alien, "alien1")
    assert cached == 0, "alien1: an unrelated request reused a checkpoint"
    assert not logged_evictions(args.log), \
        "an unrelated request evicted a checkpoint while empty slots existed"
    _, _, alien_usage = continue_chat(
        args, alien, alien1, "alien1-again", "Reply with exactly ALIEN.",
        usage["total_tokens"])

    a3, cached, usage = continue_chat(args, a2_messages, a2, "A3-image",
                                      "Reply with exactly STILL.", a2_frontier)
    a_frontier = usage["total_tokens"]
    a3_messages = a2_messages + [a2, {"role": "user", "content": "Reply with exactly STILL."}]

    # Include every competing resident: the alien conversation can be smaller
    # than short0, depending on the model's tokenizer and generated answer.
    short_frontiers = [alien_usage["total_tokens"]]
    for label, count in (("short0", 8), ("short1", 512)):
        messages = [{"role": "user",
                     "content": words(count) + " Reply with exactly OK."}]
        answer, cached, usage = chat(args, messages, label)
        assert cached == 0, "%s: a fresh conversation cannot reuse a checkpoint" % label
        _, cached, usage = continue_chat(args, messages, answer, label + "-again",
                                        "Reply with exactly OK.", usage["total_tokens"])
        short_frontiers.append(usage["total_tokens"])
    shortest = min(short_frontiers)

    # Every slot is occupied now: the next unrelated request must evict the
    # shortest fresh checkpoint, and the log must say so.
    before = len(logged_evictions(args.log))
    _, cached, _ = chat(args, [{"role": "user", "content": "Reply with exactly ALIEN2."}],
                        "alien2")
    assert cached == 0, "alien2: an unrelated request reused a checkpoint"
    evictions = logged_evictions(args.log)
    if args.log:
        print("  eviction victims: %s (shortest fresh=%d, image=%d)" % (
            evictions, shortest, a_frontier), flush=True)
        assert len(evictions) > before, "no eviction was logged with every slot occupied"
        assert evictions[-1] == shortest, \
            "the forced eviction picked %d tokens, not the shortest fresh checkpoint %d" % (
                evictions[-1], shortest)
        assert all(count < a_frontier for count in evictions), \
            "the image conversation was chosen as an eviction victim"

    # A new image conversation must not displace the resident image one either.
    other = [{"role": "user", "content": [
        {"type": "text", "text": words(32, "other") + " Reply with exactly OTHER."},
        image_part("spatial.png", "chat")]}]
    _, cached, _ = chat(args, other, "other-image")
    assert cached == 0, "other-image: a different image reused an incompatible prefix"
    continue_chat(args, a3_messages, a3, "A4-image",
                  "Reply with exactly FINAL.", a_frontier)
    if args.log:
        evictions = logged_evictions(args.log)
        assert all(count < a_frontier for count in evictions), \
            "a new image conversation evicted the resident image conversation"
    print("PASS: live-state routing kept image and text checkpoints resident", flush=True)


def tool_409_section(args):
    """Tool-result-only continuations must 409 when the frontier cannot rebuild."""
    tools = [{"name": "echo", "description": "Echo a note back to the caller.",
              "input_schema": {"type": "object", "properties": {
                  "note": {"type": "string"}}, "required": ["note"]}}]
    image = image_part("text.png", "anthropic")
    system = "Call the echo tool whenever asked; never answer in text."

    # 1. A tool call produced while the live checkpoint is image-conditioned.
    status, reply = post(args.url, "/v1/messages", {
        "model": args.model, "max_tokens": 256, "temperature": 0,
        "thinking": {"type": "disabled"}, "tools": tools, "system": system,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Call the echo tool with note=seen."},
            image]}]})
    assert status == 200, (status, reply)
    calls = [c for c in reply.get("content", []) if c.get("type") == "tool_use"]
    assert calls, "model did not emit a tool_use block: %s" % reply
    call_id = calls[0]["id"]
    print("  tool call %s on an image-conditioned checkpoint" % call_id, flush=True)

    # 2. Tool-result-only continuation carrying the same image.  The call id and
    #    frontier match, but the historical image now sits inside the tool-result
    #    tail, so the live prefix cannot be rebuilt: the request must be rejected
    #    instead of answered from the tool output alone.  Without that contract
    #    the server returns 200 and generates from a context-free prompt.
    status, reply = post(args.url, "/v1/messages", {
        "model": args.model, "max_tokens": 64, "temperature": 0,
        "thinking": {"type": "disabled"}, "tools": tools,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id,
             "content": [{"type": "text", "text": "echo: seen"}, image]}]}]})
    print("  tool-result-only with image -> HTTP %d" % status, flush=True)
    assert status == 409, (
        "expected 409 for an unreconstructable tool-output continuation, got %d: %s"
        % (status, json.dumps(reply)[:400]))
    assert "continuation state is not available" in json.dumps(reply), reply

    # 3. Positive control: the same shape with a text-only tool result must still
    #    continue from the live frontier (the normal local agent fast path).
    status, reply = post(args.url, "/v1/messages", {
        "model": args.model, "max_tokens": 256, "temperature": 0,
        "thinking": {"type": "disabled"}, "tools": tools, "system": system,
        "messages": [{"role": "user",
                      "content": "Call the echo tool with note=plain."}]})
    assert status == 200, (status, reply)
    calls = [c for c in reply.get("content", []) if c.get("type") == "tool_use"]
    assert calls, "model did not emit a tool_use block: %s" % reply
    status, reply = post(args.url, "/v1/messages", {
        "model": args.model, "max_tokens": 64, "temperature": 0,
        "thinking": {"type": "disabled"}, "tools": tools,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": calls[0]["id"],
             "content": "echo: plain"}]}]})
    assert status == 200, (status, reply)
    cached = reply["usage"].get("cache_read_input_tokens", 0)
    print("  text-only tool-result continuation cached=%d" % cached, flush=True)
    assert cached > 0, "text-only tool-result continuation lost the live frontier"
    print("PASS: tool-result-only continuations 409 when the frontier is "
          "unreconstructable and reuse it otherwise", flush=True)


def responses_tool_users_section(args):
    """Omit tool-turn reasoning to require ID reuse, rather than visible replay.

    Each case obtains a fresh real tool-call frontier with one retained image.
    No tool is executed. Only new results/users are appended; negative controls
    deliberately change the earlier parent or image. No server is launched.
    """
    assert args.log, "responses-tool-users requires --log to verify the reuse tier"
    tools = [{"type": "function", "name": "echo", "description": "Echo a note.",
              "parameters": {"type": "object", "properties": {
                  "note": {"type": "string"}}, "required": ["note"],
                  "additionalProperties": False}}]
    failures = []

    def run(history, label):
        offset = args.log.stat().st_size if args.log else 0
        started = time.monotonic()
        status, reply = post(args.url, "/v1/responses", {
            "model": args.model, "input": history, "tools": tools,
            "reasoning": {"effort": "medium", "summary": "auto"},
            "stream": False, "store": False, "temperature": 0,
            "max_output_tokens": 1024})
        wall = time.monotonic() - started
        assert status == 200 and reply.get("status") == "completed", (
            label, status, json.dumps(reply)[:500])
        usage = reply["usage"]
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        log = ""
        if args.log:
            with args.log.open("rb") as source:
                source.seek(offset)
                log = source.read().decode(errors="replace")
        match = re.search(r"responses live continuation .*match=([^ ]+)", log)
        prefill = re.search(r"prompt done ([0-9.]+)s", log)
        print(json.dumps({"case": label, "cached_tokens": cached,
                          "input_tokens": usage["input_tokens"],
                          "evaluated_tokens": usage["input_tokens"] - cached,
                          "output_tokens": usage["output_tokens"],
                          "wall_seconds": wall,
                          "prefill_seconds": float(prefill[1]) if prefill else None,
                          "tier": match[1] if match else None}), flush=True)
        return reply, cached, log

    for count, change in ((0, None), (1, None), (9, None),
                          (1, "parent"), (1, "image")):
        label = "users-%d%s" % (count, "-changed-" + change if change else "")
        parent = [{"role": "developer", "content":
                   "When asked, call echo exactly once. After its result, call "
                   "no tool; reply with the newest NOTICE marker, or OK if none. "
                   "Do not describe the image."},
                  {"role": "user", "content": [
                      {"type": "input_text", "text": "Call echo with note=ready."},
                      image_part("text.png", "responses")]}]
        first, _, _ = run(parent, label + "-call")
        calls = [item for item in first["output"] if item.get("type") == "function_call"]
        assert len(calls) == 1 and calls[0]["name"] == "echo", label
        assert any(item.get("type") == "reasoning" and item.get("summary")
                   for item in first["output"]), "need emitted reasoning to omit"
        frontier = first["usage"]["input_tokens"] + first["usage"]["output_tokens"]
        history = json.loads(json.dumps(parent))
        history.extend(item for item in first["output"]
                       if item.get("type") != "reasoning")
        history.append({"type": "function_call_output", "call_id": calls[0]["call_id"],
                        "output": "Echo result: ready. Reply briefly now."})
        for i in range(count):
            history.append({"role": "user", "content": [{"type": "input_text",
                "text": "Background notification %d: NOTICE%d." % (i + 1, i + 1)}]})
        assert history[:len(parent)] == parent
        if change == "parent":
            history[0]["content"] += " Changed earlier instruction: preserve this edit."
        elif change == "image":
            history[1]["content"][1] = image_part("spatial.png", "responses")
        reply, cached, log = run(history, label + "-result")
        should_reuse = change is None
        if should_reuse and cached < frontier:
            failures.append("%s: cached=%d below frontier=%d" % (label, cached, frontier))
        if should_reuse and args.log and not re.search(
                r"responses live continuation .*match=tool-output-ids", log):
            failures.append(label + ": missing tool-output-ID tier")
        if not should_reuse and (cached != 0 or "match=tool-output-ids" in log):
            failures.append(label + ": changed parent/image reused the old frontier")
        if should_reuse and count:
            text = " ".join(part.get("text", "") for item in reply["output"]
                            if item.get("type") == "message"
                            for part in item.get("content", []))
            if "NOTICE%d" % count not in text:
                failures.append(label + ": final user notification not reflected in answer")
    assert not failures, "\n".join(failures)
    print("PASS: tool results retain queued users and reject changed parent/image", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--section", choices=["eviction", "tool-409", "responses-tool-users"], required=True)
    parser.add_argument("--log", type=Path,
                        help="server log; required for Responses tool-user reuse assertions")
    args = parser.parse_args()
    if args.section == "eviction":
        eviction_section(args)
    elif args.section == "responses-tool-users":
        responses_tool_users_section(args)
    else:
        tool_409_section(args)


if __name__ == "__main__":
    main()
