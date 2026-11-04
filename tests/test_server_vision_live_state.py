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
  tool-409  A tool-result-only continuation has no replayable history, so when
            its live frontier cannot be rebuilt the server must answer 409
            instead of generating from a context-free prompt.
  responses-final  Qwen streaming tool/result/final-answer replay with six
                   retained images. Checks final reasoning preserved, omitted,
                   or not requested (--reasoning-replay). Needs one idle slot.

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


def responses_final_section(args):
    """Replay the Responses wire shape used by Pi without an SDK dependency.

    A tool result adds four images to the first two, then a completed final
    answer is followed by a user turn with no new images. Tool-ID reuse can
    leave canonical token/image offsets different from the sampled live KV;
    final reasoning replay must still match the Responses-visible frontier.
    """
    tools = [{"type": "function", "name": "bg_status",
              "description": "Check a background task.", "strict": False,
              "parameters": {"type": "object", "properties": {
                  "taskId": {"type": "string"}}, "required": ["taskId"],
                  "additionalProperties": False}}]
    history = [{"role": "developer", "content":
                "Respond briefly. When explicitly asked to check a task, call "
                "bg_status. After a completed task result, give a brief final "
                "answer. Read images only when asked."},
               {"role": "user", "content": [
                   {"type": "input_text", "text": "Check task demo with bg_status."},
                   image_part("text.png", "responses"),
                   image_part("spatial.png", "responses")]}]
    previous_input = []
    frontier = 0

    def run(label):
        nonlocal previous_input, frontier
        serialized = [json.dumps(item, sort_keys=True) for item in history]
        assert serialized[:len(previous_input)] == previous_input, \
            label + ": preceding input changed"
        previous_input = serialized
        reasoning = {"effort": "medium"}
        if args.reasoning_replay != "summary-off":
            reasoning["summary"] = "auto"
        body = {"model": args.model, "input": history, "tools": tools,
                "reasoning": reasoning, "stream": True, "store": False,
                "temperature": 0, "max_output_tokens": 4096}
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/responses",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        started = time.monotonic()
        first_delta = None
        reply = None
        with urllib.request.urlopen(request, timeout=300) as stream:
            for raw in stream:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    continue
                event = json.loads(data)
                kind = event.get("type", "")
                if first_delta is None and kind.endswith(".delta"):
                    first_delta = time.monotonic() - started
                assert kind not in ("error", "response.failed"), event
                if kind in ("response.completed", "response.incomplete"):
                    reply = event["response"]
        assert reply and reply.get("status") == "completed", (label, reply)
        usage = reply["usage"]
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        print(json.dumps({"mode": args.reasoning_replay, "turn": label,
                          "input_items": len(history), "cached_tokens": cached,
                          "input_tokens": usage["input_tokens"],
                          "evaluated_tokens": usage["input_tokens"] - cached,
                          "first_delta_seconds": first_delta,
                          "wall_seconds": time.monotonic() - started}), flush=True)
        assert cached >= frontier, (label, "lost live frontier", cached, frontier)
        frontier = usage["input_tokens"] + usage["output_tokens"]
        return reply["output"]

    first = run("tool-call-with-images")
    calls = [item for item in first if item.get("type") == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "bg_status", first
    history.extend(first)
    # Synthetic tool result: no background task is launched or executed.
    history.append({"type": "function_call_output", "call_id": calls[0]["call_id"],
                    "output": [{"type": "input_text", "text":
                                "Task demo completed successfully. Give a brief final answer now."}]
                    + [image_part(name, "responses") for name in
                       ("text.png", "spatial.png", "text.png", "spatial.png")]})
    final = run("tool-result-to-final")
    assert any(item.get("type") == "message" for item in final), final
    assert not any(item.get("type") == "function_call" for item in final), final
    if args.reasoning_replay != "summary-off":
        assert any(item.get("type") == "reasoning" and item.get("summary")
                   for item in final), "final answer did not emit a reasoning summary"
    if args.reasoning_replay == "omit":
        final = [item for item in final if item.get("type") != "reasoning"]
    history.extend(final)
    history.append({"role": "user", "content": [{"type": "input_text", "text":
                    "Continue the active /goal until completion or a genuine blocker. "
                    "Current goal #9: Check task demo with bg_status.\n\n[goal-continuation:9]"}]})
    log_offset = args.log.stat().st_size if args.log else 0
    run("continue-after-final")
    if args.log:
        with args.log.open("rb") as log:
            log.seek(log_offset)
            lines = log.read().decode(errors="replace")
        assert re.search(r"responses live continuation .*match=visible-prefix", lines), \
            "final continuation did not use the Responses-visible live tier: " + lines[-2000:]
    print("PASS: final-answer replay retains six image prefixes (" +
          args.reasoning_replay + ")", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--section", choices=["eviction", "tool-409", "responses-final"], required=True)
    parser.add_argument("--reasoning-replay", choices=["preserve", "omit", "summary-off"],
                        default="preserve", help="Responses final-answer replay mode")
    parser.add_argument("--log", type=Path,
                        help="server log, for eviction or Responses-visible tier assertions")
    args = parser.parse_args()
    if args.section == "eviction":
        eviction_section(args)
    elif args.section == "responses-final":
        responses_final_section(args)
    else:
        tool_409_section(args)


if __name__ == "__main__":
    main()
