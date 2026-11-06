#!/usr/bin/env python3
"""Two-turn SSE retrieval/tool regression against an already-running server.

Example: python3 tests/test_server_speculative_stop.py --output /tmp/stop-check
Use --approx-tokens 16384 for the longer story; sizes are character proxies,
not token counts. Requires Linux/macOS SIGALRM. No server startup or tool/code
execution. Each case echoes the actual assistant reply before its follow-up.
Raw requests, SSE bytes/events, replies, timings and failures are retained.
TTFT counts nonempty content/reasoning/tool deltas, not role-only events.
The tail is last such delta to body EOF. Network buffering affects timings;
this is not an internal-state oracle or a scalar free-generation comparison.
"""

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request

from generate_long_context_story_prompt import FACTS, OPENING, SCENE_TEMPLATES, assignment_sentence

MAX_BYTES = 2 * 1024 * 1024


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(text):
    def pairs(items):
        result = dict(items)
        check(len(result) == len(items), "duplicate JSON keys")
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: check(False, "nonfinite JSON: " + value))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def equal_json(actual, expected):
    check(json.dumps(actual, sort_keys=True, allow_nan=False) ==
          json.dumps(expected, sort_keys=True, allow_nan=False), "JSON value/type mismatch")


def story(size):
    scenes, chars = [], 0
    while chars < size * 4:
        i = len(scenes)
        scene = SCENE_TEMPLATES[i % len(SCENE_TEMPLATES)].format(
            lead=FACTS[i % len(FACTS)][0], friend=FACTS[(i + 5) % len(FACTS)][0])
        scenes.append(scene)
        chars += len(scene)
    for i, (name, word, _) in enumerate(FACTS):
        scenes[i * (len(scenes) - 1) // (len(FACTS) - 1)] += assignment_sentence(name, word) + "\n"
    return OPENING + "\n".join(scenes)


def post(args, body, directory):
    directory.mkdir()
    save(directory / "request.json", body)
    timing, message, calls = {}, {"role": "assistant", "content": ""}, {}
    result = {"timing": timing, "message": message, "text_events": 0, "started_unix_s": time.time()}
    start = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("absolute HTTP deadline exceeded")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        url = args.url.rstrip("/")
        url += "/chat/completions" if url.endswith("/v1") else "/v1/chat/completions"
        request = urllib.request.Request(url, json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Connection": "close"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(request, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response, (directory / "response.sse").open("wb") as raw, \
                (directory / "events.jsonl").open("w", encoding="utf-8") as events:
            result["http_status"] = response.code
            timing["headers_s"] = time.monotonic() - start
            if response.code != 200 or response.headers.get_content_type() != "text/event-stream":
                raw.write(response.read(MAX_BYTES + 1))
                raise ValueError("expected HTTP 200 text/event-stream; see response.sse")
            received, pending, done = 0, [], False
            while True:
                line = response.readline(min(65536, MAX_BYTES - received + 1))
                now = time.monotonic() - start
                if not line:
                    timing["eof_s"] = now
                    break
                raw.write(line)
                received += len(line)
                check(received <= MAX_BYTES and line.endswith(b"\n"), "oversize/truncated SSE line")
                if line.startswith(b"data:"):
                    pending.append(line[5:].strip())
                if line.strip() or not pending:
                    continue
                data, pending = b"\n".join(pending).decode("utf-8"), []
                events.write(json.dumps({"seconds": now, "data": data}) + "\n")
                check(not done, "data after [DONE]")
                if data == "[DONE]":
                    check("finish" in result and "usage" in result, "premature [DONE]")
                    done = True
                    timing["done_s"] = now
                    continue
                event = load_json(data)
                check("error" not in event, "SSE error: " + repr(event.get("error")))
                result["id"] = event.get("id", result.get("id"))
                if event.get("usage") is not None:
                    check("usage" not in result, "duplicate usage")
                    result["usage"] = event["usage"]
                choices = event["choices"]
                if not choices:
                    continue
                check(len(choices) == 1 and choices[0]["index"] == 0, "unexpected choices")
                delta = choices[0]["delta"]
                check(not delta or "finish" not in result, "delta after finish")
                check(delta.get("role", "assistant") == "assistant", "wrong role")
                check(not (set(delta) - {"role", "content", "reasoning_content", "tool_calls"}), "unknown delta")
                meaningful = False
                for key in ("content", "reasoning_content"):
                    value = delta.get(key)
                    if value is None:
                        continue
                    check(isinstance(value, str), "non-string text delta")
                    message[key] = message.get(key, "") + value
                    meaningful |= bool(value)
                    if key == "content" and value:
                        result["text_events"] += 1
                        timing.setdefault("content_ttft_s", now)
                        timing["last_text_s"] = now
                parts = delta.get("tool_calls", [])
                check(isinstance(parts, list), "non-array tool delta")
                for part in parts:
                    check(part["index"] == 0, "expected only one tool call")
                    call = calls.setdefault(0, {"function": {"name": "", "arguments": ""}})
                    for key in ("id", "type"):
                        if key in part:
                            check(key not in call or call[key] == part[key], "changed tool identity")
                            call[key] = part[key]
                    for key, value in part.get("function", {}).items():
                        check(key in ("name", "arguments") and isinstance(value, str), "invalid tool fragment")
                        call["function"][key] += value
                    meaningful |= bool(part.get("id") or any(part.get("function", {}).values()))
                if meaningful:
                    timing.setdefault("ttft_s", now)
                    timing["last_delta_s"] = now
                if choices[0].get("finish_reason") is not None:
                    check("finish" not in result, "duplicate finish")
                    result["finish"] = choices[0]["finish_reason"]
                    timing["finish_s"] = now
            check(done and not pending and "last_delta_s" in timing, "incomplete/empty SSE reply")
            timing["final_delay_s"] = timing["eof_s"] - timing["last_delta_s"]
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        timing["elapsed_s"] = time.monotonic() - start
        if calls:
            message["tool_calls"] = list(calls.values())
        save(directory / "response.json", result)
    return result


def run_case(args, kind, records):
    expected = {name: number for name, _, number in FACTS}
    prompt = story(args.approx_tokens)
    prompt += ('\nCall lookup exactly once with key="amber". Do not invent its result.' if kind == "tool" else
               '\nReturn all assignments as one JSON object, names to integers. No prose or fences.')
    body = dict(model=args.model, think=False, temperature=args.temperature, seed=7102026, max_tokens=512,
                stream=True, stream_options={"include_usage": True}, messages=[{"role": "user", "content": prompt}])
    if kind == "tool":
        body["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"],
            "additionalProperties": False}}}]
    for turn in (1, 2):
        result = post(args, body, args.output / (kind + "-" + str(turn)))
        records.append(dict(case=kind, turn=turn, **result))
        usage, message = result["usage"], result["message"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            check(type(usage[key]) is int and usage[key] >= 0, "invalid token usage")
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        check(type(cached) is int and 0 <= cached <= usage["prompt_tokens"], "invalid cached_tokens")
        check(0 < usage["completion_tokens"] <= 512 and usage["prompt_tokens"] > 0, "token budget violation")
        check(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "usage sum mismatch")
        check(result["timing"]["final_delay_s"] <= args.max_final_delay, "final delay exceeded")
        if turn == 1:
            first_prompt = usage["prompt_tokens"]
        else:
            check(cached >= first_prompt - 8, "follow-up did not reuse the original prefix")
        tool_turn = kind == "tool" and turn == 1
        check(result["finish"] == ("tool_calls" if tool_turn else "stop"), "unexpected finish reason")
        body["messages"].append(message)
        if tool_turn:
            calls = message.get("tool_calls", [])
            check(len(calls) == 1, "expected one tool call")
            call = calls[0]
            check(call.get("type") == "function" and isinstance(call.get("id"), str) and call["id"], "invalid tool ID/type")
            check(call["function"]["name"] == "lookup", "wrong tool name")
            equal_json(load_json(call["function"]["arguments"]), {"key": "amber"})
            body["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": '{"value":731}'})
            followup, expected = 'Return only {"value":N}, using the tool result. Do not call tools again.', {"value": 731}
        else:
            check(not message.get("tool_calls"), "unexpected tool call")
            equal_json(load_json(message["content"]), expected)
            followup = "Change Alice to 17 and Owen to 42. Return the entire ledger as JSON, no prose or fences."
            expected = dict(expected, Alice=17, Owen=42)
        body["messages"].append({"role": "user", "content": followup})
        print(json.dumps({"case": kind, "turn": turn, "usage": usage, "timing": result["timing"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("all", "retrieval", "tool"), default="all")
    parser.add_argument("--approx-tokens", type=int, choices=(4096, 16384), default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-final-delay", type=float, default=5)
    args = parser.parse_args()
    check(0 <= args.temperature <= 2, "temperature must be finite and in 0..2")
    check(0 < args.max_final_delay <= args.timeout <= 86400, "invalid timing limits")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"turns": [], "errors": []}
    for kind in (("retrieval", "tool") if args.case == "all" else (args.case,)):
        try:
            run_case(args, kind, report["turns"])
        except Exception as exc:
            report["errors"].append({"case": kind, "error": str(exc)})
    save(args.output / "report.json", report)
    print(json.dumps({"ok": not report["errors"], "errors": report["errors"]}), flush=True)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    sys.exit(main())
