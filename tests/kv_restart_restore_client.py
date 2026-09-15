#!/usr/bin/env python3
"""Client for tests/kv_restart_restore_smoke.sh (issue #1053).

  conversation BASE_URL OUT.json   run conversation A (warm-up) then B (tools);
                                   save B's history to OUT.json
  replay BASE_URL HIST.json        resend B's history + one user turn, print
                                   "prompt=N cached=M" on stdout

Only the Python standard library is used.
"""
import json
import sys
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the machine sorts "
    "tokens into aligned buckets and the cache layer measures the frontier. "
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def make_filler(seed, approx_bytes):
    parts = [FILLER]
    total = len(FILLER)
    i = 0
    while total < approx_bytes:
        chunk = f"Segment {seed}-{i}: " + FILLER * 4
        parts.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(parts)


def chat(base, messages, tools=None, max_tokens=512):
    body = {
        "model": "ds4",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.load(r)
    usage = data.get("usage", {})
    choice = data["choices"][0]
    print(
        f"  turn: prompt={usage.get('prompt_tokens')} "
        f"cached={usage.get('prompt_tokens_details', {}).get('cached_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"finish={choice.get('finish_reason')}",
        file=sys.stderr, flush=True,
    )
    return choice["message"], usage


def run_conversation(base, name, system, users, tools=None):
    messages = [{"role": "system", "content": system}]
    for user in users:
        messages.append({"role": "user", "content": user})
        print(f"[{name}] user turn ({len(user)} bytes)", file=sys.stderr, flush=True)
        msg, _ = chat(base, messages, tools=tools)
        if msg.get("tool_calls"):
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": "18C, light rain."})
            print(f"[{name}] tool result", file=sys.stderr, flush=True)
            msg, _ = chat(base, messages, tools=tools)
        # like every OpenAI-compatible client: content only, no reasoning_content
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
    return messages


def main():
    cmd, base = sys.argv[1], sys.argv[2].rstrip("/")
    if cmd == "conversation":
        run_conversation(
            base, "A",
            "You are Alpha, a terse logistics planner. Answer in one short paragraph.",
            [make_filler(1, 26000), "Summarize the constraints above in two sentences."],
        )
        history = run_conversation(
            base, "B",
            "You are Beta, a terse auditor. Use the get_weather tool when asked "
            "about weather. Answer in one short paragraph.",
            [make_filler(2, 46000), "What is the weather in Paris right now?",
             make_filler(3, 8000), make_filler(4, 8000)],
            tools=[WEATHER_TOOL],
        )
        with open(sys.argv[3], "w") as f:
            json.dump(history, f)
    elif cmd == "replay":
        with open(sys.argv[3]) as f:
            history = json.load(f)
        history.append({"role": "user",
                        "content": "Now give a one sentence verdict on all of it."})
        _, usage = chat(base, history, tools=[WEATHER_TOOL])
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"prompt={usage.get('prompt_tokens', 0)} cached={cached}")
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
