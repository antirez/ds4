#!/usr/bin/env python3
"""Tool-use round trip through the `anthropic` and `openai` Python SDKs against a ds4-server serving Inkling 2 Mini.
Runs: Anthropic messages (create and stream), OpenAI chat completions (create and stream), OpenAI Responses (create and stream).
Each: ask for the weather with a tool declared -> the model calls it -> the result goes back -> the final answer names the result.
    python3 tests/inkling2_server/sdk_roundtrip.py [--port 8000]   (needs the anthropic and openai packages)
"""
import argparse, json, sys

import anthropic, openai

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
base = "http://127.0.0.1:%d" % args.port
fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
ASK = "What is the weather in Paris right now? Use the get_weather tool, then tell me the temperature."
RESULT = "It is 18 degrees Celsius with light rain."

# ---------------- Anthropic
a = anthropic.Anthropic(base_url=base, api_key="x")
tools = [{"name": "get_weather", "description": "Current weather for a city.", "input_schema": PARAMS}]
msgs = [{"role": "user", "content": ASK}]
r = a.messages.create(model="inkling-2-mini", max_tokens=800, temperature=0, tools=tools, messages=msgs,
                      thinking={"type": "enabled", "budget_tokens": 1024})
uses = [b for b in r.content if b.type == "tool_use"]
check(r.stop_reason == "tool_use" and len(uses) == 1 and uses[0].name == "get_weather" and "paris" in json.dumps(uses[0].input).lower(),
      "anthropic create: tool_use get_weather %s (blocks %s)" % (uses[0].input if uses else None, [b.type for b in r.content]))
check(any(b.type == "thinking" for b in r.content), "anthropic create: thinking block present")
msgs2 = msgs + [{"role": "assistant", "content": [b.model_dump() for b in r.content]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id, "content": RESULT}]}]
r2 = a.messages.create(model="inkling-2-mini", max_tokens=800, temperature=0, tools=tools, messages=msgs2)
txt = "".join(b.text for b in r2.content if b.type == "text")
check(r2.stop_reason == "end_turn" and "18" in txt, "anthropic create: final answer mentions 18: %r" % txt[:80])

with a.messages.stream(model="inkling-2-mini", max_tokens=800, temperature=0, tools=tools, messages=msgs,
                       thinking={"type": "enabled", "budget_tokens": 1024}) as s:
    n_events = sum(1 for _ in s)
    fm = s.get_final_message()
uses = [b for b in fm.content if b.type == "tool_use"]
check(fm.stop_reason == "tool_use" and len(uses) == 1 and uses[0].name == "get_weather" and isinstance(uses[0].input, dict),
      "anthropic stream: final message has the parsed tool_use %s after %d events" % (uses[0].input if uses else None, n_events))
msgs3 = msgs + [{"role": "assistant", "content": [b.model_dump() for b in fm.content]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0].id, "content": RESULT}]}]
with a.messages.stream(model="inkling-2-mini", max_tokens=800, temperature=0, tools=tools, messages=msgs3) as s:
    final_text = s.get_final_message()
txt = "".join(b.text for b in final_text.content if b.type == "text")
check("18" in txt, "anthropic stream: final answer mentions 18: %r" % txt[:80])

# ---------------- OpenAI chat completions
o = openai.OpenAI(base_url=base + "/v1", api_key="x")
ctools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": PARAMS}}]
cm = [{"role": "user", "content": ASK}]
c = o.chat.completions.create(model="inkling-2-mini", temperature=0, max_tokens=800, tools=ctools, messages=cm, reasoning_effort="low")
tc = c.choices[0].message.tool_calls
check(c.choices[0].finish_reason == "tool_calls" and tc and tc[0].function.name == "get_weather" and "paris" in tc[0].function.arguments.lower(),
      "chat create: tool call %s" % (tc[0].function.arguments if tc else None))
cm2 = cm + [{"role": "assistant", "content": c.choices[0].message.content or "", "tool_calls": [t.model_dump() for t in tc]},
            {"role": "tool", "tool_call_id": tc[0].id, "content": RESULT}]
c2 = o.chat.completions.create(model="inkling-2-mini", temperature=0, max_tokens=800, tools=ctools, messages=cm2, reasoning_effort="low")
check("18" in (c2.choices[0].message.content or ""), "chat create: final answer mentions 18: %r" % (c2.choices[0].message.content or "")[:80])

stream = o.chat.completions.create(model="inkling-2-mini", temperature=0, max_tokens=800, tools=ctools, messages=cm, stream=True, reasoning_effort="low")
name, args_s, tid, n_arg_chunks, first_with_name = None, "", None, 0, None
order_ok = True
for i, ch in enumerate(stream):
    for t in (ch.choices[0].delta.tool_calls or []) if ch.choices else []:
        if t.function.name:
            name, tid, first_with_name = t.function.name, t.id, i
        if t.function.arguments:
            if name is None:
                order_ok = False
            args_s += t.function.arguments
            n_arg_chunks += 1
check(name == "get_weather" and order_ok and n_arg_chunks > 1 and json.loads(args_s).get("city", "").lower() == "paris",
      "chat stream: name first, %d argument chunks, args %s" % (n_arg_chunks, args_s))
cm3 = cm + [{"role": "assistant", "content": "", "tool_calls": [{"id": tid, "type": "function", "function": {"name": name, "arguments": args_s}}]},
            {"role": "tool", "tool_call_id": tid, "content": RESULT}]
st = o.chat.completions.create(model="inkling-2-mini", temperature=0, max_tokens=800, tools=ctools, messages=cm3, stream=True, reasoning_effort="low")
txt = "".join((ch.choices[0].delta.content or "") for ch in st if ch.choices)
check("18" in txt, "chat stream: final answer mentions 18: %r" % txt[:80])

# ---------------- OpenAI Responses
rt = [{"type": "function", "name": "get_weather", "description": "Current weather for a city.", "parameters": PARAMS}]
ri = [{"role": "user", "content": ASK}]
r = o.responses.create(model="inkling-2-mini", temperature=0, max_output_tokens=800, tools=rt, input=ri, reasoning={"effort": "low", "summary": "auto"})
calls = [it for it in r.output if it.type == "function_call"]
check(len(calls) == 1 and calls[0].name == "get_weather" and "paris" in calls[0].arguments.lower(), "responses create: function_call %s" % (calls[0].arguments if calls else None))
ri2 = ri + [it.model_dump(exclude_none=True) for it in r.output] + [{"type": "function_call_output", "call_id": calls[0].call_id, "output": RESULT}]
r2 = o.responses.create(model="inkling-2-mini", temperature=0, max_output_tokens=800, tools=rt, input=ri2, reasoning={"effort": "low", "summary": "auto"})
check("18" in (r2.output_text or ""), "responses create: final answer mentions 18: %r" % (r2.output_text or "")[:80])

with o.responses.stream(model="inkling-2-mini", temperature=0, max_output_tokens=800, tools=rt, input=ri, reasoning={"effort": "low", "summary": "auto"}) as s:
    types = [e.type for e in s]
    final = s.get_final_response()
calls = [it for it in final.output if it.type == "function_call"]
check(types[0] == "response.created" and types[-1] == "response.completed" and len(calls) == 1 and calls[0].name == "get_weather",
      "responses stream: %d events, first %s, last %s, function_call %s" % (len(types), types[0], types[-1], calls[0].arguments if calls else None))
ri3 = ri + [it.model_dump(exclude_none=True) for it in final.output] + [{"type": "function_call_output", "call_id": calls[0].call_id, "output": RESULT}]
with o.responses.stream(model="inkling-2-mini", temperature=0, max_output_tokens=800, tools=rt, input=ri3, reasoning={"effort": "low", "summary": "auto"}) as s:
    for _ in s:
        pass
    final2 = s.get_final_response()
check("18" in (final2.output_text or ""), "responses stream: final answer mentions 18: %r" % (final2.output_text or "")[:80])

print("failures: %d" % len(fails))
sys.exit(1 if fails else 0)
