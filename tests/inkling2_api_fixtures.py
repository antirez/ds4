#!/usr/bin/env python3
"""Builds tests/test-vectors/inkling2-api/NN_name.json: each file holds the same conversation as an OpenAI chat
request, an Anthropic /v1/messages request and an OpenAI /v1/responses request.  The model-free test checks that
the three render to the same prompt; tests/inkling2_server/api_equiv.py checks the three answers on a live server.

    python3 tests/inkling2_api_fixtures.py [outdir]
"""
import json, os, sys

TOOLS = {
    "get_weather": ("Current weather for a city.",
                    {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                     "required": ["city"]}),
    "add": ("Add two numbers.",
            {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
    "search": ("Search the notes.",
               {"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}}, "required": ["query"]}),
}

# turn kinds: ("user", text) ("assistant", {"text":..,"reasoning":..,"calls":[(id,name,args_dict)]}) ("tool", id, text)
CONVS = []

def conv(name, turns, system=None, tools=(), effort=None, budget=None, max_tokens=700):
    CONVS.append(dict(name=name, turns=turns, system=system, tools=list(tools), effort=effort, budget=budget, max_tokens=max_tokens))

U = lambda t: ("user", t)
A = lambda text="", reasoning=None, calls=(): ("assistant", {"text": text, "reasoning": reasoning, "calls": list(calls)})
T = lambda cid, t: ("tool", cid, t)

conv("plain", [U("What is the capital of France? One word.")])
conv("system_user", [U("Say hello.")], system="You are a terse assistant. Reply in at most five words.")
conv("multi_turn", [U("My name is Ada."), A("Nice to meet you, Ada."), U("What is my name? One word.")])
conv("thinking_replay", [U("What is 17*3?"), A("17*3 is 51.", reasoning="17*3 = 51."), U("And times 2? Just the number.")])
conv("effort_low", [U("Name a prime number above 100. Just the number.")], effort="low")
conv("budget_1500", [U("What is 12+30? Just the number.")], budget=1500)
conv("thinking_disabled", [U("What is the capital of Italy? One word.")], effort="none")
conv("one_tool_ask", [U("What is the weather in Paris right now? Use the tool.")], tools=["get_weather"])
conv("tool_result_final", [U("Weather in Paris?"),
                           A("", reasoning="I should call the tool.", calls=[("toolu_01", "get_weather", {"city": "Paris", "unit": "c"})]),
                           T("toolu_01", "18 C, light rain")], tools=["get_weather"])
conv("two_parallel_calls", [U("Weather in Rome and Oslo, use the tool for both."),
                            A("", calls=[("toolu_a", "get_weather", {"city": "Rome"}), ("toolu_b", "get_weather", {"city": "Oslo"})]),
                            T("toolu_a", "24 C, sunny"), T("toolu_b", "3 C, snow")], tools=["get_weather"])
conv("multi_step_tools", [U("Add 2 and 3, then add the result to 10."),
                          A("", calls=[("toolu_1", "add", {"a": 2, "b": 3})]),
                          T("toolu_1", "5"),
                          A("2+3 is 5. Now adding 10.", calls=[("toolu_2", "add", {"a": 5, "b": 10})]),
                          T("toolu_2", "15"),
                          U("Thanks. What was the last sum? Answer in a few words.")], tools=["add"])
conv("unicode", [U("Écris « bonjour » en japonais: こんにちは. Emoji: 🙂. Reply with one short line.")], system="Tu es un assistant. Réponds brièvement. ñ ü €")
conv("tools_declared_plain", [U("Just say OK.")], tools=["get_weather", "add"])
conv("text_and_call", [U("Check the weather in Madrid."),
                       A("Let me check that for you.", calls=[("toolu_m", "get_weather", {"city": "Madrid"})]),
                       T("toolu_m", "31 C")], tools=["get_weather"])
conv("json_tool_result", [U("Search my notes for apples."),
                          A("", calls=[("toolu_s", "search", {"query": "apples", "filters": {"tag": "fruit", "n": 3}})]),
                          T("toolu_s", '{"hits": [{"title": "Apple pie", "score": 0.9}, {"title": "Cider", "score": 0.4}]}')],
     tools=["search"])
conv("long_system_quotes", [U("Repeat the word in quotes from your instructions.")],
     system='Rules:\n1. Always answer in lower case.\n2. The secret word is "pelican".\n3. Use a \\ backslash never.\n\n\tIndented line with "quotes" and <tags>.')
conv("nested_unicode_args", [U("Find notes about Zürich."),
                             A("", reasoning="Search notes for Zürich.", calls=[("toolu_z", "search", {"query": "Zürich 🙂", "filters": {"city": "Zürich", "ids": [1, 2, 3], "ok": True, "x": None}})]),
                             T("toolu_z", "no notes found")], tools=["search"])
conv("reasoning_with_call_then_result", [U("What is 6+7? Use the add tool."),
                                         A("", reasoning="Use add with 6 and 7.", calls=[("toolu_r", "add", {"a": 6, "b": 7})]),
                                         T("toolu_r", "13")], tools=["add"], effort="medium")

conv("parallel_ask", [U("What is the weather in Rome and in Oslo? Call the tool once for each city, both calls in the same turn.")], tools=["get_weather"], effort="low")
conv("add_ask_system", [U("Add 123.5 and 876.25.")], system="Always use the tools for arithmetic. Never calculate yourself.", tools=["add"], effort="low")
conv("search_ask_nested", [U("Search my notes for 'tea', with filters {\"tag\": \"beverage\", \"limit\": 5}.")], tools=["search"], effort="low")
conv("call_after_result", [U("Add 2 and 3 with the tool."), A("", calls=[("toolu_q", "add", {"a": 2, "b": 3})]), T("toolu_q", "5"),
                           U("Now use the tool to add 10 to that result.")], tools=["add"], effort="low")
conv("unicode_ask", [U("Weather in Zürich and in 東京, one tool call per city.")], tools=["get_weather"], effort="low")


def tool_defs_chat(names):
    return [{"type": "function", "function": {"name": n, "description": TOOLS[n][0], "parameters": TOOLS[n][1]}} for n in names]

def tool_defs_anth(names):
    return [{"name": n, "description": TOOLS[n][0], "input_schema": TOOLS[n][1]} for n in names]

def tool_defs_resp(names):
    return [{"type": "function", "name": n, "description": TOOLS[n][0], "parameters": TOOLS[n][1]} for n in names]


def to_chat(c):
    msgs = []
    if c["system"]:
        msgs.append({"role": "system", "content": c["system"]})
    for t in c["turns"]:
        if t[0] == "user":
            msgs.append({"role": "user", "content": t[1]})
        elif t[0] == "assistant":
            d = t[1]
            m = {"role": "assistant", "content": d["text"] or ""}
            if d["reasoning"]:
                m["reasoning_content"] = d["reasoning"]
            if d["calls"]:
                m["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for i, n, a in d["calls"]]
            msgs.append(m)
        else:
            msgs.append({"role": "tool", "tool_call_id": t[1], "content": t[2]})
    req = {"model": "inkling-2-mini", "messages": msgs, "temperature": 0, "max_tokens": c["max_tokens"]}
    if c["tools"]:
        req["tools"] = tool_defs_chat(c["tools"])
    eff = c["effort"] or (budget_effort(c["budget"]) if c["budget"] else None)
    if eff:
        req["reasoning_effort"] = eff
    return req


def budget_effort(b):
    # same table as ik_anthropic_budget_mode
    return "low" if b < 2048 else "medium" if b < 8192 else "high" if b < 32000 else "xhigh"


def to_anthropic(c):
    msgs = []
    pending_results = []
    def flush():
        nonlocal pending_results
        if pending_results:
            msgs.append({"role": "user", "content": pending_results})
            pending_results = []
    for t in c["turns"]:
        if t[0] == "tool":
            pending_results.append({"type": "tool_result", "tool_use_id": t[1], "content": t[2]})
            continue
        flush()
        if t[0] == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": t[1]}]})
        else:
            d = t[1]
            blocks = []
            if d["reasoning"]:
                blocks.append({"type": "thinking", "thinking": d["reasoning"], "signature": "sig"})
            if d["text"]:
                blocks.append({"type": "text", "text": d["text"]})
            for i, n, a in d["calls"]:
                blocks.append({"type": "tool_use", "id": i, "name": n, "input": a})
            msgs.append({"role": "assistant", "content": blocks})
    flush()
    req = {"model": "inkling-2-mini", "messages": msgs, "temperature": 0, "max_tokens": c["max_tokens"]}
    if c["system"]:
        req["system"] = c["system"]
    if c["tools"]:
        req["tools"] = tool_defs_anth(c["tools"])
    if c["effort"] == "none":
        req["thinking"] = {"type": "disabled"}
    elif c["effort"]:
        req["output_config"] = {"effort": c["effort"]}
    elif c["budget"]:
        req["thinking"] = {"type": "enabled", "budget_tokens": c["budget"]}
    return req


def to_responses(c):
    items = []
    for t in c["turns"]:
        if t[0] == "user":
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": t[1]}]})
        elif t[0] == "assistant":
            d = t[1]
            if d["reasoning"]:
                items.append({"type": "reasoning", "id": "rs_x", "summary": [{"type": "summary_text", "text": d["reasoning"]}]})
            if d["text"]:
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": d["text"]}]})
            for i, n, a in d["calls"]:
                items.append({"type": "function_call", "call_id": i, "name": n, "arguments": json.dumps(a, ensure_ascii=False)})
        else:
            items.append({"type": "function_call_output", "call_id": t[1], "output": t[2]})
    req = {"model": "inkling-2-mini", "input": items, "temperature": 0, "max_output_tokens": c["max_tokens"],
           "reasoning": {"summary": "auto"}}
    if c["system"]:
        req["instructions"] = c["system"]
    if c["tools"]:
        req["tools"] = tool_defs_resp(c["tools"])
    eff = c["effort"] or (budget_effort(c["budget"]) if c["budget"] else None)
    if eff:
        req["reasoning"]["effort"] = eff
    return req


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "test-vectors", "inkling2-api")
    os.makedirs(out, exist_ok=True)
    cs = [c for c in CONVS if c]
    for i, c in enumerate(cs, 1):
        doc = {"name": "%02d_%s" % (i, c["name"]), "chat": to_chat(c), "anthropic": to_anthropic(c), "responses": to_responses(c)}
        with open(os.path.join(out, doc["name"] + ".json"), "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
    print("wrote", len(cs), "conversations to", out)


main()
