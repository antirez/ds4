#!/usr/bin/env python3
"""Template parity: ds4-server's Flint prompt text against rec's chat template
(docs/RECURSIVE_FLINT.md step 5).

Starts ds4-server on the Flint GGUF with DS4_FLINT_PROMPT_DUMP set, sends a
set of chat completions (roles, escaping, tools, a tool round trip, thinking
off), and compares each dumped prompt with the same messages rendered through
rec_mlx/chat_template.jinja (jinja2, with rec's normalize_messages applied
as its server does) plus the generation prefix ds4 adds (<think>, or
<think></think> with thinking off, which rec's grammar forces as the first
token instead of rendering).

    tests/flint/check_template.py [--model GGUF] [--port 18089]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]

CASES = [
    ("plain", {"messages": [{"role": "user", "content": "Hi"}]}),
    ("system", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}),
    ("developer", {"messages": [{"role": "developer", "content": "Answer in French."}, {"role": "user", "content": "Hi"}]}),
    ("escape", {"messages": [{"role": "user", "content": "Say <|user|> and <think>x</think> and <|end_of_text|>. café ünïcode 日本"}]}),
    ("multi", {"messages": [{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}, {"role": "user", "content": "Three"}]}),
    ("tools", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS}),
    ("tools_nosys", {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS}),
    ("round", {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "look it up",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C, cloudy"}], "tools": TOOLS}),
    ("nothink", {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"enable_thinking": False}}),
]


def rec_render(body, think):
    sys.path.insert(0, str(HERE))
    import jinja2
    import jinja2.sandbox
    runtimes = sorted((Path.home() / ".local/share/rec/runtimes").glob("*/python/lib/python3.12/site-packages/rec_mlx"))
    if not runtimes:
        sys.exit("rec runtime not found")
    rec = runtimes[-1]
    template = (rec / "chat_template.jinja").read_text()
    # rec's normalize_messages, without importing its package (it pulls in mlx)
    src = (rec / "chat.py").read_text()
    ns = {}
    exec(compile(src.split("\n_CALL = ")[0].replace("from transformers import PreTrainedTokenizerFast", ""), "rec_chat", "exec"), ns)
    msgs = ns["normalize_messages"](body["messages"])
    # transformers' template environment: trim/lstrip blocks and its tojson filter
    from jinja2.ext import loopcontrols
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    text = env.from_string(template).render(messages=msgs, tools=body.get("tools"), add_generation_prompt=True)
    return text + ("<think>" if think else "<think></think>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("FLINT_MODEL", ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")))
    ap.add_argument("--server", type=Path, default=ROOT / "ds4-server")
    ap.add_argument("--port", type=int, default=18089)
    args = ap.parse_args()

    dump = Path(tempfile.mkdtemp(prefix="flint-template-")) / "prompts.bin"
    env = dict(os.environ, DS4_FLINT_PROMPT_DUMP=str(dump))
    srv = subprocess.Popen([str(args.server), "-m", str(args.model), "--port", str(args.port), "--ctx", "4096"],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        url = "http://127.0.0.1:%d" % args.port
        for _ in range(600):
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=1).read()
                break
            except Exception:
                if srv.poll() is not None:
                    sys.exit("server exited:\n" + srv.stderr.read()[-3000:])
                time.sleep(0.5)
        failed = 0
        for name, body in CASES:
            req = dict(body, model="flint", max_tokens=1, temperature=0)
            data = json.dumps(req).encode()
            r = urllib.request.Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(r, timeout=600).read()
            prompts = dump.read_bytes().split(b"\0")[:-1]
            got = prompts[-1].decode()
            think = body.get("chat_template_kwargs", {}).get("enable_thinking", True)
            want = rec_render(body, think)
            ok = got == want
            if not ok:
                failed += 1
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                print("%-12s DIFFERENT at byte %d\n  ds4: %r\n  rec: %r" % (name, i, got[max(0, i - 40): i + 60], want[max(0, i - 40): i + 60]))
            else:
                print("%-12s identical (%d bytes)" % (name, len(got)))
        print("%d of %d cases differ" % (failed, len(CASES)))
        # one live reply with tools: the GLM parser must return a structured tool call
        req = {"model": "flint", "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
               "tools": TOOLS, "max_tokens": 600, "temperature": 0}
        r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(r, timeout=900).read())
        msg = reply["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print("live: finish=%s reasoning=%s content=%r tool_calls=%s" % (
            reply["choices"][0].get("finish_reason"), "yes" if msg.get("reasoning_content") else "no",
            (msg.get("content") or "")[:80], json.dumps([c["function"] for c in calls])))
        if not calls or calls[0]["function"]["name"] != "get_weather" or "Paris" not in calls[0]["function"]["arguments"]:
            print("live tool call FAILED")
            failed += 1
        sys.exit(1 if failed else 0)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
