#!/usr/bin/env python3
"""Compare DwarfStar V4.1 prompts with the released encoder and tokenizer."""
import __future__
import argparse
import json
from pathlib import Path
import subprocess

from tokenizers import Tokenizer


def anthropic_messages(messages):
    converted = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                       "content": message["content"]}]
            role = "user"
        else:
            blocks = []
            if message.get("reasoning_content"):
                blocks.append({"type": "thinking", "thinking": message["reasoning_content"]})
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": call["id"],
                               "name": call["function"]["name"],
                               "input": json.loads(call["function"]["arguments"])})
        if role == "user" and converted and converted[-1]["role"] == "user":
            # Consecutive text blocks have no separator in the Anthropic wire
            # protocol. Insert the equivalent merged-turn boundary explicitly.
            if blocks[0]["type"] == "text" and converted[-1]["content"][-1]["type"] == "text":
                converted[-1]["content"].append({"type": "text", "text": "\n\n"})
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"content": blocks, "role": role})
    return converted


def check(reference, tokenizer_path, model, executable):
    namespace = {"__name__": "deepseek41_reference"}
    exec(compile(Path(reference).read_text(), reference, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    encode = namespace["encode_messages"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    cases = 0
    for level in ["high", "max"] + list(range(101)):
        for system in ["", "You are a helpful assistant."]:
            prompt = "Count: 1, 2, 3. Then explain x *= 2. Caff\u00e8?"
            messages = ([{"role": "system", "content": system}] if system else [])
            messages.append({"role": "user", "content": prompt})
            rendered = encode(messages, "chat" if level == 0 else "thinking",
                              reasoning_effort=None if level == 0 else level)
            expected = tokenizer.encode(rendered, add_special_tokens=False).ids
            result = subprocess.run([executable, "--chat-fixture", model,
                                     str(level), system, prompt], check=True,
                                    capture_output=True, text=True)
            actual = json.loads(result.stdout.splitlines()[0])
            if actual != expected:
                raise AssertionError(f"effort={level}, system={bool(system)}\n"
                                     f"actual={actual}\nexpected={expected}")
            if level in ["high", "max", 0, 1, 25, 100]:
                option = ["--think" if level == "high" else "--think-max"] if isinstance(level, str) else ["--think-level", str(level)]
                dumped = subprocess.run(["./ds4", "-m", model, "--dump-tokens",
                                         "--ctx", "256", "-sys", system, "-p", prompt] + option,
                                        check=True, capture_output=True, text=True)
                assert json.loads(dumped.stdout.splitlines()[0]) == expected
            cases += 1
    for binary in ["./ds4", "./ds4-agent"]:
        for bad in ["-1", "101", "25.0", "foo", "1 2", "999999999999"]:
            result = subprocess.run([binary, "-m", "/nonexistent", "--think-level", bad],
                                    capture_output=True, text=True)
            assert result.returncode == 2 and "integer from 0 to 100" in result.stderr
    print(f"V4.1 released encoder/tokenizer: {cases} exact prompt matches")

    call = {"role": "assistant", "content": "Checking.", "reasoning_content": "Inspect first.",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "read", "arguments": '{"path":"a.c","max_lines":10}'}}]}
    conversations = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "system", "content": ""}, {"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "One"}, {"role": "user", "content": "Two"}],
        [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello", "reasoning_content": "A greeting."},
         {"role": "system", "content": "Keep going."}],
        [{"role": "user", "content": "Read a.c"}, call,
         {"role": "tool", "tool_call_id": "call_1", "content": "int main(void) {}"},
         {"role": "user", "content": "Explain it."}],
    ]
    parallel = json.loads(json.dumps(call))
    parallel["tool_calls"].append({"id": "call_2", "type": "function", "function": {
        "name": "list", "arguments": '{"path":"."}'}})
    conversations.append([
        {"role": "user", "content": "Inspect"}, parallel,
        {"role": "tool", "tool_call_id": "call_2", "content": "SECOND &lt;"},
        {"role": "user", "content": "Keep literal <tool_result>text</tool_result> in place."},
        {"role": "tool", "tool_call_id": "call_1", "content": ""},
    ])
    for level in [0, 1, 25, 75, 100]:
        for messages in conversations:
            keep = any(m.get("tool_calls") for m in messages)
            expected = encode(messages, "chat" if level == 0 else "thinking",
                              drop_thinking=not keep, reasoning_effort=level or "high")
            result = subprocess.run(["./ds4_test", "--ds41-render", str(level),
                                     json.dumps(messages)], check=True, capture_output=True, text=True)
            assert result.stdout == expected, (level, result.stdout, expected)
            result = subprocess.run(["./ds4_test", "--ds41-render-anthropic", str(level),
                                     json.dumps(anthropic_messages(messages))],
                                    check=True, capture_output=True, text=True)
            assert result.stdout == expected, (level, result.stdout, expected)
    print(f"V4.1 server: {10 * len(conversations)} exact OpenAI/Anthropic prompt matches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="released encoding/encoding.py")
    parser.add_argument("tokenizer", help="released tokenizer.json")
    parser.add_argument("model", help="V4.1 GGUF, including a sparse zero fixture")
    parser.add_argument("--executable", default="./tests/test_deepseek41_graph")
    args = parser.parse_args()
    check(args.reference, args.tokenizer, args.model, args.executable)
