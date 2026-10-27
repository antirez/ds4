#!/usr/bin/env python3
"""Long-context fact recall and a cached continuation through Chat Completions."""

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

from generate_long_context_story_prompt import FACTS, make_story


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "system", "content":
                 "Read the story, remember the assignments, and answer the task exactly."},
                {"role": "user", "content": make_story()}]
    expected = {name: number for name, _, number in FACTS}
    first_prompt = 0
    for turn in range(2):
        body = dict(model=args.model, messages=messages, temperature=0,
                    reasoning_effort="none", max_tokens=512)
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=1200) as response:
            result = json.load(response)
        (args.output / f"turn-{turn+1}.json").write_text(json.dumps(result, indent=2) + "\n")
        message = result["choices"][0]["message"]
        actual = {name: int(value) for name, value in
                  re.findall(r"\b([A-Z][a-z]+)\s*=\s*(\d+)\b", message.get("content") or "")}
        assert actual == expected, (turn+1, actual, expected)
        usage = result["usage"]
        if not turn:
            first_prompt = usage["prompt_tokens"]
            assert first_prompt > 30000, usage
        else:
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            assert cached >= first_prompt-4, usage
            assert usage["prompt_tokens"]-cached < 512, usage
        print(json.dumps(dict(turn=turn+1, seconds=time.monotonic()-start,
                              facts=len(actual), usage=usage)), flush=True)
        messages.append(message)
        messages.append({"role": "user", "content":
                         "Update the ledger: Alice is now assigned seventeen, and Owen "
                         "is assigned forty-two. Return all sixteen assignments again. "
                         "Only these two have changed. Use Name=number lines, no prose."})
        expected.update(Alice=17, Owen=42)
    print("PASS long story: all 16 facts, corrections, cached continuation", flush=True)


if __name__ == "__main__":
    main()
