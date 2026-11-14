#!/usr/bin/env python3
"""Collect tests/flint/golden.json from a running `rec serve`.

Each case is sent as chat messages with temperature 0 (argmax in rec's
engine), stream false and logprobs true; rec answers with the generated
token ids and the chosen token's logprob per step, which is all its API
exposes.  The file records the request, rec's prompt token count, the
generated ids with their logprobs and the finish reason.  ds4's reference
check renders the same messages, requires the same prompt length and ids,
and compares logprobs within rec's own spread (docs/RECURSIVE_FLINT.md,
Verification).  tests/flint/rec_logits.py later adds the exact prompt ids
and full logits from rec's runtime.

    rec serve &            # port 11500
    tests/flint/collect_golden.py [--out tests/flint/golden.json] [--max-tokens 128]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def long_story_prompt(target_chars: int) -> str:
    text = (ROOT / "tests/long_context_story_prompt.txt").read_text(encoding="utf-8")
    cut = text[:target_chars]
    cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
    return ("Read the following story and then summarize it in English in five bullet points.\n\n<story>\n"
            + cut + "\n</story>")


def cases(long_chars: int):
    return [
        {"id": "short_01_capital", "messages": [
            {"role": "user", "content": "What is the capital of Australia? Answer briefly."}]},
        {"id": "short_02_math", "messages": [
            {"role": "user", "content": "Compute 17 * 23 and explain the steps."}]},
        {"id": "short_03_code", "messages": [
            {"role": "user", "content": "Write a C function that reverses a singly linked list in place."}]},
        {"id": "short_04_sys", "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Explain the difference between TCP and UDP in two sentences."}]},
        {"id": "short_05_italian", "messages": [
            {"role": "user", "content": "Traduci in inglese: \"Quel ramo del lago di Como, che volge a mezzogiorno.\""}]},
        {"id": "tool_01_weather", "tools": TOOLS, "messages": [
            {"role": "user", "content": "What's the weather like in Paris right now? Use the tool."}]},
        {"id": "multi_01_followup", "messages": [
            {"role": "user", "content": "Name three prime numbers above 100."},
            {"role": "assistant", "content": "101, 103 and 107."},
            {"role": "user", "content": "Now give their sum."}]},
        {"id": "long_01_story", "messages": [
            {"role": "user", "content": long_story_prompt(long_chars)}]},
    ]


def api_key() -> str:
    return subprocess.run([str(Path.home() / ".local/bin/rec"), "auth", "token"],
                          capture_output=True, text=True, check=True).stdout.strip()


def post(url: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11500")
    ap.add_argument("--out", type=Path, default=HERE / "golden.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-chars", type=int, default=72000, help="characters of the story case (about 20K tokens)")
    ap.add_argument("--model", type=Path, default=ROOT.parent / "ds4/gguf/flint/flint-int4.gguf")
    ap.add_argument("--case", action="append", help="only these case ids")
    args = ap.parse_args()

    key = api_key()
    models = json.loads(urllib.request.urlopen(urllib.request.Request(
        args.url + "/v1/models", headers={"Authorization": "Bearer " + key}), timeout=30).read())
    model_id = models["data"][0]["id"]
    print("rec model id: %s" % model_id, file=sys.stderr)

    out = {"source": "rec serve (%s), temperature 0, max_tokens %d, logprobs true" % (model_id, args.max_tokens),
           "rec_version": subprocess.run([str(Path.home() / ".local/bin/rec"), "--version"],
                                         capture_output=True, text=True).stdout.strip(),
           "cases": []}
    if args.model.is_file():
        h = hashlib.sha256()
        with open(args.model, "rb") as f:
            # The header alone pins the conversion (it carries the checkpoint digests).
            h.update(f.read(1 << 20))
        out["gguf_header_sha256_1mib"] = h.hexdigest()

    for case in cases(args.long_chars):
        if args.case and case["id"] not in args.case:
            continue
        body = {"model": model_id, "messages": case["messages"], "temperature": 0.0,
                "max_tokens": args.max_tokens, "stream": False, "logprobs": True}
        if "tools" in case:
            body["tools"] = case["tools"]
        t0 = time.time()
        resp = post(args.url + "/v1/chat/completions", key, body, timeout=1800)
        dt = time.time() - t0
        choice = resp["choices"][0]
        entries = choice.get("logprobs", {}).get("content", []) or []
        rec = {
            "id": case["id"],
            "messages": case["messages"],
            "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "generated_ids": [e["id"] for e in entries],
            "logprobs": [round(float(e["logprob"]), 6) for e in entries],
            "text": choice["message"].get("content"),
            "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
            "tool_calls": choice["message"].get("tool_calls"),
            "seconds": round(dt, 2),
        }
        if "tools" in case:
            rec["tools"] = case["tools"]
        out["cases"].append(rec)
        print("%-20s prompt %6s tokens, %4d generated, %s, %.1fs" % (
            case["id"], rec["prompt_tokens"], len(rec["generated_ids"]), rec["finish_reason"], dt), file=sys.stderr)

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote %s" % args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
