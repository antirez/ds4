#!/usr/bin/env python3
"""Measure ds4-server prefill and decode rates.

Streams the OpenAI-compatible chat completion API to separate prefill
(time-to-first-token) from decode (steady-state generation), so the same
methodology covers single-host Q2, single-host Q4 (on a >=256 GB machine),
and pipeline-parallel Q4 over RPC.  Whatever ds4-server is running, this
measures what it actually produces over the wire -- no special hooks.

Usage:
    python3 bench-ds4.py [--url URL] [--long PATH] [--max-tokens N]
                        [--no-thinking | --thinking] [--runs N]

Examples:
    # Default: localhost:8000, short prompt only, thinking disabled.
    python3 bench-ds4.py

    # With the long-context test the README uses (11709 tokens):
    python3 bench-ds4.py --long tests/long_context_security_prompt.txt

    # Average over 3 runs (cold first run is reported separately):
    python3 bench-ds4.py --runs 3 --long tests/long_context_security_prompt.txt

The same prompt is reused across runs in --runs mode so the rendered-prefix
cache absorbs setup overhead; the first run is reported as "cold" and the
remaining runs are averaged as "warm".
"""

import argparse
import json
import sys
import time
import urllib.request


def stream_request(url: str, prompt: str, max_tokens: int, thinking_enabled: bool):
    """One streamed chat completion.  Returns timing + token counts."""
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not thinking_enabled:
        body["thinking"] = {"type": "disabled"}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first = None
    t_last = None
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0

    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") is not None or delta.get("reasoning_content") is not None:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    if delta.get("reasoning_content") is not None:
                        reasoning_tokens += 1

            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

    t_end = time.perf_counter()
    return {
        "t_total": t_end - t_start,
        "t_prefill": (t_first - t_start) if t_first is not None else None,
        "t_decode": (t_last - t_first) if (t_first is not None and t_last is not None) else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def fmt_rate(tokens, seconds):
    if seconds is None or seconds <= 0 or tokens <= 0:
        return "n/a"
    return f"{tokens / seconds:.1f} t/s"


def print_run(label, r):
    print(f"[{label}]")
    print(f"  prompt        : {r['prompt_tokens']} tokens")
    print(f"  completion    : {r['completion_tokens']} tokens"
          + (f"  ({r['reasoning_tokens']} reasoning)" if r['reasoning_tokens'] else ""))
    if r["t_prefill"] is not None:
        print(f"  prefill       : {r['t_prefill']:6.2f} s  "
              f"({fmt_rate(r['prompt_tokens'], r['t_prefill'])})")
    else:
        print("  prefill       : n/a (no streamed token observed)")
    if r["t_decode"] is not None and r["completion_tokens"] > 1:
        # Decode rate excludes the first token (which is part of prefill latency).
        print(f"  decode        : {r['t_decode']:6.2f} s  "
              f"({fmt_rate(r['completion_tokens'] - 1, r['t_decode'])})")
    else:
        print("  decode        : n/a")
    print(f"  wall total    : {r['t_total']:6.2f} s")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--short", default=(
        "Scrivi una breve storia su un gatto che impara a programmare in C. "
        "Mantieni la storia entro tre paragrafi."
    ), help="Short-prompt text (default: small Italian story prompt).")
    ap.add_argument("--long", help="Path to a long-context prompt file.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true",
                    help="Leave thinking mode enabled (default: disabled for "
                         "predictable decode rate).")
    ap.add_argument("--runs", type=int, default=1,
                    help="Repeat each prompt N times; first is reported cold, "
                         "remaining are averaged warm.")
    args = ap.parse_args()

    long_prompt = None
    if args.long:
        with open(args.long, "r", encoding="utf-8") as f:
            long_prompt = f.read()

    cases = [("short", args.short)]
    if long_prompt is not None:
        cases.append((f"long ({len(long_prompt)} chars)", long_prompt))

    for case_label, prompt in cases:
        results = []
        for i in range(args.runs):
            try:
                r = stream_request(args.url, prompt, args.max_tokens, args.thinking)
            except Exception as e:
                print(f"[{case_label} run {i+1}] failed: {e}")
                sys.exit(1)
            results.append(r)
            tag = "cold" if i == 0 and args.runs > 1 else f"run {i+1}"
            print_run(f"{case_label} {tag}", r)

        if args.runs > 1:
            warm = results[1:]
            warm_prefill = [r["t_prefill"] for r in warm if r["t_prefill"]]
            warm_decode = [(r["completion_tokens"] - 1, r["t_decode"]) for r in warm
                           if r["t_decode"] and r["completion_tokens"] > 1]
            if warm_prefill and results[0]["prompt_tokens"]:
                avg_prefill = sum(warm_prefill) / len(warm_prefill)
                print(f"[{case_label} warm avg over {len(warm)} run(s)]")
                print(f"  prefill       : {avg_prefill:6.2f} s  "
                      f"({fmt_rate(results[0]['prompt_tokens'], avg_prefill)})")
            if warm_decode:
                tot_tokens = sum(t for t, _ in warm_decode)
                tot_seconds = sum(s for _, s in warm_decode)
                print(f"  decode        : {tot_seconds:6.2f} s  "
                      f"({fmt_rate(tot_tokens, tot_seconds)})")
            print()


if __name__ == "__main__":
    main()
