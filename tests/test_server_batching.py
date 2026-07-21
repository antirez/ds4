#!/usr/bin/env python3
"""Concurrent API correctness/load check for ds4-server session batching.

Each case is submitted twice with the same non-zero seed. The pairs run in one
cold concurrent wave and must return identical output, even though prompt sizes
and output limits differ across cases. A fresh nonce avoids accidental reuse of
an earlier run's in-memory or disk checkpoint.
"""

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


CASES = (
    ("short-greedy", 0, 80, 0.0, 101),
    ("medium-sampled", 384, 72, 0.7, 202),
    ("long-greedy", 2304, 48, 0.0, 303),
    ("short-sampled", 32, 96, 0.65, 404),
    ("very-long-sampled", 3600, 32, 0.75, 505),
    ("medium-greedy", 1024, 64, 0.0, 606),
    ("long-sampled", 2048, 40, 0.7, 707),
    ("tiny-sampled", 0, 96, 0.6, 808),
)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def make_payload(case, case_number, nonce, stream):
    name, filler_words, max_tokens, temperature, seed = case
    filler = (" alpha beta gamma delta" * ((filler_words + 3) // 4))
    filler = " ".join(filler.split()[:filler_words])
    prompt = (
        "This is batching test %s case %d (%s). Read the filler, then write "
        "a concise explanation of why deterministic request isolation matters. "
        "Do not quote the filler.\nFILLER:\n%s"
    ) % (nonce, case_number, name, filler)
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "seed": seed + case_number * 1000,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return name, filler_words, payload


def parse_stream(raw):
    content = []
    reasoning = []
    finish = None
    usage = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content") is not None:
            content.append(delta["content"])
        if delta.get("reasoning_content") is not None:
            reasoning.append(delta["reasoning_content"])
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "finish": finish,
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def post_chat(url, payload, timeout, start_event):
    start_event.wait()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP %d: %s" % (exc.code, detail)) from exc
    elapsed = time.monotonic() - started

    if payload["stream"]:
        result = parse_stream(raw)
    else:
        response = json.loads(raw)
        choice = response["choices"][0]
        message = choice.get("message") or {}
        result = {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "finish": choice.get("finish_reason"),
            "completion_tokens": (response.get("usage") or {}).get(
                "completion_tokens"
            ),
        }
    result["elapsed"] = elapsed
    return result


def comparable(result):
    return (
        result["content"],
        result["reasoning"],
        result["finish"],
        result["completion_tokens"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--nonce", default="")
    parser.add_argument(
        "--case", choices=[case[0] for case in CASES],
        help="repeat one case shape instead of cycling through mixed lengths",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")

    nonce = args.nonce or "cold-%d" % time.time_ns()
    requests = []
    metadata = []
    for i in range(args.pairs):
        case = next((c for c in CASES if c[0] == args.case), None)
        if case is None:
            case = CASES[i % len(CASES)]
        name, filler_words, payload = make_payload(case, i, nonce, args.stream)
        for copy in range(2):
            requests.append(payload)
            metadata.append((i, copy, name, filler_words))

    workers = args.workers or len(requests)
    workers = max(1, min(workers, len(requests)))
    start_event = threading.Event()
    wall_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(post_chat, args.url, payload, args.timeout, start_event)
            for payload in requests
        ]
        start_event.set()
        results = [future.result() for future in futures]
    wall = time.monotonic() - wall_start

    failures = 0
    for i in range(args.pairs):
        a = results[2 * i]
        b = results[2 * i + 1]
        if comparable(a) != comparable(b):
            failures += 1
            print("MISMATCH pair=%d case=%s" % (i, metadata[2 * i][2]), file=sys.stderr)
            print("  A=%s" % (json.dumps(a, ensure_ascii=True)[:1000]), file=sys.stderr)
            print("  B=%s" % (json.dumps(b, ensure_ascii=True)[:1000]), file=sys.stderr)

    latencies = [result["elapsed"] for result in results]
    known_tokens = [
        result["completion_tokens"]
        for result in results
        if result["completion_tokens"] is not None
    ]
    summary = {
        "status": "PASS" if failures == 0 else "FAIL",
        "pairs": args.pairs,
        "requests": len(requests),
        "workers": workers,
        "stream": args.stream,
        "wall_seconds": round(wall, 3),
        "latency_p50_seconds": round(statistics.median(latencies), 3),
        "latency_p95_seconds": round(percentile(latencies, 0.95), 3),
        "completion_tokens": sum(known_tokens) if known_tokens else None,
        "completion_tokens_per_second": (
            round(sum(known_tokens) / wall, 2) if known_tokens else None
        ),
        "nonce": nonce,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
