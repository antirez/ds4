#!/usr/bin/env python3
"""End-to-end serving benchmark for ds4-server under concurrent load.

The engine-level harness (speed-bench/session_concurrency_bench.c) measures
what the graph can do.  This one measures what a client actually gets: queueing
in front of the slots, prompt rendering, prefix-cache decisions, streaming, and
the decode worker's batching all count here.  Metrics follow the vocabulary
serving benchmarks converged on, so the numbers are comparable with what people
quote elsewhere:

  TTFT  time to first token, the wait before anything appears
  ITL   inter-token latency, the gap between consecutive streamed chunks
  TPOT  (end-to-end latency - TTFT) / (output tokens - 1), per-request average
  E2EL  end-to-end latency of the whole request

Two load shapes:

  --concurrency N   N clients, each sending the next request as soon as the
                    previous one finishes.  This is the closed-loop shape the
                    engine benchmark mirrors, and the one to use when comparing
                    batching work.
  --request-rate R  Poisson arrivals at R requests per second, unbounded
                    in-flight.  Use it to find where latency degrades.

Each request carries a fresh nonce so the server cannot answer from a cached
prefix; --shared-prefix measures the cached path on purpose.

Only the standard library is used, like the other tools in this directory.

Example:

  ./ds4-server --ctx 32768 --batched-session 8 &
  python3 speed-bench/serve_concurrency_bench.py \\
      --concurrency 8 --prompt-tokens 4096 --max-tokens 128 --requests 32
"""

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    ok: bool = False
    error: str = ""
    start: float = 0.0
    ttft: float = 0.0
    e2el: float = 0.0
    itl: list = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def tpot(self):
        if self.output_tokens > 1:
            return (self.e2el - self.ttft) / (self.output_tokens - 1)
        return 0.0


def post_stream(url, payload, timeout):
    """Send one streaming chat completion, timing every SSE delta."""
    result = RequestResult()
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result.start = time.perf_counter()
    last = result.start
    done = False
    finish = None
    counted = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            pending = b""
            for raw in response:
                pending += raw
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        continue
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise ValueError(str(chunk["error"]))
                    usage = chunk.get("usage")
                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens", 0)
                        counted = usage.get("completion_tokens")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason") is not None:
                        finish = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                    if not text:
                        continue
                    now = time.perf_counter()
                    if result.ttft == 0.0:
                        result.ttft = now - result.start
                    else:
                        result.itl.append(now - last)
                    last = now
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result.error = str(exc)
        return result

    result.e2el = time.perf_counter() - result.start
    if not done or finish not in ("stop", "length"):
        result.error = "incomplete or failed stream"
    elif not isinstance(counted, int) or not 0 < counted <= payload["max_tokens"]:
        result.error = "missing or invalid completion token count"
    elif not result.ttft:
        result.error = "no tokens streamed"
    else:
        result.output_tokens = counted
        result.ok = True
    return result


def calibrate_chars_per_token(url, model, text, timeout):
    """Ask the server what a known slice costs, so prompt sizes are real."""
    sample = text[:20000]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": sample}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    prompt_tokens = body.get("usage", {}).get("prompt_tokens", 0)
    if prompt_tokens <= 0:
        raise RuntimeError("server did not report prompt_tokens; cannot size prompts")
    return len(sample) / float(prompt_tokens)


def build_prompt(text, chars, nonce):
    """A slice of the corpus at a nonce-derived offset, prefixed by the nonce.

    Different offsets keep the prompts distinct, which keeps MoE routing and
    cache behaviour honest across concurrent streams.
    """
    if chars <= 0:
        return f"{nonce} Reply with a short paragraph."
    span = max(1, len(text) - chars)
    start = (hash(nonce) & 0x7FFFFFFF) % span
    return (
        f"[{nonce}]\n"
        + text[start:start + chars]
        + "\n\nSummarize the passage above in a few sentences."
    )


def percentiles(values, fractions=(0.5, 0.9, 0.95, 0.99)):
    if not values:
        return {f: 0.0 for f in fractions}
    ordered = sorted(values)
    out = {}
    for fraction in fractions:
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        out[fraction] = ordered[index]
    return out


def summarize(results, wall):
    done = [r for r in results if r.ok]
    failed = len(results) - len(done)
    output_tokens = sum(r.output_tokens for r in done)
    prompt_tokens = sum(r.prompt_tokens for r in done)
    ttfts = [r.ttft * 1e3 for r in done]
    tpots = [r.tpot * 1e3 for r in done if r.output_tokens > 1]
    itls = [v * 1e3 for r in done for v in r.itl]
    e2els = [r.e2el * 1e3 for r in done]
    return {
        "completed": len(done),
        "failed": failed,
        "duration_s": wall,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "request_throughput": len(done) / wall if wall > 0 else 0.0,
        "output_throughput": output_tokens / wall if wall > 0 else 0.0,
        "total_token_throughput": (prompt_tokens + output_tokens) / wall if wall > 0 else 0.0,
        "ttft_ms": {
            "mean": statistics.fmean(ttfts) if ttfts else 0.0,
            **{f"p{int(f * 100)}": v for f, v in percentiles(ttfts).items()},
        },
        "tpot_ms": {
            "mean": statistics.fmean(tpots) if tpots else 0.0,
            **{f"p{int(f * 100)}": v for f, v in percentiles(tpots).items()},
        },
        "itl_ms": {
            "mean": statistics.fmean(itls) if itls else 0.0,
            **{f"p{int(f * 100)}": v for f, v in percentiles(itls).items()},
        },
        "e2el_ms": {
            "mean": statistics.fmean(e2els) if e2els else 0.0,
            **{f"p{int(f * 100)}": v for f, v in percentiles(e2els).items()},
        },
    }


def print_summary(label, stats):
    print(f"\n===== {label} =====")
    print(f"{'Completed requests':<34}{stats['completed']}")
    if stats["failed"]:
        print(f"{'Failed requests':<34}{stats['failed']}")
    print(f"{'Benchmark duration (s)':<34}{stats['duration_s']:.2f}")
    print(f"{'Total input tokens':<34}{stats['prompt_tokens']}")
    print(f"{'Total generated tokens':<34}{stats['output_tokens']}")
    print(f"{'Request throughput (req/s)':<34}{stats['request_throughput']:.3f}")
    print(f"{'Output throughput (tok/s)':<34}{stats['output_throughput']:.2f}")
    print(f"{'Total token throughput (tok/s)':<34}{stats['total_token_throughput']:.2f}")
    for name, key in (("TTFT", "ttft_ms"), ("TPOT", "tpot_ms"),
                      ("ITL", "itl_ms"), ("E2EL", "e2el_ms")):
        block = stats[key]
        print(f"---- {name} (ms) "
              f"mean {block['mean']:.2f} | p50 {block['p50']:.2f} | "
              f"p95 {block['p95']:.2f} | p99 {block['p99']:.2f}")


def run_closed_loop(args, url, text, chars, results, lock):
    """N clients, each sending the next request as soon as one finishes."""
    counter = make_counter()

    def worker(worker_id):
        while True:
            index = counter()
            if index >= args.requests:
                return
            nonce = args.prefix_nonce if args.shared_prefix else f"{worker_id}-{index}-{random.getrandbits(48):012x}"
            payload = make_payload(args, build_prompt(text, chars, nonce))
            result = post_stream(url, payload, args.timeout)
            with lock:
                results.append(result)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(args.concurrency)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


def run_poisson(args, url, text, chars, results, lock):
    """Unbounded in-flight requests arriving at the configured rate."""
    threads = []
    start = time.perf_counter()
    for index in range(args.requests):
        nonce = args.prefix_nonce if args.shared_prefix else f"{index}-{random.getrandbits(48):012x}"
        payload = make_payload(args, build_prompt(text, chars, nonce))

        def task(payload=payload):
            result = post_stream(url, payload, args.timeout)
            with lock:
                results.append(result)

        thread = threading.Thread(target=task, daemon=True)
        thread.start()
        threads.append(thread)
        if index + 1 < args.requests:
            time.sleep(random.expovariate(args.request_rate))
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


def make_counter():
    lock = threading.Lock()
    state = {"n": 0}

    def next_index():
        with lock:
            value = state["n"]
            state["n"] += 1
            return value

    return next_index


def make_payload(args, prompt):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.no_think:
        payload["think"] = False
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent serving benchmark for ds4-server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--prompt-file", default="speed-bench/promessi_sposi.txt")
    parser.add_argument("--prompt-tokens", type=int, default=1024,
                        help="target prompt length; 0 sends a one-line prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=0.0,
                        help="Poisson arrivals per second; 0 keeps the closed loop")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="generate exactly --max-tokens per request")
    parser.add_argument("--no-think", action="store_true",
                        help="disable thinking so output length is predictable")
    parser.add_argument("--shared-prefix", action="store_true",
                        help="reuse one prompt so the prefix cache is measured")
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed requests sent before the measured run")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", dest="json_path",
                        help="write the metrics to this file")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) <= 0 or min(args.prompt_tokens, args.warmup) < 0:
        parser.error("requests, concurrency and max-tokens must be positive; prompt-tokens and warmup cannot be negative")
    if not math.isfinite(args.request_rate) or args.request_rate < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("request-rate must be finite and nonnegative; timeout must be finite and positive")
    args.prefix_nonce = "shared"

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    try:
        with open(args.prompt_file, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read()
    except OSError as exc:
        print(f"serve-concurrency-bench: {exc}", file=sys.stderr)
        return 1

    chars = 0
    if args.prompt_tokens > 0:
        try:
            ratio = calibrate_chars_per_token(url, args.model, text, args.timeout)
        except Exception as exc:  # noqa: BLE001 - any failure here is fatal and worth showing
            print(f"serve-concurrency-bench: calibration failed: {exc}", file=sys.stderr)
            return 1
        chars = int(args.prompt_tokens * ratio)
        if chars >= len(text):
            print(f"serve-concurrency-bench: {args.prompt_file} is too short for "
                  f"{args.prompt_tokens} tokens (needs about {chars} characters)",
                  file=sys.stderr)
            return 1
        print(f"serve-concurrency-bench: {ratio:.2f} characters per token, "
              f"{chars} characters per prompt")

    for i in range(args.warmup):
        nonce = f"warmup-{i}-{random.getrandbits(48):012x}"
        post_stream(url, make_payload(args, build_prompt(text, chars, nonce)),
                    args.timeout)

    results = []
    lock = threading.Lock()
    if args.request_rate > 0.0:
        wall = run_poisson(args, url, text, chars, results, lock)
        shape = f"poisson {args.request_rate} req/s"
    else:
        wall = run_closed_loop(args, url, text, chars, results, lock)
        shape = f"concurrency {args.concurrency}"

    failures = [r for r in results if not r.ok]
    for result in failures[:3]:
        print(f"serve-concurrency-bench: request failed: {result.error}",
              file=sys.stderr)

    stats = summarize(results, wall)
    label = args.label or f"{shape}, prompt {args.prompt_tokens}, out {args.max_tokens}"
    print_summary(label, stats)

    if args.json_path:
        stats["config"] = {
            "model": args.model,
            "shape": shape,
            "concurrency": args.concurrency,
            "request_rate": args.request_rate,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "requests": args.requests,
            "shared_prefix": args.shared_prefix,
        }
        with open(args.json_path, "w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)
        print(f"\nwrote {args.json_path}")

    return 0 if stats["completed"] == args.requests and not stats["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
