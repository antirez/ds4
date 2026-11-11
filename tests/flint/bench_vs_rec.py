#!/usr/bin/env python3
"""Step 3 of docs/RECURSIVE_FLINT.md: rec and ds4 on the same prompts through
the same OpenAI-style HTTP harness, so both pay identical client overheads.

For each frontier (prompt length in tokens) the script sends one streaming
chat completion with the first N tokens of the story prompt as the user
message, temperature 0, max_tokens M, and records the time to the first
streamed token (prefill) and the rate of the remaining tokens (decode).  Run
it against `rec serve` (port 11500) and against `ds4-server` (never both at
once: 48 GB), then compare the two tables.

    rec serve --context 40000 &  ...  tests/flint/bench_vs_rec.py --engine rec
    ./ds4-server -m flint-int4.gguf --ctx 40000 --port 18089 &  ...  --engine ds4

Prefill times include the chat template and the first token's decode step.
rec sends its tokens in bursts of 8 (one event per token, eight events a few
ms apart, then a step gap of 8 decode steps), so the first event arrives 7
decode steps after the first token: the decode rate counts from the end of
the first burst, and the prefill time has the burst's extra steps taken out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def tokenizer():
    from tokenizers import Tokenizer
    models = Path.home() / ".local/share/rec/models"
    cands = sorted(models.glob("*/tokenizer.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        sys.exit("rec's tokenizer.json not found")
    return Tokenizer.from_file(str(cands[-1]))


def auth_headers(engine):
    h = {"Content-Type": "application/json"}
    if engine == "rec":   # rec serve wants its local token (`rec auth token`)
        key = subprocess.run([str(Path.home() / ".local/bin/rec"), "auth", "token"], capture_output=True, text=True).stdout.strip()
        if key:
            h["Authorization"] = "Bearer " + key
    return h


def stream_once(url, model, text, max_tokens, think, headers, tok, burst_s):
    """Returns (prefill_s, completion_tokens, decode_s_per_token).  Tokens are
    counted by re-tokenizing the streamed text (events are not tokens);
    usage.completion_tokens wins when the server sends it.  The decode rate is
    the time from the first burst's last event to the last event over the
    tokens after the first burst (a burst: events within burst_s of the
    first; 10 ms for rec, whose eight events arrive within about 5 ms, 0 for
    ds4, which sends one event per decode step)."""
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    t_first = None
    t_burst = None          # last event of the first burst
    burst_pieces = []
    pieces = []
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage") and ev["usage"].get("completion_tokens"):
                usage_tokens = ev["usage"]["completion_tokens"]
            choices = ev.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            text_part = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
            if text_part:
                pieces.append(text_part)
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                if now - t_first <= burst_s:
                    t_burst = now
                    burst_pieces.append(text_part)
    t_end = time.perf_counter()
    if t_first is None:
        return None
    n = usage_tokens or len(tok.encode("".join(pieces), add_special_tokens=False).ids)
    n_burst = max(1, len(tok.encode("".join(burst_pieces), add_special_tokens=False).ids))
    per_token = (t_end - t_burst) / max(1, n - n_burst)
    first_token = t_first - t0 - (n_burst - 1) * per_token
    return first_token, n, per_token


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["rec", "ds4"], required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--prompt-file", type=Path, default=ROOT / "tests/long_context_story_prompt.txt")
    ap.add_argument("--frontiers", type=int, nargs="+", default=[256, 4096, 16384, 28000])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=2, help="runs per frontier; the best is reported")
    ap.add_argument("--no-think", action="store_true", help="prefill an empty think block (ds4 only; rec has no switch)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the server to answer /v1/models")
    args = ap.parse_args()
    args.think = not args.no_think
    url = args.url or ("http://127.0.0.1:11500" if args.engine == "rec" else "http://127.0.0.1:18089")
    model = "flint"
    headers = auth_headers(args.engine)
    deadline = time.time() + args.wait
    while True:
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers=headers), timeout=2).read()
            break
        except Exception:
            if time.time() > deadline:
                sys.exit("server at %s did not come up" % url)
            time.sleep(2)
    tok = tokenizer()
    text = args.prompt_file.read_text()
    ids = tok.encode(text, add_special_tokens=False).ids
    print("%s via %s; prompt file has %d tokens; %d new tokens per run, best of %d" % (
        args.engine, url, len(ids), args.max_tokens, args.repeat))
    print("%9s %10s %12s %10s %12s" % ("prompt", "prefill s", "prefill tok/s", "decode ms", "decode tok/s"))
    # each frontier starts at its own offset so no prompt is a prefix of another
    # (both servers cache prefixes); the first run's prefill is the cold one and
    # the decode rate is the best run's
    for i, n in enumerate(args.frontiers):
        off = (i * 977) % max(1, len(ids) - n) if n < len(ids) else 0
        if n > len(ids):
            print("%9d (prompt file too short)" % n)
            continue
        piece = tok.decode(ids[off:off + n])
        first = None
        best_dec = None
        for _ in range(args.repeat):
            r = stream_once(url, model, piece, args.max_tokens, args.think, headers, tok, 0.010 if args.engine == "rec" else 0.0)
            if r is None:
                continue
            if first is None:
                first = r
            per = r[2]
            if best_dec is None or per < best_dec:
                best_dec = per
        if first is None:
            print("%9d no tokens streamed" % n)
            continue
        print("%9d %10.2f %12.0f %10.1f %12.1f   (%d tokens)" % (n, first[0], n / first[0], 1000 * best_dec, 1 / best_dec if best_dec else 0, first[1]))


if __name__ == "__main__":
    main()
