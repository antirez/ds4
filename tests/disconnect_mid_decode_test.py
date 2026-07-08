#!/usr/bin/env python3
"""disconnect_mid_decode_test.py — integration harness for ds4-server slot
release on client disconnect during decode.

Unlike the in-process unit tests (ds4_test), this exercises the real server:
a client opens a streaming completion, reads a few decoded tokens, then hard-
closes the socket mid-decode. The server handles this reactively (a failed SSE
write sets finish="error" + stops the decode; SIGPIPE is ignored; POLLHUP/
POLLERR are polled). This harness proves the observable contract:

  1. baseline        — a normal streaming completion returns output;
  2. slot is freed   — after aborting a stream mid-decode, a fresh request
                       still completes (the slot did not leak / hang);
  3. peer unaffected — a concurrent completion on another slot finishes
                       normally while a sibling stream is aborted mid-decode.

NOTE on thinking: ds4 / DeepSeek V4 Flash default to reasoning ("THINKING"),
so the first tokens of a turn stream as `delta.reasoning_content`, not
`delta.content`. This harness counts BOTH as decode output, otherwise the
abort would never trigger (no visible content until thinking closes) and the
serve-checks would see an "empty" but actually-served response.

It targets an ALREADY-RUNNING dev server (starting one needs the model + GPU,
which the operator provides). Run against the ds4-dev worktree binary on a
non-prod port, e.g.:

    # on the VM, prod stopped or dev on its own port:
    /data/src/ds4-dev/ds4-server --port 8011 --parallel 4 --ctx 32768 \
        --prefill-chunk 1024 <flash-q2.gguf> &
    python3 tests/disconnect_mid_decode_test.py --base http://127.0.0.1:8011

For the STRICTEST single-slot proof (a leaked slot makes the follow-up hang),
run the dev server with --parallel 1. With --parallel N > 1 the harness runs
more abort rounds than slots so a leak would still exhaust them.

Exit code 0 = all scenarios PASS, non-zero = failure. Pure stdlib (no deps).
"""

import argparse
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlparse

LONG_PROMPT = (
    "Explain, step by step and at length, how a thermostatic expansion valve "
    "regulates superheat in a direct-expansion refrigeration circuit, and why "
    "hunting can occur. Be thorough."
)


def _conn(host, port, timeout):
    return http.client.HTTPConnection(host, port, timeout=timeout)


def discover_model(host, port, fallback):
    """Pick a model id from /v1/models, falling back to the CLI default."""
    try:
        c = _conn(host, port, 5)
        c.request("GET", "/v1/models")
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        c.close()
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        if ids:
            return ids[0]
    except Exception as e:  # noqa: BLE001
        print(f"  (/v1/models probe failed: {e}; using fallback '{fallback}')")
    return fallback


def _open_stream(host, port, model, prompt, max_tokens, timeout):
    c = _conn(host, port, timeout)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    })
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    if resp.status != 200:
        payload = resp.read()
        c.close()
        raise RuntimeError(f"HTTP {resp.status}: {payload[:200]!r}")
    return c, resp


def _iter_output(resp):
    """Yield (kind, piece) for each content/reasoning delta until [DONE]/EOF.
    kind is "content" or "reasoning" — both count as decode progress."""
    while True:
        line = resp.readline()
        if not line:
            return
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                yield ("content", d["content"])
            if d.get("reasoning_content"):
                yield ("reasoning", d["reasoning_content"])


def run_to_completion(host, port, model, prompt, max_tokens=48, timeout=120):
    """Return (content, reasoning) text produced by a full streaming turn."""
    c, resp = _open_stream(host, port, model, prompt, max_tokens, timeout)
    content, reasoning = [], []
    try:
        for kind, piece in _iter_output(resp):
            (content if kind == "content" else reasoning).append(piece)
    finally:
        c.close()
    return "".join(content), "".join(reasoning)


def decode_then_abort(host, port, model, prompt, read_tokens=4, timeout=120):
    """Open a stream, read a few decoded tokens (content or reasoning), then
    hard-close the socket mid-decode so the next server write fails."""
    c, resp = _open_stream(host, port, model, prompt, 256, timeout)
    got = 0
    for _ in _iter_output(resp):
        got += 1
        if got >= read_tokens:
            break
    # Hard close mid-decode: drops the TCP connection so the next server write
    # fails (EPIPE / POLLHUP) and the slot must be released.
    c.close()
    return got


def _served(content, reasoning):
    return bool(content.strip() or reasoning.strip())


def scenario_baseline(host, port, model):
    content, reasoning = run_to_completion(host, port, model,
                                           "Say hello in one short sentence.")
    ok = _served(content, reasoning)
    print(f"[1] baseline completion .......... {'PASS' if ok else 'FAIL'} "
          f"({len(content)} content / {len(reasoning)} reasoning chars)")
    return ok


def scenario_slot_freed(host, port, model, rounds):
    ok = True
    for i in range(rounds):
        got = decode_then_abort(host, port, model, LONG_PROMPT, read_tokens=4)
        if got < 1:
            print(f"    round {i}: decode never started before abort (got {got})")
            ok = False
        # Small settle so the worker observes the disconnect before we re-ask.
        time.sleep(0.2)
        content, reasoning = run_to_completion(host, port, model,
                                               "Reply with a single word: ok.",
                                               max_tokens=16)
        if not _served(content, reasoning):
            print(f"    round {i}: server did not serve after abort")
            ok = False
            break
    print(f"[2] slot freed after abort (x{rounds}) {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_peer_unaffected(host, port, model):
    result = {}

    def peer():
        try:
            result["out"] = run_to_completion(host, port, model, LONG_PROMPT,
                                              max_tokens=64)
        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)

    t = threading.Thread(target=peer)
    t.start()
    # Give the peer a head start onto its own slot, then abort a sibling stream.
    time.sleep(0.5)
    try:
        decode_then_abort(host, port, model, LONG_PROMPT, read_tokens=3)
    except Exception as e:  # noqa: BLE001
        print(f"    sibling abort raised (non-fatal): {e}")
    t.join(timeout=180)
    served = "out" in result and _served(*result["out"])
    ok = (not t.is_alive() and "error" not in result and served)
    if "error" in result:
        detail = result["error"]
    elif "out" in result:
        c, r = result["out"]
        detail = f"{len(c)} content / {len(r)} reasoning chars"
    else:
        detail = "peer did not finish"
    print(f"[3] peer unaffected by abort ..... {'PASS' if ok else 'FAIL'} "
          f"({detail})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8010",
                    help="ds4-server base URL (default prod-adjacent :8010)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="model id fallback if /v1/models is empty")
    ap.add_argument("--rounds", type=int, default=6,
                    help="disconnect/re-serve rounds for scenario 2 "
                         "(keep > --parallel so a slot leak would exhaust them)")
    args = ap.parse_args()

    u = urlparse(args.base)
    host, port = u.hostname, u.port or 8000
    model = discover_model(host, port, args.model)
    print(f"target {host}:{port}  model={model}")

    results = [
        scenario_baseline(host, port, model),
        scenario_slot_freed(host, port, model, args.rounds),
        scenario_peer_unaffected(host, port, model),
    ]
    if all(results):
        print("disconnect_mid_decode_test: OK")
        return 0
    print("disconnect_mid_decode_test: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
