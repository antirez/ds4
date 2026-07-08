#!/usr/bin/env python3
"""disconnect_mid_decode_test.py — integration harness for ds4-server slot
release on client disconnect during decode.

Unlike the in-process unit tests (ds4_test), this exercises the real server:
a client opens a streaming completion, reads a few decoded tokens, then hard-
closes the socket mid-decode. The server handles this reactively (a failed SSE
write sets finish="error" + stops the decode; SIGPIPE is ignored; POLLHUP/
POLLERR are polled). This harness proves the observable contract:

  1. baseline        — a normal streaming completion returns tokens;
  2. slot is freed   — after aborting a stream mid-decode, a fresh request
                       still completes (the slot did not leak / hang);
  3. peer unaffected — a concurrent completion on another slot finishes
                       normally while a sibling stream is aborted mid-decode.

It targets an ALREADY-RUNNING dev server (starting one needs the model + GPU,
which the operator provides). Run against the ds4-dev worktree binary on a
non-prod port, e.g.:

    # on the VM, prod stopped or dev on its own port:
    /data/src/ds4-dev/ds4-server --port 8011 --parallel 4 --ctx 32768 \
        --prefill-chunk 1024 <flash-q2.gguf> &
    python3 tests/disconnect_mid_decode_test.py --base http://127.0.0.1:8011

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


def _iter_deltas(resp):
    """Yield assistant content deltas from an SSE stream until [DONE]/EOF."""
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
            piece = ch.get("delta", {}).get("content")
            if piece:
                yield piece


def run_to_completion(host, port, model, prompt, max_tokens=48, timeout=120):
    c, resp = _open_stream(host, port, model, prompt, max_tokens, timeout)
    try:
        text = "".join(_iter_deltas(resp))
    finally:
        c.close()
    return text


def decode_then_abort(host, port, model, prompt, read_tokens=4, timeout=120):
    """Open a stream, read a few decoded tokens, then hard-close the socket."""
    c, resp = _open_stream(host, port, model, prompt, 256, timeout)
    got = 0
    for _ in _iter_deltas(resp):
        got += 1
        if got >= read_tokens:
            break
    # Hard close mid-decode: drops the TCP connection so the next server write
    # fails (EPIPE / POLLHUP) and the slot must be released.
    c.close()
    return got


def scenario_baseline(host, port, model):
    text = run_to_completion(host, port, model, "Say hello in one short sentence.")
    ok = len(text.strip()) > 0
    print(f"[1] baseline completion .......... {'PASS' if ok else 'FAIL'} "
          f"({len(text)} chars)")
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
        text = run_to_completion(host, port, model,
                                 "Reply with a single word: ok.", max_tokens=16)
        if len(text.strip()) == 0:
            print(f"    round {i}: server did not serve after abort")
            ok = False
            break
    print(f"[2] slot freed after abort (x{rounds}) {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_peer_unaffected(host, port, model):
    result = {}

    def peer():
        try:
            result["text"] = run_to_completion(host, port, model, LONG_PROMPT,
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
    ok = (not t.is_alive() and "error" not in result
          and len(result.get("text", "").strip()) > 0)
    detail = result.get("error") or f"{len(result.get('text', ''))} chars"
    print(f"[3] peer unaffected by abort ..... {'PASS' if ok else 'FAIL'} "
          f"({detail})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8010",
                    help="ds4-server base URL (default prod-adjacent :8010)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="model id fallback if /v1/models is empty")
    ap.add_argument("--rounds", type=int, default=3,
                    help="disconnect/re-serve rounds for scenario 2")
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
