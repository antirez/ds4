#!/usr/bin/env python3
"""pulsar-model-router — route Anthropic-protocol requests to a backend by model name.

Claude Code sends ALL traffic (main model + haiku-class background tasks) to a
single ANTHROPIC_BASE_URL; the only difference between requests is the "model"
field in the JSON body (upstream feature requests for split endpoints were
declined: anthropics/claude-code#25146, #24160). This router makes the split:

    model matches SMALL_PATTERN  ->  SMALL_UPSTREAM  (haiku-class chores)
    everything else              ->  MAIN_UPSTREAM   (the big model)

Both upstreams must speak the Anthropic /v1/messages protocol themselves
(pulsar-server does; llama.cpp and vLLM do too). The router never re-serializes
anything: the request body and the entire upstream response are relayed as raw
bytes, so replay byte-stability — which pulsar's prefix cache depends on — is
preserved exactly. The body is parsed once, read-only, to find "model".

No dependencies beyond the Python 3 stdlib.

Usage:
    pulsar-model-router.py [--listen HOST:PORT] [--main HOST:PORT]
                           [--small HOST:PORT] [--small-pattern REGEX]

Defaults: --listen 0.0.0.0:8100 --main 127.0.0.1:8000
          --small ellie.defense.lan:8000 --small-pattern haiku
"""

import argparse
import json
import re
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_TIMEOUT_S = 3600  # prefills at deep context can run minutes
RELAY_CHUNK = 65536

ARGS = None


def pick_upstream(body: bytes) -> tuple[str, int, str]:
    """Return (host, port, why) for this request body."""
    model = ""
    try:
        model = str(json.loads(body).get("model", ""))
    except (ValueError, AttributeError):
        pass
    if model and re.search(ARGS.small_pattern, model, re.IGNORECASE):
        return (*ARGS.small_hp, f"model={model!r} -> small")
    return (*ARGS.main_hp, f"model={model!r} -> main")


class Router(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # We write raw bytes to the socket ourselves; silence default logging noise.

    def log_message(self, fmt, *args):
        pass

    def _relay(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""

        if self.command == "POST":
            host, port, why = pick_upstream(body)
        else:
            host, port = ARGS.main_hp
            why = "non-POST -> main"
        print(f"{self.command} {self.path} [{why}] -> {host}:{port}",
              file=sys.stderr, flush=True)

        # Rebuild the request verbatim, minus hop-by-hop headers.
        head = [f"{self.command} {self.path} HTTP/1.1", f"Host: {host}:{port}"]
        for k, v in self.headers.items():
            if k.lower() in ("host", "connection", "keep-alive",
                             "transfer-encoding"):
                continue
            head.append(f"{k}: {v}")
        head.append("Connection: close")
        raw = ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body

        try:
            up = socket.create_connection((host, port),
                                          timeout=UPSTREAM_TIMEOUT_S)
        except OSError as e:
            self.send_error(502, f"upstream {host}:{port} unreachable: {e}")
            return
        try:
            up.sendall(raw)
            # Relay the raw response byte stream (status line, headers, body —
            # including SSE) until the upstream closes. No parsing at all.
            while True:
                chunk = up.recv(RELAY_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
            self.wfile.flush()
        except OSError:
            pass  # client or upstream went away mid-stream
        finally:
            up.close()
            self.close_connection = True

    do_GET = do_POST = do_OPTIONS = do_DELETE = _relay


def hostport(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host, int(port)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", default="0.0.0.0:8100")
    ap.add_argument("--main", default="127.0.0.1:8000",
                    help="upstream for the main model (pulsar-server)")
    ap.add_argument("--small", default="ellie.defense.lan:8000",
                    help="upstream for the small/fast model")
    ap.add_argument("--small-pattern", default="haiku",
                    help="case-insensitive regex on the request's model field")
    ARGS = ap.parse_args()
    ARGS.main_hp = hostport(ARGS.main)
    ARGS.small_hp = hostport(ARGS.small)

    lhost, lport = hostport(ARGS.listen)
    srv = ThreadingHTTPServer((lhost, lport), Router)
    srv.daemon_threads = True
    print(f"pulsar-model-router listening on {lhost}:{lport} "
          f"(main -> {ARGS.main}, small[{ARGS.small_pattern}] -> {ARGS.small})",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
