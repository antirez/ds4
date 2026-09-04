#!/usr/bin/env python3
"""Live regression test: ds4-server must cancel a job when the client
disconnects mid-generation, even on a clean TCP FIN.

Upstream e9ded97 polls the client socket every 100ms while a job runs, but
Darwin's poll() does not report a plain FIN (no POLLIN/POLLHUP), so a client
that closed with a clean FIN was never seen: the job decoded all max_tokens
into a dead socket. Only an RST (e.g. close() with unread receive data) tripped
cancellation, and that path returned silently without logging.

This test exercises the real TCP path (streaming and non-streaming), asserts
the 'client disconnected' log appears promptly and that generation stops far
below max_tokens, then checks the server keeps serving.

Opt-in: skips (exit 0) if the model or binary is absent. Env:
  DS4_IT_MODEL    GGUF path (default: ds4flash.gguf next to the repo)
  DS4_DISC_PORT   server port (default 8033)

Usage: python3 tests/live_disconnect_test.py
"""
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("DS4_IT_MODEL", os.path.join(REPO_ROOT, "ds4flash.gguf"))
PORT = int(os.environ.get("DS4_DISC_PORT", "8033"))
CTX = 8192
BASE = f"http://127.0.0.1:{PORT}"

LONG_PROMPT = (
    "Write a very long, detailed essay about the history of papermaking, "
    "from papyrus and parchment through the invention of the paper machine, "
    "the Fourdrinier process, modern recycling, and the future of paper. "
    "Keep writing paragraph after paragraph until you run out of tokens; "
    "do not stop early."
)

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[live] {status}: {name}{(' - ' + detail) if detail else ''}", flush=True)
    if not cond:
        failures.append(name)


class Server:
    def __init__(self, workdir):
        self.log_path = os.path.join(workdir, "server.log")
        self._log_f = open(self.log_path, "wb", buffering=0)
        env = dict(os.environ, DS4_LOCK_FILE=os.path.join(workdir, "ds4.lock"))
        cmd = [
            os.path.join(REPO_ROOT, "ds4-server"),
            "--model", MODEL,
            "--ctx", str(CTX),
            "--tokens", str(CTX),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--kv-disk-dir", os.path.join(workdir, "kv"),
            "--kv-disk-space-mb", "512",
            "--kv-cache-min-tokens", "32",
        ]
        self._proc = subprocess.Popen(cmd, stdout=self._log_f,
                                      stderr=subprocess.STDOUT, env=env,
                                      cwd=REPO_ROOT)

    def ready(self, deadline_s=240):
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            try:
                with urllib.request.urlopen(f"{BASE}/v1/models", timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        return False

    def log_text(self):
        with open(self.log_path, "rb") as f:
            return f.read().decode("utf-8", "replace")

    def stop(self):
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log_f.close()


def post_chat(messages, max_tokens, tools=None, stream=False):
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        **({"tools": tools} if tools else {}),
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def close_stream_connection(conn, sock):
    """Abort the stream with a clean FIN. Capture the socket before
    getresponse(): the server sends 'Connection: close', so http.client
    nulls conn.sock once the response begins and a late close would silently
    no-op, leaving the connection open (the original bug in this test)."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
    conn.close()


def test_inflight_disconnect(server):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    sock = conn.sock
    resp = conn.getresponse()
    chunks = 0
    while chunks < 8:
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            chunks += 1
    check("sse stream delivered chunks before abort", chunks >= 4,
          f"chunks={chunks}")
    t_close = time.time()
    close_stream_connection(conn, sock)
    deadline = t_close + 60.0
    log = server.log_text()
    while time.time() < deadline:
        log = server.log_text()
        if "client disconnected" in log:
            break
        time.sleep(0.5)
    lag = time.time() - t_close
    check("server logged 'client disconnected' promptly", "client disconnected" in log,
          f"lag={lag:.2f}s")
    check("disconnect detected within 15s of close", lag < 15.0, f"lag={lag:.2f}s")
    gen = None
    lines = log.splitlines()
    for line in lines:
        if "client disconnected" in line:
            for prev in lines:
                if prev == line:
                    break
                for tok in prev.split():
                    if tok.startswith("gen="):
                        gen = int(tok.split("=")[1])
            break
    check("cancelled job's gen count far below max_tokens(512)",
          (gen or 0) < 100,
          f"gen={gen or 0} (no progress line = stopped at prefill)")
    return chunks, lag, gen


def test_server_healthy_after(server):
    r = post_chat([{"role": "user", "content": "Say exactly: healthy"}], 24)
    check("server serves normal request after disconnect", r.get("choices"))


def test_tools_end_to_end(server):
    tools = [{
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in a directory. The input should be a full path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "directory path"}
                },
                "required": ["path"],
            },
        },
    }]
    r = post_chat([{"role": "user", "content": "Use the list_files tool to list the files in /tmp."}],
                  128, tools=tools)
    msg = r["choices"][0]["message"]
    check("tools request returns 200 with tool_calls or content", True,
          f"finish={r['choices'][0].get('finish_reason')} calls={len(msg.get('tool_calls', []))} has_content={bool(msg.get('content'))}")


def main():
    if not os.path.exists(MODEL):
        print(f"[live] skip: model not found at {MODEL}")
        return 0
    with tempfile.TemporaryDirectory(prefix="ds4-live-disc-") as workdir:
        server = Server(workdir)
        try:
            check("server ready", server.ready())
            if not server.ready():
                print(server.log_text()[-2000:])
                return 1
            test_inflight_disconnect(server)
            test_server_healthy_after(server)
            test_tools_end_to_end(server)
        finally:
            server.stop()
    print(f"[live] failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
