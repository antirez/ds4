#!/usr/bin/env python3
"""V4.1 HTTP/session plumbing with the sparse zero-weight GGUF, NOT model quality."""
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="sparse zero-weight V4.1 fixture")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", default=0, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", args.port))
        args.port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{args.port}"
    trace = args.output / "trace.log"
    model = "deepseek-v4.1-flash"

    def request(path, body=None):
        req = urllib.request.Request(base + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = response.read().decode()
        except urllib.error.HTTPError as error:
            raise AssertionError(error.read().decode()) from error
        return data if body and body.get("stream") else json.loads(data)

    with (args.output / "server.log").open("w") as log:
        proc = subprocess.Popen(["./ds4-server", "-m", args.model,
            "--ssd-streaming", "--ssd-streaming-cache-experts", "512",
            "--ctx", "512", "--tokens", "4", "--host", "127.0.0.1",
            "--port", str(args.port), "--trace", str(trace),
            "--kv-disk-dir", str(args.output / "kv"), "--kv-disk-space-mb", "128"],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert proc.poll() is None, "server exited during startup"
                try:
                    models = request("/v1/models")
                    break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, "server startup timed out"
                    time.sleep(0.2)
            assert any(item["id"] == model for item in models["data"])
            body = {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0, "reasoning_effort": "none", "max_tokens": 1}
            cold = request("/v1/chat/completions", body)
            warm = request("/v1/chat/completions", body)
            assert cold["choices"][0]["message"] == warm["choices"][0]["message"]
            body["messages"] += [warm["choices"][0]["message"], {"role": "user", "content": "Again"}]
            continued = request("/v1/chat/completions", body)
            assert continued["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
            body["messages"] = [{"role": "user", "content": "Read a.c"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_fixture",
                 "type": "function", "function": {"name": "read", "arguments": '{"path":"a.c"}'}}]},
                {"role": "tool", "tool_call_id": "call_fixture", "content": "int main(void) {}"}]
            tool = request("/v1/chat/completions", body)
            assert tool["choices"][0]["finish_reason"] == "length"
            body["stream"] = True
            stream = request("/v1/chat/completions", body)
            assert "data: [DONE]" in stream
            for line in stream.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    json.loads(line[6:])
            response = request("/v1/responses", {"model": model, "input": "Hi",
                "reasoning": {"effort": "none"}, "max_output_tokens": 4, "temperature": 0})
            assert response["object"] == "response"
            anthropic = request("/v1/messages", {"model": model,
                "system": "Be precise.", "messages": [{"role": "user", "content": "Anthropic greeting"}],
                "thinking": {"type": "disabled"}, "max_tokens": 1, "temperature": 0})
            assert anthropic["type"] == "message"
            (args.output / "results.json").write_text(json.dumps({
                "cold": cold, "warm": warm, "continued": continued,
                "tool_replay": tool, "responses": response, "anthropic": anthropic}, indent=2))
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    assert proc.returncode == 0, f"server exit status {proc.returncode}"
    assert "<｜DSML｜ calls>" in trace.read_text()
    assert "<｜begin▁of▁sentence｜><｜System｜>Be precise.<｜User｜>Anthropic greeting" in trace.read_text()
    print("PASS: V4.1 model ID, repeat request, continued cache, tool replay, SSE, Responses, clean shutdown")


if __name__ == "__main__":
    main()
