"""Live turn-boundary cache regression; requires a self-contained model GGUF.

The server renders every assistant turn closed by the model's end token.  If the
live session does not keep that token, the next request's exact token prefix
diverges at the turn boundary and stays diverged: the conversation drifts away
from the live KV until a request is answered from a disk checkpoint or
re-prefilled in full, even though the client only appended to the prompt.

Every continuation request below must therefore reuse the whole previous turn:

  cached_tokens >= previous prompt_tokens
  cached_tokens == live_tokens_before
  disk_cached_tokens == 0
  prompt_tokens - cached_tokens < 64

Tool-call turns matter most: the server stops as soon as </tool_call> is parsed
and never samples the turn's end token, so nothing else can put it in the KV.

python tests/test_server_turn_boundary.py --model MODEL
"""

import argparse
import json
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.request


def wait_ready(proc, base):
    deadline = time.monotonic() + 300
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=1) as response:
                json.load(response)
            break
        except (OSError, TimeoutError):
            if time.monotonic() > deadline:
                raise RuntimeError("startup timeout")
            time.sleep(1)


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def post(base, body):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def decisions(trace):
    """Parse the server's cache-decision trace blocks, oldest first."""
    if not trace.exists():
        return []
    blocks = []
    for chunk in trace.read_text(errors="replace").split("--- cache decision ---")[1:]:
        block = {}
        for line in chunk.lstrip("\r\n").splitlines():
            if ": " in line and not line.startswith(" "):
                key, _, value = line.partition(": ")
                block[key.strip()] = value.strip()
            elif not line.strip():
                break
        if "cache_source" in block:
            blocks.append(block)
    return blocks


def assert_turn_boundary(label, row, block, previous_prompt_tokens):
    usage = row["usage"]
    prompt = usage["prompt_tokens"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert prompt - cached < 64, (label, "turn was re-prefilled", usage)
    assert cached >= previous_prompt_tokens, (
        label,
        "reuse behind the previous prompt",
        usage,
        previous_prompt_tokens,
        block,
    )
    # Omitted reasoning and sampled BPE segmentation can shorten the
    # retokenized common prefix even when the visible-text tier reuses all KV.
    assert cached == int(block["live_tokens_before"]), (
        label,
        "live KV frontier was not fully reused",
        usage,
        block,
    )
    assert int(block["disk_cached_tokens"]) == 0, (label, "disk fallback", block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out = (args.output or pathlib.Path(tempfile.mkdtemp(prefix="ds4-boundary-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=out))
    logpath = out / "server.log"
    trace = out / "server.trace"
    print("Artifacts:", out, flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    cmd = [
        str(root / "ds4-server"),
        "-m",
        str(pathlib.Path(args.model).resolve()),
        "--ctx",
        "8192",
        "--port",
        str(port),
        "--prefill-chunk",
        "1024",
        "--mtp",
        "--mtp-exact-sampling",
        "--kv-disk-dir",
        str(cache),
        "--kv-disk-space-mb",
        "256",
        "--kv-cache-min-tokens",
        "128",
        "--kv-cache-continued-interval-tokens",
        "1024",
        "--trace",
        str(trace),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a stored record. Call it whenever a request names a record.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    rows = []
    with logpath.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=log)
        try:
            wait_ready(proc, base)

            # Tool-less follow-ups: the sampled end token must reach the live KV.
            history = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer arithmetic yourself. Think briefly.",
                },
                {"role": "user", "content": "What is 17 multiplied by 3? Answer with just the number."},
            ]
            previous_prompt = 0
            for turn in range(3):
                result = post(
                    base,
                    {
                        "model": "qwen",
                        "messages": history,
                        "temperature": 0,
                        "max_tokens": 256,
                        "reasoning_effort": "low",
                    },
                )
                row = {
                    "case": "follow-up",
                    "turn": turn + 1,
                    "finish": result["choices"][0]["finish_reason"],
                    "usage": result["usage"],
                }
                block = decisions(trace)[-1]
                row["cache_source"] = block["cache_source"]
                row["live_prompt_common"] = block["live_prompt_common"]
                print(json.dumps(row), flush=True)
                rows.append(row)
                if turn:
                    assert_turn_boundary("follow-up", row, block, previous_prompt)
                previous_prompt = result["usage"]["prompt_tokens"]
                message = result["choices"][0]["message"]
                history += [
                    {"role": "assistant", "content": message.get("content") or ""},
                    {
                        "role": "user",
                        "content": "Now add 2 to that number. Answer with just the number.",
                    },
                ]

            # Tool-call turn: the server cuts generation before the end token.
            history = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Call the lookup tool when a request names a record. Think briefly.",
                },
                {"role": "user", "content": "Look up the harbor tide tables record and report it."},
            ]
            called = None
            for attempt in range(3):
                result = post(
                    base,
                    {
                        "model": "qwen",
                        "messages": history,
                        "tools": tools,
                        "temperature": 0,
                        "max_tokens": 256,
                        "reasoning_effort": "low",
                    },
                )
                if result["choices"][0]["finish_reason"] == "tool_calls":
                    called = result
                    break
                history[-1] = {
                    "role": "user",
                    "content": "Call the lookup tool now for the harbor tide tables record. Do not answer from memory.",
                }
            assert called is not None, "model never called the lookup tool"
            message = called["choices"][0]["message"]
            call = message["tool_calls"][0]
            previous_prompt = called["usage"]["prompt_tokens"]
            history += [
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]},
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "harbor tide tables: high 03:14, low 09:41",
                },
            ]
            for turn in range(2):
                result = post(
                    base,
                    {
                        "model": "qwen",
                        "messages": history,
                        "tools": tools,
                        "temperature": 0,
                        "max_tokens": 256,
                        "reasoning_effort": "low",
                    },
                )
                row = {
                    "case": "tool-turn",
                    "turn": turn + 1,
                    "finish": result["choices"][0]["finish_reason"],
                    "usage": result["usage"],
                }
                block = decisions(trace)[-1]
                row["cache_source"] = block["cache_source"]
                row["live_prompt_common"] = block["live_prompt_common"]
                print(json.dumps(row), flush=True)
                rows.append(row)
                assert_turn_boundary("tool-turn", row, block, previous_prompt)
                previous_prompt = result["usage"]["prompt_tokens"]
                message = result["choices"][0]["message"]
                history += [
                    {"role": "assistant", "content": message.get("content") or ""},
                    {"role": "user", "content": "Report the low tide time only."},
                ]
            (out / "results.json").write_text(json.dumps(rows, indent=2))
        finally:
            stop(proc)
    logtext = logpath.read_text()
    assert "turn end token not kept in live KV" not in logtext, logpath
    print("Artifacts:", out, "cache:", cache, flush=True)


if __name__ == "__main__":
    main()
