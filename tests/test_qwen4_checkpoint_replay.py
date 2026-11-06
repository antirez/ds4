"""Live Qwen checkpoint regression; requires a self-contained model GGUF.

python tests/test_qwen4_checkpoint_replay.py --model MODEL
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out = (args.output or pathlib.Path(tempfile.mkdtemp(prefix="ds4-qwen-replay-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=out))
    logpath = out / "server.log"
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
        "16384",
        "--port",
        str(port),
        "--prefill-chunk",
        "1024",
        "--mtp",
        "--mtp-exact-sampling",
        "--kv-disk-dir",
        str(cache),
        "--kv-disk-space-mb",
        "512",
        "--kv-cache-min-tokens",
        "128",
        "--kv-cache-cold-max-tokens",
        "0",
        "--kv-cache-continued-interval-tokens",
        "1024",
        "--kv-cache-boundary-align-tokens",
        "128",
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up information only when explicitly requested.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    results = []
    with logpath.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=log)
        try:
            wait_ready(proc, base)
            for name, has_tools, echo in [
                ("tool-less-omit", False, False),
                ("tools-omit", True, False),
                ("tools-echo", True, True),
            ]:
                history = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer arithmetic yourself; do not call tools. Think briefly.",
                    },
                    {
                        "role": "user",
                        "content": name
                        + "\n"
                        + (
                            "The town archive records weather and routine shipping schedules. "
                            * 300
                        )
                        + "\nWhat is 17 multiplied by 3? Answer with just the number.",
                    },
                ]
                for turn in range(3):
                    if turn == 2:
                        stop(proc)
                        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=log)
                        wait_ready(proc, base)
                    body = {
                        "model": "qwen",
                        "messages": history,
                        "temperature": 0,
                        "max_tokens": 512,
                        "reasoning_effort": "low",
                        "stream": False,
                    }
                    if has_tools:
                        body["tools"] = tools
                    req = urllib.request.Request(
                        base + "/v1/chat/completions",
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    t0 = time.monotonic()
                    with urllib.request.urlopen(req, timeout=180) as response:
                        result = json.load(response)
                    message = result["choices"][0]["message"]
                    finish = result["choices"][0]["finish_reason"]
                    row = {
                        "case": name,
                        "turn": turn + 1,
                        "seconds": time.monotonic() - t0,
                        "usage": result.get("usage"),
                        "finish": finish,
                        "reasoning_chars": len(message.get("reasoning_content") or ""),
                        "content": message.get("content"),
                        "tool_calls": message.get("tool_calls"),
                    }
                    print(json.dumps(row), flush=True)
                    results.append(row)
                    (out / (name + f"-{turn + 1}.json")).write_text(
                        json.dumps(result, indent=2)
                    )
                    assert finish == "stop" and not message.get("tool_calls"), row
                    if turn == 0:
                        assert row["reasoning_chars"] > 0, row
                    if turn > 0:
                        cached = result["usage"]["prompt_tokens_details"][
                            "cached_tokens"
                        ]
                        assert cached > 3000, row
                        assert result["usage"]["prompt_tokens"] - cached < 128, row
                    assistant = {
                        "role": "assistant",
                        "content": message.get("content") or "",
                    }
                    if echo:
                        assistant["reasoning_content"] = (
                            message.get("reasoning_content") or ""
                        )
                    history += [
                        assistant,
                        {
                            "role": "user",
                            "content": "Now add 2 to your previous answer. Answer with just the number.",
                        },
                    ]
            (out / "results.json").write_text(json.dumps(results, indent=2))
        finally:
            stop(proc)
    logtext = logpath.read_text()
    assert "KV payload staging failed" not in logtext, logpath
    assert "session has no valid checkpoint to stage" not in logtext, logpath
    assert logtext.count("reason=continued") >= 9, logpath
    assert "kv cache evicted reason=disk-cache-full" in logtext, logpath
    assert "reason=evict" in logtext, logpath
    print("Artifacts:", out, "cache:", cache, flush=True)


if __name__ == "__main__":
    main()
