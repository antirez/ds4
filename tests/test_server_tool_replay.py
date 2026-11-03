"""Long Qwen tool-history persistence/reuse regression (GPU server, MTP model).

Build the GPU server and run with no other DS4 model process active:
  python3 tests/test_server_tool_replay.py --model gguf/Qwen3.8-Flash-Next-Q4.gguf

Owns its server subprocesses and uses isolated log/KV directories; no vision
model or external launcher scripts are required. The negative control clears TOOL_MAP only in copied test checkpoints, never in
operator caches. A discarded streaming answer requires Qwen recurrent-state
recovery from disk; retain the saved long prefix and measure any checkpoint gap,
not arbitrary RAM rewind.
"""

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

helpers = importlib.import_module("test_server_turn_boundary")
decisions, post, stop, wait_ready = helpers.decisions, helpers.post, helpers.stop, helpers.wait_ready


class Server:
    def __init__(self, env):
        self.env = env
        self.proc = None
        self.log = None

    def manage(self, start):
        if not start:
            proc, self.proc = self.proc, None
            try:
                if proc is not None:
                    was_running = proc.poll() is None
                    stop(proc)
                    assert not was_running or proc.returncode == 0, "server did not shut down gracefully"
            finally:
                if self.log is not None:
                    self.log.close()
                    self.log = None
            return
        assert self.proc is None
        env = self.env
        root = pathlib.Path(env["DS4_DIR"])
        logs = pathlib.Path(env["DS4_LOG_DIR"])
        logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs / "ds4-q38fn-server.log").open("a")
        cmd = [str(root / "ds4-server"), "-m", env["DS4_MODEL"],
               "--host", "127.0.0.1", "--port", env["DS4_PORT"],
               "--ctx", "65536", "--tokens", "8192", "--warm-weights",
               "--mtp", "--mtp-exact-sampling", "--prefill-chunk", "2048",
               "--kv-disk-dir", env["DS4_KV_DIR"], "--kv-disk-space-mb", "8192",
               "--kv-cache-cold-max-tokens", "65536",
               "--kv-cache-continued-interval-tokens", "8192",
               "--kv-cache-min-tokens", "128", "--kv-cache-boundary-trim-tokens", "0",
               "--kv-cache-boundary-align-tokens", "0", "--trace", env["DS4_TRACE"]]
        self.proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=self.log, stderr=self.log)
        wait_ready(self.proc, f"http://127.0.0.1:{env['DS4_PORT']}")


def test_trace_decisions():
    with tempfile.TemporaryDirectory() as directory:
        trace = pathlib.Path(directory) / "trace.txt"
        assert decisions(trace) == []
        trace.write_text("prefix\n--- cache decision ---\ncache_source: none\ncached_tokens: 0\n\n"
                         "unrelated: ignore\n--- cache decision ---\r\ncache_source: disk-text\r\n"
                         "cached_tokens: 34567\r\n\r\n")
        assert decisions(trace) == [
            {"cache_source": "none", "cached_tokens": "0"},
            {"cache_source": "disk-text", "cached_tokens": "34567"},
        ]
    # Full KV reuse can coexist with a shorter retokenized BPE prefix.
    block = {"live_tokens_before": "499", "live_prompt_common": "391", "disk_cached_tokens": "0"}
    row = {"usage": {"prompt_tokens": 525, "prompt_tokens_details": {"cached_tokens": 499}}}
    helpers.assert_turn_boundary("full reuse", row, block, 392)
    row["usage"]["prompt_tokens_details"]["cached_tokens"] = 498
    try:
        helpers.assert_turn_boundary("partial reuse", row, block, 392)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial live frontier accepted")


def retain(history, result):
    message = result["choices"][0]["message"]
    assistant = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        calls = json.loads(json.dumps(message["tool_calls"]))
        # Reverse the observed NESTED object order: the Qwen renderer fixes
        # top-level order from the schema, but preserves nested JSON order.
        # This guarantees canonical divergence while retaining identical values.
        for call in calls:
            arguments = json.loads(call["function"]["arguments"])
            assert isinstance(arguments["detail"], dict) and len(arguments["detail"]) >= 2
            arguments["detail"] = dict(reversed(list(arguments["detail"].items())))
            call["function"]["arguments"] = json.dumps(arguments)
        assistant["tool_calls"] = calls
    history.append(assistant)
    return assistant.get("tool_calls", [])


def main():
    test_trace_decisions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out = (args.output or pathlib.Path(tempfile.mkdtemp(prefix="ds4-tool-replay-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "kv"
    cache.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "DS4_DIR": str(root),
        "DS4_MODEL": str(pathlib.Path(args.model).resolve()),
        "DS4_LOG_DIR": str(out / "logs"),
        "DS4_KV_DIR": str(cache),
        "DS4_PORT": str(port),
        "DS4_TRACE": str(out / "before.trace"),
    }
    server = Server(env)
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up the named archive record. Call once when a key is requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "detail": {"type": "object", "properties": {
                        "note": {"type": "string"}, "format": {"type": "string"}},
                        "required": ["note", "format"]},
                },
                "required": ["query", "detail"],
            },
        },
    }]
    history = [
        {"role": "system", "content": "Use lookup for requested archive keys; never invent records. "
         "Set query to the requested key and detail to {\"note\":\"exact record\",\"format\":\"plain\"}. Think briefly. "
         "After a tool result, reply ACK only, without calling tools. "
         "Otherwise follow the user's instruction."},
        {"role": "user", "content": "Look up archive key tide-000 with lookup now."},
    ]
    rows = []

    def body(stream=False):
        return {"model": "qwen", "messages": history, "tools": tools,
                "temperature": 0, "max_tokens": 8192 if stream else 512,
                "reasoning_effort": "low", "stream": stream}

    def request(label, warm=False, small=True, missing=False):
        request_body = body()
        (out / f"{label}-request.json").write_text(json.dumps(request_body, indent=2))
        t0 = time.monotonic()
        result = post(base, request_body)
        (out / f"{label}-response.json").write_text(json.dumps(result, indent=2))
        block = decisions(pathlib.Path(env["DS4_TRACE"]))[-1]
        usage = result["usage"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        replay = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", block["tool_replay"])}
        row = {"case": label, "seconds": round(time.monotonic() - t0, 3),
               "usage": usage, "prefill_tokens": usage["prompt_tokens"] - cached,
               "finish": result["choices"][0]["finish_reason"], "decision": block}
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: row[k] for k in ("case", "seconds", "prefill_tokens", "finish")}
                         | {"cached_tokens": cached, "cache_source": block["cache_source"],
                            "tool_replay": replay}), flush=True)
        if not missing:
            assert replay["canonical"] == replay["missing_ids"] == 0, row
        if warm:
            assert cached == int(block["live_tokens_before"]), row
            assert int(block["disk_cached_tokens"]) == 0, row
        if small:
            assert row["prefill_tokens"] < 256, row
        return result, row, replay

    print("Artifacts:", out, flush=True)
    try:
        server.manage(True)
        for turn in range(12):
            called, row, _ = request(f"call-{turn:02}", warm=turn > 0, small=turn > 0)
            assert row["finish"] == "tool_calls", row
            calls = retain(history, called)
            assert len(calls) == 1, calls
            arguments = json.loads(calls[0]["function"]["arguments"])
            assert arguments == {"query": f"tide-{turn:03}",
                                 "detail": {"note": "exact record", "format": "plain"}}, arguments
            record = f"tide-{turn:03}: archived value {turn}."
            if turn == 0:
                # Put the first sampled tool BEFORE the long history. Without
                # its id, canonical key order loses essentially all this prefix.
                record += "\n" + "\n".join(
                    f"Archive ledger row {i:05}: harbor logs describe routine weather, dates, "
                    "arrivals, cargo quantities, and record identifiers." for i in range(1600)
                )
            history.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": record})
            if turn == 11:
                break  # The first post-restart request appends this tool result.
            answered, ack, _ = request(f"result-{turn:02}", warm=True, small=turn > 0)
            assert ack["finish"] == "stop", ack
            if turn == 0:
                assert ack["usage"]["prompt_tokens"] >= 32768, ack
            assert not retain(history, answered), answered
            history.append({"role": "user", "content": f"Look up archive key tide-{turn+1:03} with lookup now."})

        server.manage(False)
        saved, adapted = [], []
        control = out / "kv-no-tool-map"
        legacy = out / "kv-legacy-visible-key"
        control.mkdir()
        legacy.mkdir()
        for path in cache.glob("*.kv"):
            with path.open("rb") as file:
                header = file.read(48)
            assert len(header) == 48, path
            tokens = int.from_bytes(header[8:12], "little")
            saved.append({"file": path.name, "tokens": tokens, "flags": header[6], "bytes": path.stat().st_size})
            copy = control / path.name
            shutil.copyfile(path, copy)
            with copy.open("r+b") as file:
                file.seek(6)
                file.write(bytes([header[6] & ~1]))  # TOOL_MAP only; preserve tokens/payload/other flags.
            with path.open("rb") as file:
                file.seek(48)
                length = int.from_bytes(file.read(4), "little")
                key = file.read(length)
                eos = b"<|im_end|>"
                if header[6] & 4 and key.endswith(eos) and key[:-len(eos)].rstrip().endswith(b"</tool_call>"):
                    # Emulate the old visible key, not a rewritten token count.
                    # The exact KV payload and all trailers remain byte-identical.
                    old_key = key[:-len(eos)]
                    name = hashlib.sha1(old_key).hexdigest() + ".kv"
                    with (legacy / name).open("wb") as target:
                        target.write(header)
                        target.write(len(old_key).to_bytes(4, "little"))
                        target.write(old_key)
                        shutil.copyfileobj(file, target)
                    adapted.append({"original": path.name, "legacy": name, "tokens": tokens})
                else:
                    shutil.copyfile(path, legacy / path.name)
        (out / "checkpoints.json").write_text(json.dumps(saved, indent=2))
        (out / "legacy-fixtures.json").write_text(json.dumps(adapted, indent=2))
        frontier = max(item["tokens"] for item in saved)
        log = (out / "logs" / "ds4-q38fn-server.log").read_text(errors="replace")
        # Accept both the current "qwen tool-turn ..." spelling and the generic
        # "tool-turn ..." spelling from #1058's renamed helper.
        remembered = re.findall(r"(?:qwen )?tool-turn visible checkpoint remembered [^\n]* live=(\d+)", log)
        assert remembered and frontier == int(remembered[-1]), (saved, remembered)
        assert any(item["tokens"] == frontier and item["flags"] & 1 for item in saved), saved
        assert any(item["tokens"] == frontier for item in adapted), adapted

        env["DS4_KV_DIR"] = str(legacy)
        env["DS4_TRACE"] = str(out / "legacy.trace")
        server.manage(True)
        _, legacy_row, replay = request("restart-with-legacy-visible-key")
        assert replay["disk"] >= 12, legacy_row
        assert legacy_row["usage"]["prompt_tokens_details"]["cached_tokens"] == frontier, legacy_row
        server.manage(False)

        env["DS4_KV_DIR"] = str(control)
        env["DS4_TRACE"] = str(out / "negative.trace")
        server.manage(True)
        _, negative, replay = request("restart-without-map", small=False, missing=True)
        assert replay["disk"] == 0 and replay["missing_ids"] >= 12, negative
        assert negative["prefill_tokens"] >= 32768, negative
        server.manage(False)

        env["DS4_KV_DIR"] = str(cache)
        env["DS4_TRACE"] = str(out / "after.trace")
        server.manage(True)
        answered, restart, replay = request("restart-with-map")
        assert replay["disk"] >= 12, restart
        assert int(restart["decision"]["disk_cached_tokens"]) == frontier, (frontier, restart)
        assert restart["usage"]["prompt_tokens_details"]["cached_tokens"] == frontier, restart
        # Same exact payload and new-turn suffix: legacy key must not add an EOS.
        assert legacy_row["usage"]["prompt_tokens"] == restart["usage"]["prompt_tokens"], (legacy_row, restart)
        assert not retain(history, answered), answered
        history.append({"role": "user", "content": "Report the last archived value only. Do not call tools."})
        answered, _, _ = request("live-after-restart", warm=True)
        assert not retain(history, answered), answered

        history.append({"role": "user", "content": "Write a detailed numbered list of 3000 harbor observations. "
                        "Do not call tools. Continue listing until the output limit."})
        stream_body = body(stream=True)
        (out / "interrupted-request.json").write_text(json.dumps(stream_body, indent=2))
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(stream_body).encode(),
                                     headers={"Content-Type": "application/json"})
        chunks = 0
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if delta.get("content"):
                    chunks += 1
                if chunks == 16:
                    break  # Disconnect; client retains no generated assistant text.
        assert chunks == 16, "model finished before the test could interrupt it"
        history.append({"role": "user", "content": "Interrupted. Instead report the last archived value only. Do not call tools."})
        # Existing policy may leave a replay gap. Measure it; require the saved
        # long prefix to survive, not a new per-turn checkpoint or RAM rollback.
        answered, retry, _ = request("after-interruption", small=False)
        assert retry["usage"]["prompt_tokens_details"]["cached_tokens"] >= frontier, retry
        assert retry["finish"] == "stop", retry
        assert "cancelled during generation" in pathlib.Path(env["DS4_TRACE"]).read_text(), retry
        assert not answered["choices"][0]["message"].get("tool_calls"), answered
    finally:
        server.manage(False)
    log = (out / "logs" / "ds4-q38fn-server.log").read_text(errors="replace")
    assert "turn end token not kept in live KV" not in log
    assert "KV payload staging failed" not in log
    assert "session has no valid checkpoint to stage" not in log
    print("PASS: sampled ids survive restart; long-prefix reuse and interrupted recovery verified.", flush=True)
    print("Artifacts:", out, flush=True)


if __name__ == "__main__":
    main()
