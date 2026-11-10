#!/usr/bin/env python3
"""Real-model check that image conversations survive a server restart.

Starts ./ds4-server on its own port with an empty disk KV cache and restarts
it between steps, so every reuse asserted here comes from disk:

    python3 tests/test_server_vision_disk.py MODEL.gguf \
        --vision ENCODER.gguf --output /tmp/vision-disk-qa

The fixtures are the GLM 5.3 ones. The run is repeated with reasoning on,
where the client replays answers without the hidden reasoning, and ends with a
long prompt whose checkpoints are written while it is still being read.
"""

import argparse
import base64
import copy
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

FIXTURES = Path(__file__).resolve().parent / "vision-fixtures/glm53"


def image(name):
    data = base64.b64encode((FIXTURES / name).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def archive(lines):
    return ["Archive record %d: the build passed with no warnings." % i
            for i in range(lines)]


class Server:
    def __init__(self, args, kv, log, options=()):
        self.base = "http://127.0.0.1:%d" % args.port
        self.command = [args.server, "-m", args.model, "--vision", args.vision,
                        "--ctx", str(args.ctx), "--host", "127.0.0.1",
                        "--port", str(args.port), "--kv-disk-dir", str(kv),
                        "--kv-disk-space-mb", "8192", *options]
        self.log = log
        self.proc = None
        self.model = None

    def __enter__(self):
        self.proc = subprocess.Popen(self.command, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            assert self.proc.poll() is None, "server exited during startup"
            try:
                with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as reply:
                    self.model = json.load(reply)["data"][0]["id"]
                return self
            except (urllib.error.URLError, ConnectionError):
                assert time.monotonic() < deadline, "server startup timed out"
                time.sleep(0.5)

    def __exit__(self, kind, value, traceback):
        if self.proc.poll() is None:
            # A clean stop saves the live session, as a deployed server does.
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if kind is None:
            assert self.proc.returncode == 0, \
                "server exit status %d" % self.proc.returncode

    def ask(self, results, label, messages, thinking, max_tokens=None):
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "reasoning_effort": "low" if thinking else "none",
                "max_tokens": max_tokens or (768 if thinking else 48),
                "stream": False}
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        usage = reply["usage"]
        result = {"label": label, "prompt": usage["prompt_tokens"],
                  "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                  "content": reply["choices"][0]["message"].get("content") or ""}
        results.append(result)
        print(json.dumps(result), flush=True)
        return result


def empty_cache(out):
    kv = out / "kv"
    kv.mkdir(parents=True, exist_ok=True)
    assert not any(kv.iterdir()), "%s is not empty" % kv
    return kv


def run(args, name, thinking):
    out = args.output / name
    kv = empty_cache(out)
    results = []

    def reply(result):
        return {"role": "assistant", "content": result["content"]}

    # The archive makes the text before the first image worth a checkpoint.
    question = ("What is the access code in this image? "
                "Reply with just the code, or NONE if it shows none.")
    history = [{"role": "system", "content": "\n".join(archive(args.archive_lines)) +
                "\nAnswer briefly and accurately."},
               {"role": "user", "content": [{"type": "text", "text": question},
                                            image("text.png")]}]

    with (out / "server.log").open("a") as log:
        with Server(args, kv, log) as server:
            first = server.ask(results, "first-image", history, thinking)
            assert first["cached"] == 0, "the disk cache was not empty"
            assert "MINT-731" in first["content"]
            history += [reply(first), {"role": "user", "content":
                        "Give just the train number from that image."}]
            saved = server.ask(results, "same-image", history, thinking)
            assert "482" in saved["content"]

        # A restarted server has no live session: what it reuses is on disk.
        # The saved prefix holds the first image; a second one is appended.
        with Server(args, kv, log) as server:
            history += [reply(saved), {"role": "user", "content": [
                {"type": "text", "text": "How many shapes are in this second image? "
                                         "Reply with just a number."},
                image("spatial.png")]}]
            answer = server.ask(results, "restart-second-image", history, thinking)
            assert answer["cached"] >= saved["prompt"], \
                "the saved image conversation did not come back from disk"
            assert "3" in answer["content"] or "three" in answer["content"].lower()
            history += [reply(answer), {"role": "user", "content":
                        "Give just the access code from the first image."}]
            saved = server.ask(results, "two-images", history, thinking)
            assert "MINT-731" in saved["content"]

        with Server(args, kv, log) as server:
            history += [reply(saved), {"role": "user", "content":
                        "Give just the departure time from the first image."}]
            answer = server.ask(results, "restart-two-images", history, thinking)
            assert answer["cached"] >= saved["prompt"], \
                "the saved two-image conversation did not come back from disk"
            assert "19:45" in answer["content"]

        # Other pixels at the place of the first image: only the text before
        # it may come from disk. The same request without the picture measures
        # how many tokens that is.
        with Server(args, kv, log) as server:
            changed = copy.deepcopy(history[:2])
            changed[1]["content"][1] = image("spatial.png")
            answer = server.ask(results, "changed-image", changed, thinking)
            assert "MINT-731" not in answer["content"], \
                "answered from the image that was replaced"
            text_only = copy.deepcopy(history[:2])
            del text_only[1]["content"][1]
            limit = server.ask(results, "text-before-image", text_only, thinking,
                               max_tokens=1)["prompt"]
            assert 0 < answer["cached"] < limit, \
                "a changed image must reuse the text before it and nothing else"

    (out / "results.json").write_text(json.dumps(results, indent=2))


def run_prefill(args):
    """The image comes first and a long text follows it, so the checkpoints
    the server writes while it reads that text all hold the image. Without
    the cold checkpoint, which would stand in for them, a second conversation
    that shares the image and most of the text can only continue from one."""
    out = args.output / "prefill"
    kv = empty_cache(out)
    results = []
    lines = archive(args.prefill_lines)
    shared = len(lines) * 5 // 6

    def conversation(tail, question):
        text = "\n".join(lines[:shared] + tail) + "\n" + question
        return [{"role": "user", "content": [image("text.png"),
                                            {"type": "text", "text": text}]}]

    options = ["--kv-cache-cold-max-tokens", "0",
               "--kv-cache-continued-interval-tokens", "2048"]
    with (out / "server.log").open("a") as log:
        with Server(args, kv, log, options) as server:
            first = server.ask(results, "long-prompt", conversation(
                lines[shared:], "What is the access code in the image? "
                                "Reply with just the code."), False)
            assert first["cached"] == 0, "the disk cache was not empty"
            assert "MINT-731" in first["content"]
        with Server(args, kv, log, options) as server:
            answer = server.ask(results, "restart-shared-part", conversation(
                ["Later record %d: nothing to report." % i
                 for i in range(shared, len(lines))],
                "What is the train number in the image? "
                "Reply with just the number."), False)
            assert answer["cached"] > 0, \
                "no checkpoint written during the prefill came back from disk"
            assert "482" in answer["content"]
    (out / "results.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="language GGUF of a vision model")
    parser.add_argument("--vision", required=True, help="its vision encoder GGUF")
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for logs, results and the KV cache")
    parser.add_argument("--server", default="./ds4-server")
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--ctx", default=16384, type=int)
    parser.add_argument("--archive-lines", default=120, type=int)
    parser.add_argument("--prefill-lines", default=900, type=int)
    args = parser.parse_args()
    if not args.port:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            args.port = listener.getsockname()[1]
    run(args, "plain", False)
    run(args, "thinking", True)
    run_prefill(args)
    print("PASS: image conversations restored from disk after restarts, "
          "a second image appended to a restored one, a changed image "
          "limited to the text before it, and checkpoints written during "
          "a long prefill")


if __name__ == "__main__":
    main()
