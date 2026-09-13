#!/usr/bin/env python3
"""Real-model image-cache regression. Run against an otherwise idle vision server."""

import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--archive-lines", type=int, default=320)
    parser.add_argument("--append-only", action="store_true",
                        help="only run the long-prefix image append sequence")
    parser.add_argument("--thinking-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = Path(__file__).resolve().parent / "vision-fixtures/glm53"
    results = []
    results_lock = Lock()

    def image(name):
        data = base64.b64encode((fixtures / name).read_bytes()).decode()
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}

    def ask(label, history, expected_cache=None, tools=None, thinking=False):
        previous_frontier = results[-1]["usage"]["total_tokens"] if expected_cache and results else 0
        body = {"model": args.model, "messages": history, "temperature": 0,
                "reasoning_effort": "low" if thinking else "none",
                "max_tokens": 512 if tools or thinking else 48, "stream": False}
        if tools:
            body["tools"] = tools
        (args.output / (label + ".request.json")).write_text(json.dumps(body))
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        (args.output / (label + ".response.json")).write_text(json.dumps(reply, indent=2))
        usage = reply["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        result = {"label": label, "seconds": time.monotonic() - started, "usage": usage,
                  "message": reply["choices"][0]["message"]}
        with results_lock:
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
        if expected_cache is not None:
            if expected_cache:
                assert cached > 0 and cached >= previous_frontier, \
                    label + ": lost part of the live image/text prefix"
            else:
                assert cached == 0, label + ": reused an incompatible image prefix"
        return result["message"]

    def thinking_replay():
        history = [{"role": "user", "content": [
            {"type": "text", "text": "What is the access code in this image? Reply with just the code."},
            image("text.png")]}]
        answer = ask("thinking-image", history, thinking=True)
        assert "MINT-731" in answer.get("content", "")
        # These clients replay the visible answer, not its hidden reasoning.
        history.append({"role": "assistant", "content": answer["content"]})
        history.append({"role": "user", "content": "Give just the train number from that image."})
        answer = ask("thinking-replay", history, True, thinking=True)
        assert "482" in answer.get("content", "")
        history.append({"role": "assistant", "content": answer["content"]})
        history.append({"role": "user", "content": [
            {"type": "text", "text": "How many shapes in this image? Reply with just a number."},
            image("spatial.png")]})
        answer = ask("thinking-new-image", history, True, thinking=True)
        assert "3" in answer.get("content", "") or "three" in answer.get("content", "").lower()
        history.append({"role": "assistant", "content": answer["content"]})
        history.append({"role": "user", "content": "Give just the access code from the first image."})
        answer = ask("thinking-two-images", history, True, thinking=True)
        assert "MINT-731" in answer.get("content", "")
        history.append({"role": "assistant", "content": answer["content"]})
        history.append({"role": "user", "content": "Give just the train number from the first image."})
        answer = ask("thinking-replay-again", history, True, thinking=True)
        assert "482" in answer.get("content", "")
        changed = copy.deepcopy(history)
        changed[0]["content"][1] = image("spatial.png")
        ask("thinking-changed-image", changed, False, thinking=True)

    if args.thinking_only:
        thinking_replay()
        print("PASS: image-bearing visible replay without hidden reasoning")
        return

    archive = "\n".join("Archive record %d: the build passed with no warnings." % i
                        for i in range(args.archive_lines))
    history = [{"role": "system", "content": "Answer briefly and accurately."},
               {"role": "user", "content": archive + "\nReply with exactly READY."}]
    history.append(ask("text", history))
    history.append({"role": "user", "content": "Reply with exactly TEXT_OK."})
    history.append(ask("text-append", history, True))
    history.append({"role": "user", "content": [
        {"type": "text", "text": "Read the text in this picture. Reply briefly."},
        image("text.png")]})
    history.append(ask("first-image", history, True))
    history.append({"role": "user", "content": "Repeat only the text in the image."})
    history.append(ask("same-image", history, True))
    history.append({"role": "user", "content": [
        {"type": "text", "text": "Describe the shapes in this second image briefly."},
        image("spatial.png")]})
    history.append(ask("second-image", history, True))
    history.append({"role": "user", "content": "Reply with exactly FINISHED."})
    history.append(ask("two-images", history, True))
    if args.append_only:
        print("PASS: long-prefix image appends")
        return

    # The text/token placeholders can stay identical while pixels change.
    changed = copy.deepcopy(history[:-1])
    changed[5]["content"][1] = image("spatial.png")
    ask("changed-old-image", changed, False)
    removed = copy.deepcopy(changed)
    removed[5]["content"] = [{"type": "text", "text": "No image was supplied."}]
    ask("removed-old-image", removed, False)
    reordered = copy.deepcopy(history[:-1])
    reordered[5]["content"][1], reordered[9]["content"][1] = (
        reordered[9]["content"][1], reordered[5]["content"][1])
    ask("reordered-images", reordered, False)

    # A real tool-call exchange, with an image added in the following turn.
    tools = [{"type": "function", "function": {
        "name": "record", "description": "Record the exact text visible in the image.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"], "additionalProperties": False}}}]
    history = [{"role": "system", "content": "Use record when asked to record image text."},
               {"role": "user", "content": [
                   {"type": "text", "text": "Record the text in this image using the record tool."},
                   image("text.png")]}]
    answer = ask("tool-image", history, tools=tools)
    assert answer.get("tool_calls"), "model did not produce the required tool call"
    history.append(answer)
    for call in answer["tool_calls"]:
        assert call["function"]["name"] == "record"
        arguments = json.loads(call["function"]["arguments"])
        assert isinstance(arguments.get("text"), str) and arguments["text"]
        history.append({"role": "tool", "tool_call_id": call["id"],
                        "content": "Recorded successfully: " + arguments["text"]})
    history.append(ask("tool-result", history, True, tools=tools))
    history.append({"role": "user", "content": [
        {"type": "text", "text": "Now describe this second image. Do not call tools."},
        image("spatial.png")]})
    history.append(ask("tool-new-image", history, True, tools=tools))
    history.append({"role": "user", "content": "Reply with exactly DONE."})
    ask("tool-final", history, True, tools=tools)
    probes = [
        [{"role": "user", "content": [
            {"type": "text", "text": "Read the text in the image. Be brief."}, image(name)]}]
        for name in ("text.png", "spatial.png")
    ]
    controls = [ask("control-" + str(i), probe) for i, probe in enumerate(probes)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ask, "concurrent-" + str(i), probe)
                   for i, probe in enumerate(probes)]
        for i, future in enumerate(futures):
            assert future.result()["content"] == controls[i]["content"], "cross-session image mismatch"
    thinking_replay()
    print("PASS: image appends, invalidation, tool calls, concurrent sessions, and thinking replay")


if __name__ == "__main__":
    main()
