#!/usr/bin/env python3
"""Run Pi image-reading/editing tasks against an otherwise idle vision server."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8080")
parser.add_argument("--pi", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--api", action="append", choices=[
    "openai-completions", "openai-responses", "anthropic-messages"])
args = parser.parse_args()
base = args.url.rstrip("/")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parent.parent
for api in args.api or ["openai-completions", "openai-responses", "anthropic-messages"]:
    project = out / api
    project.mkdir(exist_ok=True)
    settings = project / "pi-settings"
    settings.mkdir(exist_ok=True)
    (settings / "settings.json").write_text(json.dumps({"compaction": {"enabled": False}}))
    provider = {"baseUrl": base + ("" if api == "anthropic-messages" else "/v1"),
                "api": api, "apiKey": "local-test",
                "models": [{"id": "deepseek-v4-flash", "name": "Vision QA", "reasoning": True,
                            "input": ["text", "image"], "contextWindow": 16384, "maxTokens": 1536,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}
    if api == "openai-completions":
        provider["compat"] = {"supportsDeveloperRole": False}
    (settings / "models.json").write_text(json.dumps({"providers": {"ds4": provider}}))
    for name in ["text.png", "spatial.png"]:
        shutil.copyfile(root / "tests/vision-fixtures/glm53" / name, project / name)
    (project / "ticket.py").write_text("def ticket():\n    return {}\n\ndef shapes():\n    return []\n")
    (project / "CONTEXT.txt").write_text("\n".join(
        "Archive record %d: the build passed with no warnings." % i for i in range(200)))
    cmd = [str(args.pi.resolve()),
           "--print", "--mode", "json", "--provider", "ds4", "--model", "deepseek-v4-flash",
           "--thinking", "off", "--no-session", "--no-extensions", "--no-skills",
           "--no-context-files", "--no-prompt-templates",
           "Read CONTEXT.txt for background, then read ticket.py. Use the read tool to inspect "
           "text.png. Implement ticket() returning a dictionary with keys train (integer), "
           "gate (string), access (string), using the actual image. After editing ticket(), "
           "use read to inspect spatial.png. Implement shapes() returning a list of lowercase "
           "color/shape strings like 'purple oval', using the actual shapes. "
           "Read each image separately, in that order. Run Python to check the implementation. "
           "Do not create other files. Finish with a brief result."]
    (project / "command.json").write_text(json.dumps(cmd))
    with (project / "pi.jsonl").open("w") as log:
        subprocess.run(cmd, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PI_CODING_AGENT_DIR=str(settings),
                                PATH=os.environ.get("PATH", "")),
                       timeout=600, check=True)
    oracle = "import ticket; assert ticket.ticket() == {'train': 482, 'gate': 'C7', 'access': 'MINT-731'}; assert set(ticket.shapes()) == {'blue circle', 'red triangle', 'green square'}"
    subprocess.run([sys.executable, "-c", oracle], cwd=project, check=True)
    events = []
    saw_image = False
    for line in (project / "pi.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "tool_execution_end":
            saw_image |= any(c.get("type") == "image"
                             for c in event.get("result", {}).get("content", []))
        if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant":
            assert event["message"].get("stopReason") != "error", event["message"].get("errorMessage")
            if saw_image:
                assert event["message"].get("usage", {}).get("cacheRead", 0) > 0, \
                    "image tool result lost the matching context prefix"
        if event.get("type") == "tool_execution_start":
            events.append(event)
    read_images = [e for e in events if e.get("toolName") == "read" and
                   any(name in json.dumps(e.get("args", {})) for name in ["text.png", "spatial.png"])]
    assert len(read_images) >= 2, "Pi did not actually read both images"
    print("PASS Pi " + api + ": both image reads, code edits, independent output oracle", flush=True)
