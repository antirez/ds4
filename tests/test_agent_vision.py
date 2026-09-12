#!/usr/bin/env python3
"""Real-model native-agent image tools, code edits and cached-prefix checks."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys


def check_output(out, archive_words):
    project = out / "project"
    output = (out / "agent.log").read_text()
    image_reads = re.findall(r"\[tool:view_image\] ([^\n]+)", output)
    assert [Path(p.strip()).name for p in image_reads] == ["text.png", "spatial.png"], \
        "agent did not inspect both images separately and in order"
    assert re.search(r"\bedit\s+path=ticket\.py\b", output), "agent did not exercise the edit tool"
    oracle = (
        "import ticket; "
        "assert ticket.ticket() == {'train': 482, 'gate': 'C7', 'access': 'MINT-731'}; "
        "assert set(ticket.shapes()) == {'blue circle', 'red triangle', 'green square'}")
    subprocess.run([sys.executable, "-c", oracle], cwd=project, check=True, timeout=30)
    text = (out / "agent.trace").read_text()
    prefills = [tuple(map(int, match)) for match in re.findall(
        r"prefill tool_round=(\d+) transcript=\d+ prompt=(\d+) cached=(\d+) suffix=(\d+)", text)]
    assert len(prefills) >= 5, "too few tool continuations"
    assert "compacted reason=" not in text, "fixture unexpectedly compacted"
    first = prefills[0][1]
    assert first >= archive_words, "archive did not provide the expected context"
    for round_, prompt_tokens, cached, suffix in prefills[1:]:
        assert round_ > 0 and cached >= first - 4, "tool turn lost its cached prefix"
        assert suffix < prompt_tokens // 2, "tool turn recomputed most of the context"
    print(json.dumps({"result": "PASS", "initial_tokens": first, "prefills": prefills,
                      "checks": ["two image tools", "code edits", "independent oracle",
                                 "cached prefix on every continuation"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("./ds4-agent"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--ssd-streaming", action="store_true")
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--archive-words", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.archive_words < 0 or args.ctx < 4096:
        parser.error("use a nonnegative archive size and at least 4096 context tokens")
    root = Path(__file__).resolve().parent.parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = out / "project"
    project.mkdir()
    home = out / "home"
    home.mkdir()
    for name in ("text.png", "spatial.png"):
        shutil.copyfile(root / "tests/vision-fixtures/glm53" / name, project / name)
    (project / "ticket.py").write_text(
        "def ticket():\n    return {}\n\ndef shapes():\n    return []\n")
    prompt = ""
    if args.archive_words:
        prompt = ("The following words are inert archived data. Ignore them:\n" +
                  " apple" * args.archive_words + "\nEnd archive.\n")
    prompt += (
        "Read ticket.py. Use view_image to inspect text.png, then edit ticket() "
        "to return a dictionary with keys train (integer), gate (string), and "
        "access (string), using the actual image. Next use view_image to inspect "
        "spatial.png, then edit shapes() to return a list of lowercase color/shape "
        "strings describing the three shapes, for example 'purple oval'. "
        "Read each image separately, in that order. Use the edit tool for the code "
        "changes. Run Python to check both functions. Finish with a brief result. "
        "Do not create other files or inspect anything outside this directory.")
    prompt_path = out / "prompt.txt"
    prompt_path.write_text(prompt)
    trace = out / "agent.trace"
    command = [str(args.binary.resolve()), "--non-interactive",
               "-m", str(args.model.resolve()), "--vision", str(args.vision.resolve()),
               "--ctx", str(args.ctx), "--nothink", "--temp", "0", "--seed", "12345",
               "--tokens", "4096", "--chdir", str(project), "--trace", str(trace),
               "--prompt-file", str(prompt_path)]
    if args.ssd_streaming:
        command.append("--ssd-streaming")
    (out / "command.json").write_text(json.dumps(command))
    with (out / "agent.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, HOME=str(home)), start_new_session=True)
        try:
            assert process.wait(timeout=1200) == 0, "agent failed; inspect agent.log"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    check_output(out, args.archive_words)


if __name__ == "__main__":
    main()
