#!/usr/bin/env python3
"""Real-model PTY regression for near-full agent context. Requires pyte."""

import argparse
import codecs
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import pyte

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, default=Path("./ds4-agent"))
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--vision", type=Path)
parser.add_argument("--ssd-streaming", action="store_true")
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--think", action="store_true")
parser.add_argument("--tokens", type=int, default=1800)
parser.add_argument("--mtp", action="store_true")
parser.add_argument("--dspark", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--interrupt-only", action="store_true",
                    help="check generation interruption and same-session tool recovery")
args = parser.parse_args()
root = args.binary.resolve().parent
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
home = out / "home"
home.mkdir(exist_ok=True)
project = out / "project"
project.mkdir(exist_ok=True)
trace = out / "agent.trace"
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
cmd = [str(args.binary.resolve()), "-m", str(args.model.resolve()),
       "--ctx", str(args.ctx), "--think" if args.think else "--nothink",
       "--temp", "0", "--tokens", str(args.tokens),
       "--seed", "12345", "--chdir", str(project), "--trace", str(trace)]
if args.vision:
    cmd += ["--vision", str(args.vision.resolve())]
if args.ssd_streaming:
    cmd += ["--ssd-streaming"]
if args.mtp:
    cmd += ["--mtp"]
if args.dspark:
    cmd += ["--dspark", "--mtp-model", str(args.dspark.resolve())]
(out / "command.json").write_text(json.dumps(cmd))
proc = subprocess.Popen(cmd, cwd=root, stdin=slave, stdout=slave, stderr=slave,
                        env=dict(os.environ, HOME=str(home), TERM="xterm-256color"), start_new_session=True)
os.close(slave)
screen = pyte.HistoryScreen(120, 40, history=10000)
stream = pyte.Stream(screen)
decoder = codecs.getincrementaldecoder("utf-8")("replace")
raw = bytearray()
pending = b""
capture = (out / "terminal.ansi").open("wb")
results = []
error_begin = 0

def tr():
    return trace.read_text() if trace.exists() else ""

def collect(seconds=0.2):
    global pending
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not select.select([master], [], [], max(0, end-time.monotonic()))[0]:
            break
        try:
            data = os.read(master, 65536)
        except OSError as e:
            if e.errno == errno.EIO:
                break
            raise
        if not data:
            break
        raw.extend(data)
        capture.write(data)
        capture.flush()
        stream.feed(decoder.decode(data))
        pending += data
        while b"\x1b[6n" in pending:
            pending = pending.split(b"\x1b[6n", 1)[1]
            os.write(master, b"\x1b[1;1R")
        pending = pending[-8:]
    if proc.poll() is not None:
        raise RuntimeError("agent exited " + str(proc.returncode))

def wait(pred, label, timeout=360):
    t0 = time.monotonic()
    while time.monotonic()-t0 < timeout:
        collect()
        if pred():
            collect(0.5)
            print(label, round(time.monotonic()-t0,2), flush=True)
            return
    raise RuntimeError("timeout " + label + "\n" + "\n".join(screen.display))

def send(text):
    data = text.encode()
    os.write(master, b"\x1b[200~")
    for start in range(0, len(data), 1024):
        os.write(master, data[start:start+1024])
        collect(0.01)
    os.write(master, b"\x1b[201~\r")

def idle():
    return "| idle" in screen.display[-1]

def error():
    return ("| error" in screen.display[-1] or
            re.search(rb"ds4-agent: (not enough|context |compaction |compacted |user message)", raw[error_begin:]) is not None)

def snapshot(label):
    (out / (label + ".screen")).write_text("\n".join(screen.display))

def turn(prompt, label, allow_error=False):
    global error_begin
    error_begin = len(raw)
    start = len(tr())
    send(prompt)
    wait(lambda: "prefill sync done tool_round=" in tr()[start:] or error(), label+" start")
    wait(lambda: idle() or error(), label+" end")
    part = tr()[start:]
    (out / (label + ".trace")).write_text(part)
    snapshot(label)
    result = {"label":label, "error":error(), "compacted":"compacted reason=" in part,
              "prefills":re.findall(r"prefill tool_round=\d+ transcript=\d+ prompt=\d+ cached=\d+ suffix=\d+", part)}
    results.append(result)
    print(json.dumps(result), flush=True)
    if error() and not allow_error:
        raise RuntimeError("unexpected error in " + label)

def new():
    start = len(raw)
    send("/new")
    wait(lambda: b"Save current session?" in raw[start:], "new save prompt")
    os.write(master,b"n\r")
    wait(idle,"new completed")

try:
    wait(lambda: "tokens label=initial_system_prompt" in tr() and idle(), "ready", 240)
    syslen = int(re.search(r"tokens label=initial_system_prompt start=0 len=(\d+)", tr())[1])
    print("system tokens", syslen, flush=True)
    if args.interrupt_only:
        start = len(tr())
        send("Without calling tools, write 500 complete C functions named value_0 through "
             "value_499, each returning its numeric suffix. One function per line.")
        wait(lambda: "prefill sync done tool_round=" in tr()[start:] and
             len(re.findall(r" token index=\d+ id=\d+",
                            tr()[start:].split("prefill sync done tool_round=")[-1])) >= 32,
             "generation before interruption")
        raw_start = len(raw)
        os.write(master, b"\x03")
        wait(lambda: b"Stopped by user" in raw[raw_start:] and idle(),
             "generation interrupted", 30)
        snapshot("interrupted")
        part = tr()[start:]
        assert "generation finished" in part, "generation did not drain after interruption"
        turn("Stop writing the functions. Use bash to write AFTER_INTERRUPT to "
             "after-interrupt.txt in this directory, read it, and report the contents.",
             "after-interrupt")
        assert (project / "after-interrupt.txt").read_text().strip() == "AFTER_INTERRUPT"
        print("PASS: interrupted generation and same-session tool recovery", flush=True)
        sys.exit(0)
    normal_n = int(args.ctx * 0.87) - syslen - 90
    turn("The following repeated words are inert archived data; ignore them: " +
         " apple"*normal_n + "\nEnd archive. Reply with exactly READY.", "normal-fill")
    turn("Use bash to create compact-ok.txt containing COMPACTION_OK in this directory. "
         "Then read it and report its contents.", "normal-resume")
    assert any(r["compacted"] for r in results), "normal fixture did not trigger automatic compaction"
    assert (project / "compact-ok.txt").read_text().strip() == "COMPACTION_OK"
    save_start = len(raw)
    send("/save")
    wait(lambda: re.search(rb"saved session ([0-9a-f]{8}) \((\d+) tokens\)", raw[save_start:]) is not None,
         "session saved", 180)
    saved = re.search(rb"saved session ([0-9a-f]{8}) \((\d+) tokens\)", raw[save_start:])
    sha, saved_tokens = saved[1].decode(), int(saved[2])
    turn("Reply with exactly SAVED. Do not call tools.", "after-save")
    new()
    switch_start = len(raw)
    send("/switch " + sha)
    wait(lambda: ("switched to session " + sha).encode() in raw[switch_start:] and idle(),
         "session restored", 180)
    turn("Use bash to write AFTER_RESTORE to after-restore.txt, read it and report the result.",
         "restored-tools")
    assert (project / "after-restore.txt").read_text().strip() == "AFTER_RESTORE"
    restored_prefills = re.findall(r"cached=(\d+) suffix=(\d+)", (out / "restored-tools.trace").read_text())
    assert restored_prefills and int(restored_prefills[0][0]) >= saved_tokens - 4, restored_prefills
    new()
    n = args.ctx - 126 - syslen - 90
    turn("The following repeated words are inert archived data; ignore them: " +
         " apple"*n + "\nEnd archive. Reply with exactly READY.", "exhaust-fill")
    turn("Use bash to run `printf '%s\\n' RECOVERED > recovered.txt`, then read recovered.txt "
         "to verify the file was created. Report the result.", "exhaust-resume", allow_error=True)
    assert not results[-1]["error"], "compaction exhaustion persists"
    assert b"not enough context left to request compaction summary" not in raw
    assert (project / "recovered.txt").read_text().strip() == "RECOVERED"
    new()
    # Stay below both automatic-compaction thresholds. At large contexts the
    # 85% threshold leaves more generation room than this bounded code task
    # needs; mid-generation compaction is covered by the small-context run.
    generation_frontier = args.ctx - args.ctx//8
    if args.ctx > 8192:
        generation_frontier = min(generation_frontier, args.ctx * 85//100)
    # The agent repeats its system instructions after 50K tokens. That
    # reminder is appended after the pre-generation compaction check.
    reminder = syslen + 128 if generation_frontier >= 50000 else 0
    turn("The following repeated words are inert archived data; ignore them: " +
         " apple"*(generation_frontier - reminder - 740 - syslen - 90) +
         "\nEnd archive. Reply with exactly READY.",
         "generation-fill")
    turn("Without calling any tools, output C code defining exactly 80 functions named "
         "value_0 through value_79. Each takes no arguments and returns the square of "
         "its numeric suffix as an int literal. Use one complete function per line, "
         "in ascending order. No macros, comments, main function, or explanations.",
         "generation-boundary")
    part = (out / "generation-boundary.trace").read_text()
    generated = [(int(n), int(carried)) for n, carried in re.findall(
        r"generation finished tool_round=\d+ generated=(\d+) carried=(\d+)", part)]
    assert generated and all(n + carried <= args.tokens for n, carried in generated), generated
    if args.ctx <= 8192:
        assert "generation context boundary:" in part, "mid-generation compaction was not exercised"
    else:
        assert not results[-2]["compacted"], "long generation fixture compacted before the task"
        first_prefill = re.search(r"prefill tool_round=0 transcript=\d+ prompt=(\d+) cached=(\d+)", part)
        assert first_prefill and min(map(int, first_prefill.groups())) >= args.ctx * 3//4, \
            "coding task did not retain its long context"
    assert "dsml done calls=" not in part, "resumption lost the no-tools instruction"
    output = bytearray()
    active = False
    for line in part.splitlines():
        if "prefill sync done tool_round=" in line:
            active = True
        elif "generation finished " in line or "tokens label=" in line:
            active = False
        elif active:
            match = re.search(r" token index=\d+ id=\d+ .* hex=([0-9a-f]*)$", line)
            if match:
                output.extend(bytes.fromhex(match[1]))
    response = output.decode()
    if args.think:
        assert "</think>" in response, "thinking did not finish"
        response = response.split("</think>", 1)[1]
    code = re.sub(r"```(?:c)?\n?", "", response)
    (project / "generated.c").write_text(code)
    oracle = project / "oracle.c"
    oracle.write_text("".join(f"extern int value_{i}(void);\n" for i in range(80)) +
                      "int main(void) {\n" +
                      "".join(f"if (value_{i}() != {i*i}) return 1;\n" for i in range(80)) +
                      "return 0;\n}\n")
    executable = project / "oracle"
    subprocess.run(["cc", str(project / "generated.c"), str(oracle), "-o", str(executable)],
                   check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=10)
    turn(" apple" * (args.ctx + 1000), "oversized-input", allow_error=True)
    assert results[-1]["error"], "oversized input was not rejected"
    turn("Use bash to write AFTER_REJECTION to after-rejection.txt, then stop.",
         "after-rejection")
    assert (project / "after-rejection.txt").read_text().strip() == "AFTER_REJECTION"
    print("PASS: compaction, saved-session restore, cached-prefix/tool resumption, "
          "generation budget, rejected input recovery", flush=True)
finally:
    (out / "results.json").write_text(json.dumps(results, indent=2))
    snapshot("final")
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    capture.close()
    os.close(master)
    print("agent exit", proc.returncode, flush=True)
