#!/usr/bin/env bash
set -euo pipefail

# Manual hardware validation for the interactive local multi-GPU ROCm path.
# This depends on private Pro Q4 shard names by default and self-skips outside
# that environment.

skip() {
    echo "SKIP rocm pro q4 multiturn smoke: $*"
    exit 0
}

DS4_BIN=${DS4_BIN:-./ds4}
MODEL0=${DS4_PRO_Q4_MODEL0:-gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf}
MODEL1=${DS4_PRO_Q4_MODEL1:-gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf}
GPUS=${DS4_GPUS:-0,1,2,3,4,5,6,7}
CTX=${DS4_CTX:-262144}
PREFILL_CHUNK=${DS4_PREFILL_CHUNK:-4096}
DIST_PREFILL_CHUNK=${DS4_DIST_PREFILL_CHUNK:-4096}
DIST_PREFILL_WINDOW=${DS4_DIST_PREFILL_WINDOW:-8}
TOKENS=${DS4_MULTITURN_SMOKE_TOKENS:-64}
TIMEOUT=${DS4_MULTITURN_SMOKE_TIMEOUT:-360}

[[ -x "$DS4_BIN" ]] || skip "$DS4_BIN is not executable"
[[ -f "$MODEL0" ]] || skip "$MODEL0 is missing"
[[ -f "$MODEL1" ]] || skip "$MODEL1 is missing"
if [[ ! -e /dev/kfd && ! -e /dev/dri ]]; then
    skip "ROCm device nodes are not available"
fi

REQ_GPUS=$(python3 - "$GPUS" <<'PY'
import sys
print(len([p for p in sys.argv[1].split(",") if p.strip()]))
PY
)
if command -v rocminfo >/dev/null 2>&1; then
    HAVE_GPUS=$(rocminfo 2>/dev/null | grep -E 'Name:[[:space:]]+gfx' | wc -l | tr -d ' ')
    if [[ "$HAVE_GPUS" != "0" && "$HAVE_GPUS" -lt "$REQ_GPUS" ]]; then
        skip "requested $REQ_GPUS GPUs but rocminfo reports $HAVE_GPUS"
    fi
fi

LOG=${DS4_MULTITURN_SMOKE_LOG:-$(mktemp -t ds4-rocm-pro-q4-multiturn.XXXXXX.log)}

python3 - "$LOG" "$TIMEOUT" "$REQ_GPUS" "$DS4_BIN" "$MODEL0" "$MODEL1" "$GPUS" "$CTX" \
    "$PREFILL_CHUNK" "$DIST_PREFILL_CHUNK" "$DIST_PREFILL_WINDOW" "$TOKENS" <<'PY'
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time

(
    log_path,
    timeout_s,
    req_gpus_s,
    ds4_bin,
    model0,
    model1,
    gpus,
    ctx,
    prefill_chunk,
    dist_prefill_chunk,
    dist_prefill_window,
    tokens,
) = sys.argv[1:13]

timeout = float(timeout_s)
req_gpus = int(req_gpus_s)
argv = [
    ds4_bin,
    "--rocm",
    "-m", model0,
    "-m", model1,
    "--gpus", gpus,
    "--ctx", ctx,
    "--prefill-chunk", prefill_chunk,
    "--dist-prefill-chunk", dist_prefill_chunk,
    "--dist-prefill-window", dist_prefill_window,
    "--tokens", tokens,
    "--nothink",
    "--temp", "0",
]

master, slave = pty.openpty()
proc = subprocess.Popen(
    argv,
    stdin=slave,
    stdout=slave,
    stderr=slave,
    close_fds=True,
    start_new_session=True,
)
os.close(slave)

buf = bytearray()
scan_pos = 0
deadline = time.monotonic() + timeout

def write_log():
    with open(log_path, "wb") as f:
        f.write(buf)

def fail(msg):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
    write_log()
    print(f"FAIL multiturn smoke: {msg}", file=sys.stderr)
    print(f"log: {log_path}", file=sys.stderr)
    sys.exit(1)

def drain_once(label):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        fail(f"timed out waiting for {label}")
    r, _, _ = select.select([master], [], [], min(1.0, remaining))
    if not r:
        if proc.poll() is not None:
            fail(f"process exited while waiting for {label} (rc={proc.returncode})")
        return
    try:
        chunk = os.read(master, 65536)
    except OSError as exc:
        if proc.poll() is not None:
            fail(f"process exited while waiting for {label} (rc={proc.returncode})")
        fail(f"pty read failed while waiting for {label}: {exc}")
    if chunk:
        buf.extend(chunk)
        if b"\x1b[6n" in chunk:
            os.write(master, b"\x1b[1;1R")

def read_until(pattern, label):
    global scan_pos
    pat = pattern if isinstance(pattern, bytes) else pattern.encode()
    start = scan_pos
    while True:
        found = buf.find(pat, scan_pos)
        if found >= 0:
            scan_pos = found + len(pat)
            return bytes(buf[start:scan_pos])
        drain_once(label)

def send_line(text):
    os.write(master, text.encode() + b"\r")

ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

def plain(data):
    return ansi.sub("", data.decode("utf-8", errors="replace"))

def meaningful_output(segment, prompt):
    text = plain(segment).replace(prompt, "")
    lines = []
    for raw in text.splitlines():
        line = raw.strip().replace("ds4> ", "").strip()
        if not line or line.startswith("ds4:"):
            continue
        lines.append(line)
    visible = re.sub(r"[^A-Za-z0-9]+", "", " ".join(lines))
    return len(visible) >= 8

read_until(b"ds4> ", "initial prompt")
def run_turn(prompt, label):
    start = len(buf)
    send_line(prompt)
    read_until(b"ds4: prefill:", f"{label} timing")
    read_until(b"ds4> ", f"{label} prompt")
    return bytes(buf[start:scan_pos])

first = "tell me about yourself"
seg1 = run_turn(first, "first answer")
if not meaningful_output(seg1, first):
    fail("first turn produced no meaningful output")

second = "tell me a short story about a lighthouse"
seg2 = run_turn(second, "second answer")
if not meaningful_output(seg2, second):
    fail("second turn produced no meaningful output")

send_line("/exit")

while proc.poll() is None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        fail("timed out waiting for process exit")
    r, _, _ = select.select([master], [], [], min(1.0, remaining))
    if r:
        try:
            chunk = os.read(master, 65536)
        except OSError:
            break
        if chunk:
            buf.extend(chunk)
            if b"\x1b[6n" in chunk:
                os.write(master, b"\x1b[1;1R")

if proc.returncode is None:
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        fail("process did not exit after /exit")

write_log()
text = plain(buf)
bad = [
    "Kernel Name:",
    "HSA_STATUS_ERROR_EXCEPTION",
    "unspecified launch failure",
    "prompt processing failed",
    "decode failed",
    "Aborted",
    "unable to connect to 127.0.0.1",
]
for needle in bad:
    if needle in text:
        print(f"FAIL multiturn smoke saw {needle!r}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        sys.exit(1)

if req_gpus > 1 and "local GPU worker: coordinator disconnected; exiting" not in text:
    print("FAIL multiturn smoke did not observe local worker shutdown", file=sys.stderr)
    print(f"log: {log_path}", file=sys.stderr)
    sys.exit(1)

if proc.returncode != 0:
    print(f"FAIL multiturn smoke exited rc={proc.returncode}", file=sys.stderr)
    print(f"log: {log_path}", file=sys.stderr)
    sys.exit(1)

print("PASS multiturn smoke")
PY

echo "log: $LOG"
