#!/usr/bin/env python3
"""Model-free terminal regressions. Requires pyte; pass the agent test binary."""

import codecs
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time

import pyte
from wcwidth import wcswidth


def check_fixtures(binary, directory):
    subprocess.run([binary, "--terminal-fixtures", str(directory)], check=True)
    data = (directory / "status.ansi").read_bytes()
    screen = pyte.Screen(80, 24)
    stream = pyte.Stream(screen)
    seen_prompt = False
    for char in data.decode("utf-8"):
        stream.feed(char)
        visible = "ds4-agent> draft" in screen.display[22]
        if seen_prompt:
            assert visible, "status update erased the live input"
        seen_prompt |= visible
    assert screen.display[23].strip() == "done"
    assert (screen.cursor.y, screen.cursor.x) == (22, 16)
    assert b"\x1b[0K" not in data
    assert data.count(b"\x1b[?2026h") == data.count(b"\x1b[?2026l") == 3
    footer = (directory / "queue-footer.txt").read_text(encoding="utf-8")
    assert all(wcswidth(line) <= 40 for line in footer.splitlines()[:-1])
    print(f"Footer: {len(data)} bytes including setup, no prompt erasure, 3 balanced frames")
    expected = {"google_search", "visit_page", "bash", "bash_status", "bash_stop",
                "read", "more", "write", "edit", "search", "list"}
    for family in ("glm", "dsml"):
        for vision in (0, 1):
            prompt = (directory / f"prompt-{family}-{vision}.txt").read_text()
            schemas = {}
            for match in re.finditer(r"(?m)^\{", prompt):
                schema, _ = json.JSONDecoder().raw_decode(prompt[match.start():])
                schema = schema.get("function", schema)
                assert schema["name"] not in schemas
                schemas[schema["name"]] = schema
            assert set(schemas) == expected | ({"view_image"} if vision else set())
            assert schemas["search"]["parameters"]["properties"]["mode"]["enum"] == ["literal", "regex"]
    print("Tool schemas: both formats parse as JSON, with and without vision")


def check_pty(binary, directory):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    process = subprocess.Popen([binary, "--terminal-driver"], stdin=slave, stdout=slave,
                               stderr=slave, start_new_session=True)
    os.close(slave)
    screen = pyte.Screen(80, 24)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    transcript = bytearray()

    def collect(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            transcript.extend(chunk)
            stream.feed(decoder.decode(chunk))

    try:
        collect(0.3)
        for byte in "a\u4e2db".encode():
            os.write(master, bytes([byte]))
            collect(0.03)
        for byte in b"\x1b[DZ":
            os.write(master, bytes([byte]))
            collect(0.03)
        collect(0.3)
        assert any("ds4-agent> a\u4e2dZb" in line for line in screen.display), screen.display
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 16, 40, 0, 0))
        screen.resize(lines=16, columns=40)
        collect(0.5)
        assert screen.display[14].startswith("ds4-agent> a\u4e2dZb"), screen.display
        assert screen.display[15].startswith("generation "), screen.display
        assert any("model output" in line for line in screen.display[:14]), screen.display
        os.write(master, b"\r")
        collect(0.3)
        assert process.wait(timeout=3) == 0
        result = re.search(rb"RESULT:([0-9a-f]+)", transcript)
        assert result and bytes.fromhex(result[1].decode()).decode() == "a\u4e2dZb"
        assert transcript.count(b"\x1b[?2026h") == transcript.count(b"\x1b[?2026l")
        print("PTY: fragmented UTF-8/navigation, concurrent output, resize, and submission passed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        (directory / "pty.ansi").write_bytes(transcript)


def main():
    binary = str(Path(sys.argv[1] if len(sys.argv) > 1 else "./ds4_agent_test").resolve())
    with tempfile.TemporaryDirectory(prefix="ds4-agent-terminal-") as temporary:
        directory = Path(temporary)
        check_fixtures(binary, directory)
        check_pty(binary, directory)


if __name__ == "__main__":
    main()
