#!/usr/bin/env python3
"""Model-free CLI UI regressions. Requires pyte; pass tests/test_cli_ui."""

import codecs
import copy
import errno
import fcntl
import locale
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

import pyte


def utf8_locale():
    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        for name in ("C.UTF-8", "en_US.UTF-8", "UTF-8"):
            try:
                locale.setlocale(locale.LC_CTYPE, name)
                return name
            except locale.Error:
                pass
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)
    raise RuntimeError("a UTF-8 locale is required to test the Braille logo")


class Screen(pyte.Screen):
    """Model xterm's alternate buffer and background color erase."""
    def index(self):
        bottom = self.margins.bottom if self.margins else self.lines - 1
        scrolling = self.cursor.y == bottom
        super().index()
        if scrolling:
            # Real terminals use the active background for newly scrolled rows.
            self.erase_in_line(2)

    def write_process_input(self, data):
        if hasattr(self, "reply"):
            self.reply(data)
    def set_mode(self, *modes, **kwargs):
        if kwargs.get("private") and 1049 in modes:
            state = copy.deepcopy(self.__dict__)
            super().reset()
            self.main_state = state
            modes = tuple(mode for mode in modes if mode != 1049)
        super().set_mode(*modes, **kwargs)

    def reset_mode(self, *modes, **kwargs):
        if kwargs.get("private") and 1049 in modes:
            rows, columns = self.lines, self.columns
            self.__dict__.update(self.main_state)
            self.main_state = None
            self.resize(lines=rows, columns=columns)
            modes = tuple(mode for mode in modes if mode != 1049)
        super().reset_mode(*modes, **kwargs)


class Terminal:
    def __init__(self, binary, *, columns=80, rows=30, no_alt_screen=False, args=None, early_input="", title="DwarfStar Chat", **environment):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        env = dict(os.environ, TERM="xterm-256color", LC_ALL=utf8_locale())
        env.pop("NO_COLOR", None)
        env.update(environment)
        self.no_alt_screen = no_alt_screen
        self.title = title
        args = [binary] + (args or []) + (["--no-alt-screen"] if no_alt_screen else [])
        self.process = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave,
                                        env=env, start_new_session=True)
        os.close(slave)
        self.screen = Screen(columns, rows)
        pending = [early_input]
        self.screen.reply = lambda data: os.write(self.master, ((pending.pop() if pending else "") + data).encode())
        self.stream = pyte.Stream(self.screen)
        self.stream.feed("existing shell output\r\n")
        self.main_display = self.screen.display.copy()
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.output = bytearray()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.collect(0.05)
            frames = self.output.count(b"\x1b[?2026l")
            settled = env["TERM"] == "dumb" or (frames and frames == self.output.count(b"\x1b[?2026h"))
            if settled:
                break

    def collect(self, seconds=0.15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline or (
            time.monotonic() < deadline + 1 and
            self.output.rfind(b"\x1b[?2026h") > self.output.rfind(b"\x1b[?2026l")
        ):
            ready, _, _ = select.select([self.master], [], [], max(0.01, deadline - time.monotonic()))
            if not ready:
                break
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                break
            if not chunk:
                break
            self.output.extend(chunk)
            self.stream.feed(self.decoder.decode(chunk))

    def send(self, text, seconds=0.15):
        os.write(self.master, text.encode() if isinstance(text, str) else text)
        self.collect(seconds)

    def wait_for(self, predicate, message):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.collect(0.05)
        assert predicate(), message

    def resize(self, columns, rows):
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        self.screen.resize(lines=rows, columns=columns)
        self.collect(0.25)

    def logo(self):
        return [(x, y) for y, line in enumerate(self.screen.display)
                for x, char in enumerate(line) if "\u2801" <= char <= "\u28ff"]

    def centered(self):
        cells = self.logo()
        assert cells, self.screen.display
        xs, ys = zip(*cells)
        assert abs((min(xs) + max(xs)) / 2 - (self.screen.columns - 1) / 2) <= 2
        assert abs((min(ys) + max(ys)) / 2 - (self.screen.lines - 5) / 2) <= 1
        assert self.screen.display[0].strip() == self.title
        assert self.screen.cursor.y == self.screen.lines - 3
        assert self.input_text() == ""
        assert "/help" in self.screen.display[-1]

    def input_text(self):
        text = self.screen.display[self.screen.cursor.y].strip()
        return "" if text == "Message DwarfStar..." else text

    def assert_panel(self, first, count):
        for row in range(first, first + count):
            assert all(self.screen.buffer[row][x].bg == "2a303a" for x in range(self.screen.columns)), (row, self.screen.display)
        assert not self.screen.display[first].strip() and not self.screen.display[first + count - 1].strip()

    def footer(self):
        return next(line for line in reversed(self.screen.display) if "/help" in line)

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.send(b"\x15\x04")
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
            assert self.process.wait(timeout=3) == 0
            assert self.screen.margins is None, "UI unexpectedly installed a scroll region"
            entered = self.output.count(b"\x1b[?1049h")
            assert entered == self.output.count(b"\x1b[?1049l")
            if entered:
                assert self.screen.display == self.main_display, "normal terminal contents were not restored"
            attrs = termios.tcgetattr(self.master)
            assert attrs[3] & termios.ICANON and attrs[3] & termios.ECHO
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            os.close(self.master)


def check_interaction(binary):
    terminal = Terminal(binary)
    try:
        terminal.centered()
        footer = terminal.footer()
        for value in ("Preview model (high)", "ctx 0 / 32,768 (0%)", "/help"):
            assert value in footer, footer
        assert not any(value in footer for value in ("metal", "RAM", "GPU", "think "))
        assert not any("Commands" in line or "─" in line for line in terminal.screen.display)
        terminal.send("draft")
        assert not terminal.logo()
        assert terminal.input_text() == "draft"
        terminal.send(b"\x7f" * 5)
        terminal.centered()
        terminal.send(" ")
        assert not terminal.logo()
        terminal.send(b"\x15")  # Ctrl+U empties the input.
        terminal.centered()
        terminal.send(b"\r")  # An empty submission keeps the idle screen.
        terminal.centered()
        terminal.send("cancel\x03")
        terminal.centered()
        terminal.send(b"\x0c")  # Ctrl+L redraws both the logo and footer.
        terminal.centered()
        for columns, rows in ((120, 40), (40, 16), (80, 30)):
            terminal.resize(columns, rows)
            terminal.centered()
        for columns in (4, 1):
            terminal.resize(columns, 6)
            assert not terminal.logo()
        terminal.resize(80, 30)
        terminal.centered()
        # Long input grows upward; the status footer remains at the bottom.
        terminal.send("x" * 500, seconds=0.5)
        assert not terminal.logo()
        assert terminal.footer() == footer
        terminal.send(b"\x15")
        terminal.centered()
        for byte in "a中b".encode():
            terminal.send(bytes([byte]), seconds=0.01)
        terminal.send(b"\x1b[DZ")
        assert terminal.input_text() == "a中Zb"
        terminal.send(b"\r", seconds=0.1)
        assert not terminal.logo()
        assert terminal.screen.display[-1] == footer, "streamed output overwrote the footer"
        terminal.collect(0.4)
        assert not terminal.logo(), "logo returned after the first message"
        assert terminal.input_text() == ""
        assert any("response row 39" in line for line in terminal.screen.display)
        response = next(y for y, line in enumerate(terminal.screen.display) if "response row 39" in line)
        assert all(terminal.screen.buffer[response][x].bg == "default" for x in range(terminal.screen.columns))
        terminal.send("draft\x15")
        assert not terminal.logo()
        terminal.send("/power 50\r")
        assert terminal.footer() == footer
        terminal.send("/help\r")
        assert any("Commands" in line for line in terminal.screen.display)
        assert "/help" in terminal.footer()
    finally:
        terminal.close()
    print("CLI UI: centering, editing, clearing, resize, Unicode, first submission, footer and cleanup passed")


def check_screen_modes(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, no_alt_screen=no_alt_screen)
        try:
            assert (b"\x1b[?1049h" in terminal.output) == (not no_alt_screen)
            if no_alt_screen:
                assert not terminal.logo()
                assert terminal.screen.display[0].startswith("existing shell output")
                assert terminal.screen.display[1].strip() == terminal.title
                assert all(terminal.screen.buffer[1][x].fg == "d7af87" for x in range(2, 2 + len(terminal.title)))
                terminal.send("draft")
                assert terminal.screen.display[0].startswith("existing shell output")
                assert terminal.screen.display[1].strip() == terminal.title
                assert terminal.input_text() == "draft"
                assert terminal.screen.cursor.y == terminal.screen.lines - 3
                terminal.send(b"\x15")
                assert not terminal.logo()
                assert b"\x1b[?1000h" in terminal.output
                terminal.send("/quit\r")
            else:
                terminal.centered()
            assert b"\x1b[2J" not in terminal.output and b"\x1b[3J" not in terminal.output
        finally:
            terminal.close()
        assert b"\x1b[2J" not in terminal.output and b"\x1b[3J" not in terminal.output
        if no_alt_screen:
            assert terminal.screen.display[0].startswith("existing shell output")
            assert b"\x1b[?1049" not in terminal.output
    print("CLI UI: alternate screen restoration and inline gold label preserve terminal contents")


def check_animation(binary):
    terminal = Terminal(binary)
    try:
        terminal.centered()
        cells = terminal.logo()
        xs, ys = zip(*cells)
        idle = terminal.screen.display.copy()
        footer = idle[-1]
        colors = lambda: {terminal.screen.buffer[y][x].fg for x, y in terminal.logo()}
        def shimmer_position():
            glowing = [(x, y) for x, y in terminal.logo()
                       if int(terminal.screen.buffer[y][x].fg[:2], 16) > 215]
            for x, y in glowing:
                color = terminal.screen.buffer[y][x].fg
                red, green, blue = (int(color[i:i + 2], 16) for i in (0, 2, 4))
                assert red - green >= 30 and green - blue >= 30, "shimmer turned pale/white"
            if not glowing: return None
            lit_rows.extend(y for _, y in glowing)
            return sum(((x - min(xs)) / (max(xs) - min(xs)) +
                        (max(ys) - y) / (max(ys) - min(ys))) / 2
                       for x, y in glowing) / len(glowing)
        idle_colors = colors()
        lit_rows = []
        assert len(idle_colors) > 3, "logo lacks internal shading"
        assert max(int(color[:2], 16) for color in idle_colors) <= 108
        assert all(terminal.screen.buffer[y][x].bg == "default" for x, y in cells)
        assert 1000 << 5 in terminal.screen.mode and 1006 << 5 in terminal.screen.mode
        x, y = cells[len(cells) // 2]
        terminal.send(f"\x1b[<0;1;{terminal.screen.lines}M")  # Outside the logo.
        terminal.send(f"\x1b[<2;{x + 1};{y + 1}M")  # Right click.
        assert colors() == idle_colors and terminal.screen.display == idle
        # Fragment a left click and sample both complete diagonal sweeps.
        started = time.monotonic()
        for byte in f"\x1b[<0;{x + 1};{y + 1}M\x1b[<0;{x + 1};{y + 1}m".encode():
            terminal.send(bytes([byte]), seconds=0.005)
        positions, frames = [], []
        while time.monotonic() - started < 3.4:
            terminal.collect(0.12)
            position = shimmer_position()
            if position is not None:
                positions.append(position)
                if len(frames) < 2: frames.append(terminal.screen.display.copy())
        assert len(frames) == 2, "logo did not shimmer"
        first, second = frames
        assert first != second, "flame shape did not move"
        assert second[-1] == footer and second[-3].strip() == "Message DwarfStar..."
        # Shading and glints change color, while lettering/obelisk stay still.
        assert first[max(ys) - 3:max(ys) + 1] == idle[max(ys) - 3:max(ys) + 1]
        left, right = min(xs), max(xs)
        center = slice(left + int((right - left) * .42), left + int((right - left) * .62))
        for row in range(min(ys), max(ys) - 4):
            assert first[row][center] == idle[row][center], "obelisk moved with the flames"
        resets = [i + 1 for i, (a, b) in enumerate(zip(positions, positions[1:])) if a - b > .35]
        assert len(resets) == 1, "expected exactly two shimmer passes"
        for sweep in (positions[:resets[0]], positions[resets[0]:]):
            assert all(a <= b + .04 for a, b in zip(sweep, sweep[1:])), "shimmer moved backward"
            assert sweep[-1] - sweep[0] > .4
        assert min(lit_rows) < min(ys) + (max(ys) - min(ys)) * .4
        assert max(lit_rows) > max(ys) - (max(ys) - min(ys)) * .2
        assert terminal.screen.display == idle and colors() == idle_colors
        before = len(terminal.output)
        terminal.collect(0.2)
        assert len(terminal.output) == before, "idle logo kept repainting"
        terminal.send(f"\x1b[<0;{x + 1};{y + 1}M")
        terminal.send("draft")
        assert not terminal.logo() and 1000 << 5 in terminal.screen.mode
        terminal.send(f"\x1b[<0;{x + 1};{y + 1}m")  # Late release is not input.
        assert terminal.input_text() == "draft"
        terminal.send(b"\x15")
        terminal.centered()
        assert colors() == idle_colors
    finally:
        terminal.close()
    assert terminal.output.count(b"\x1b[?1000h") == terminal.output.count(b"\x1b[?1000l")
    assert terminal.output.count(b"\x1b[?1006h") == terminal.output.count(b"\x1b[?1006l")
    print("CLI UI: click fade, flames, two diagonal gold shimmers, internal shading, idle return and mouse cleanup passed")


def check_user_panels(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, no_alt_screen=no_alt_screen)
        try:
            text_row = terminal.screen.cursor.y
            terminal.assert_panel(text_row - 1, 3)
            terminal.send("draft")
            assert terminal.input_text() == "draft"
            terminal.assert_panel(terminal.screen.cursor.y - 1, 3)
            assert terminal.screen.buffer[terminal.screen.cursor.y][2].fg == "d0d0d0"
            terminal.send(b"\x15")
            terminal.send("x" * 78)
            terminal.assert_panel(terminal.screen.cursor.y - 2, 4)
            terminal.send(b"\x15")
            # Commands use the same sent-message panel, without long model output.
            terminal.send("/power 50\r")
            message = next(y for y, line in enumerate(terminal.screen.display) if line.strip() == "/power 50")
            terminal.assert_panel(message - 1, 3)
            terminal.assert_panel(terminal.screen.cursor.y - 1, 3)
            assert all(terminal.screen.buffer[terminal.screen.cursor.y + 2][x].bg == "default"
                       for x in range(terminal.screen.columns)), "input background leaked into footer"
            assert not any("you >" in line or "ds4 >" in line for line in terminal.screen.display)
            # Unicode and exact column wrapping keep the panel full width.
            terminal.send("中" * 80, seconds=0.4)
            assert terminal.screen.cursor.y < terminal.screen.lines - 1
            if not no_alt_screen:
                terminal.assert_panel(terminal.screen.lines - 6, 5)
            terminal.send(b"\x15")
        finally:
            terminal.close()
    print("CLI UI: unlabeled input, full-width shaded panels, vertical padding, Unicode wrapping and clean footer passed")


def check_answers(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, columns=20, rows=30, no_alt_screen=no_alt_screen)
        try:
            terminal.send("/layout\r")
            display = terminal.screen.display
            message = next(y for y, line in enumerate(display) if line.strip() == "/layout")
            first = next(y for y, line in enumerate(display) if line.strip() == "answer first")
            terminal.assert_panel(message - 1, 3)
            assert first == message + 3, "missing plain bottom margin after the user panel"
            assert not display[first - 1].strip()
            assert all(terminal.screen.buffer[first - 1][x].bg == "default" for x in range(20))
            assert [line.strip() for line in display[first:first + 6]] == [
                "answer first", "abcdefghijklmnop", "qrstuv", "中é", "tab end", "colored"]
            for y in range(first, first + 6):
                assert display[y].startswith("  ") and display[y].endswith("  "), display[y]
                assert all(terminal.screen.buffer[y][x].bg == "default" for x in range(20))
        finally:
            terminal.close()
    print("CLI UI: answer side margins, right-edge wrapping, UTF-8 chunks and user/answer spacing passed")


def check_capabilities(binary):
    for environment in ({"LC_ALL": "C"}, {"LC_ALL": "ds4_invalid_locale"}, {"TERM": "dumb"}):
        terminal = Terminal(binary, **environment)
        try:
            assert not terminal.logo(), environment
            if environment.get("TERM") == "dumb":
                assert b"\x1b" not in terminal.output
                terminal.send("/quit\r")
        finally:
            terminal.close()
    terminal = Terminal(binary, NO_COLOR="1")
    try:
        terminal.centered()
        assert not re.search(rb"\x1b\[[0-9;]*m", terminal.output)
    finally:
        terminal.close()
    result = subprocess.run([binary], input="/quit\n", capture_output=True, text=True, check=True)
    assert not result.stderr and "\x1b" not in result.stdout
    assert not any("\u2801" <= char <= "\u28ff" for char in result.stdout)
    # Either redirected stream prevents alternate-screen entry and raw editing.
    for tty_input in (False, True):
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 80, 0, 0))
            process = subprocess.Popen([binary], stdin=slave if tty_input else subprocess.PIPE,
                                       stdout=subprocess.PIPE if tty_input else slave,
                                       stderr=subprocess.PIPE)
            os.close(slave)
            if tty_input:
                os.write(master, b"/quit\n")
                output, errors = process.communicate(timeout=3)
            else:
                _, errors = process.communicate(input=b"/quit\n", timeout=3)
                output = bytearray()
                while True:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as error:
                        if error.errno == errno.EIO: break
                        raise
                    if not chunk: break
                    output.extend(chunk)
            assert process.returncode == 0 and not errors
            assert b"\x1b[?1049" not in output
        finally:
            os.close(master)
    print("CLI UI: unsupported locales/terminals, NO_COLOR and pipes passed")


def check_content_layout(binary):
    terminal = Terminal(binary)
    try:
        terminal.centered()
        terminal.send("/layout\r")
        display = terminal.screen.display
        assert display[0].strip() == terminal.title
        assert not display[1].strip()
        assert display[3].strip() == "/layout", "conversation did not start below the title"
        assert display[6].strip() == "answer first"

        # Exercise both a short conversation and a completely filled viewport.
        for filled in (False, True):
            if filled:
                terminal.send("fill\r", seconds=0.5)
            baseline = terminal.screen.display.copy()
            for length in (80, 240, 500, 160):
                terminal.send("x" * length, seconds=0.2)
                terminal.send(b"\x15")
                terminal.wait_for(lambda: terminal.screen.display == baseline,
                                  "input shrink lost conversation rows")
            terminal.send("x" * 100)
            terminal.send(b"\x7f" * 100)
            terminal.wait_for(lambda: terminal.screen.display == baseline,
                              "backspacing left a gap in the conversation")

        # Sending a two-line message shrinks the editor and adds exactly its
        # own panel rows and symmetric outer gaps; reserved editor rows return.
        terminal.send("/power 50" + " " * 90 + "\r")
        bottom = terminal.screen.margins.bottom
        assert terminal.screen.display[1:bottom - 4] == baseline[6:bottom + 1]
        assert terminal.screen.display[bottom - 2].strip() == "/power 50"
        assert terminal.screen.display[0].strip() == terminal.title
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        assert "GPU" not in terminal.footer()
    finally:
        terminal.close()
    print("CLI UI: fixed title, top-aligned content, reversible input growth, deletion and submission passed")


def check_system_padding(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, columns=40, rows=30, no_alt_screen=no_alt_screen)
        try:
            terminal.send("/system\r")
            display = terminal.screen.display
            first = next(y for y, line in enumerate(display) if "processing" in line)
            assert [line.strip() for line in display[first:first + 4]] == [
                "processing 10 tokens: done", "ds4: prefill: 120 t/s",
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ", "KLMNOPQRSTUVWXYZ"]
            for line in display[first:first + 4] + [terminal.footer()]:
                assert line.startswith("  ") and line.endswith("  "), line
            before = terminal.screen.display.copy()
            terminal.send("x" * 100)
            terminal.send(b"\x15")
            if not no_alt_screen:
                assert terminal.screen.display == before, "progress output was lost during repaint"
        finally:
            terminal.close()
    print("CLI UI: processing/prefill padding, carriage-return updates, wrapping and footer margins passed")


def check_footer_placeholder(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, no_alt_screen=no_alt_screen)
        try:
            y = terminal.screen.cursor.y
            assert terminal.screen.display[y].strip() == "Message DwarfStar..."
            assert terminal.screen.cursor.x == 2
            assert terminal.screen.buffer[y][2].fg == "brightblack"
            terminal.assert_panel(y - 1, 3)
            footer = terminal.footer()
            assert "Preview model (high) | ctx 0 / 32,768 (0%)" in footer
            assert not any(text in footer for text in ("metal", "RAM", "GPU", "think "))
            footer_y = terminal.screen.display.index(footer)
            assert all(terminal.screen.buffer[footer_y][x].fg == "brightwhite"
                       for x, char in enumerate(footer) if char != " ")
            terminal.send("draft")
            assert terminal.screen.display[terminal.screen.cursor.y].strip() == "draft"
            terminal.send(b"\x15")
            assert terminal.screen.display[terminal.screen.cursor.y].strip() == "Message DwarfStar..."
            terminal.send("/context\r")
            assert "ctx 8,192 / 32,768 (25%)" in terminal.footer()
            terminal.send("/status\r")
            for text in ("metal", "Model in RAM", "GPU power 100%"):
                assert any(text in line for line in terminal.screen.display), text
        finally:
            terminal.close()
    terminal = Terminal(binary, columns=20)
    try:
        line = terminal.screen.display[terminal.screen.cursor.y]
        assert line == "  Message DwarfSta  ", line
        terminal.send("/power 50\r")
        assert not any(line.strip() == "Message DwarfSta" for line in terminal.screen.display[:-4])
    finally:
        terminal.close()
    print("CLI UI: white compact footer, context usage, status details and non-editable placeholder passed")


def check_message_spacing(binary):
    for no_alt_screen in (False, True):
        terminal = Terminal(binary, no_alt_screen=no_alt_screen)
        try:
            terminal.send("/spacing\r")
            terminal.send("/power 50\r")
            display = terminal.screen.display
            user = next(y for y, line in enumerate(display) if line.strip() == "/spacing")
            processing = next(y for y, line in enumerate(display) if line.strip() == "processing: done")
            answer = next(y for y, line in enumerate(display) if line.strip() == "answer text")
            timing = next(y for y, line in enumerate(display) if line.strip() == "ds4: prefill: 120 t/s")
            command = next(y for y, line in enumerate(display) if line.strip() == "/power 50")
            assert processing == user + 3
            assert answer == processing + 2 and timing == answer + 2
            assert command == timing + 3
            for y in (user - 2, processing - 1, answer - 1, timing - 1, command - 2, command + 2):
                assert not display[y].strip(), (y, display)
                assert all(terminal.screen.buffer[y][x].bg == "default" for x in range(80))
            for y in (user, command):
                terminal.assert_panel(y - 1, 3)
            before = terminal.screen.display.copy()
            terminal.send("x" * 160)
            terminal.send(b"\x15")
            if not no_alt_screen:
                terminal.wait_for(lambda: terminal.screen.display == before,
                                  "repaint changed message spacing")
        finally:
            terminal.close()
    print("CLI UI: symmetric user margins and consistent progress/answer/timing block spacing passed")


def check_scroll_cursor(binary):
    terminal = Terminal(binary)
    try:
        terminal.send("fill\r", seconds=0.5)
        terminal.send("draft")
        bottom = terminal.screen.display.copy()
        position = (terminal.screen.cursor.y, terminal.screen.cursor.x)
        terminal.send("\x1b[<64;20;10M")
        assert terminal.screen.display[:26] != bottom[:26]
        assert terminal.input_text() == "draft"
        assert terminal.screen.display[-4:] == bottom[-4:]
        assert (terminal.screen.cursor.y, terminal.screen.cursor.x) == position
        terminal.send("\x1b[<65;20;10M")
        assert terminal.screen.display == bottom
        terminal.send("\x1b[<64;20;28M")  # Wheel over input never recalls history.
        assert terminal.screen.display == bottom
        terminal.send(b"\x15\x1b[A")
        assert terminal.input_text() == "fill", "Up arrow stopped recalling history"
        terminal.send(b"\x15")
        terminal.send("/slow\r", seconds=0.6)
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        assert terminal.screen.cursor.x == 2 and 25 << 5 in terminal.screen.mode
        terminal.send("must not send\r", seconds=0.1)
        assert not any("must not send" in line for line in terminal.screen.display)
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        terminal.send("\x1b[<64;20;10M", seconds=0.05)
        older = terminal.screen.display[:26]
        terminal.collect(0.2)
        assert terminal.screen.display[:26] == older, "streaming moved a scrolled conversation"
        terminal.resize(60, 24)
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        terminal.collect(1.5)
        assert terminal.input_text() == ""
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        terminal.resize(80, 30)
        terminal.send("\x1b[<65;20;10M" * 50)
        assert any("response row 99" in line for line in terminal.screen.display)
        assert terminal.input_text() == ""
    finally:
        terminal.close()
    print("CLI UI: input-only cursor, busy input discard, wheel scrollback and arrow-only history passed")


def check_input_gap(binary):
    for inline in (False, True):
        for color in (True, False):
            cases = ("partial", "line", "blank", "paragraph", "system", "user") if color else ("line", "user")
            for kind in cases:
                terminal = Terminal(binary, no_alt_screen=inline, **({} if color else {"NO_COLOR": "1"}))
                try:
                    terminal.send("/margin " + kind + "\r")
                    display = terminal.screen.display
                    gap = terminal.screen.cursor.y - 2
                    assert not display[gap].strip()
                    assert all(terminal.screen.buffer[gap][x].bg == "default" for x in range(80))
                    if kind == "user":
                        assert display[gap - 2].strip() == "last user", (kind, display)
                        assert not display[gap - 1].strip()
                        if color: terminal.assert_panel(gap - 3, 3)
                    else:
                        expected = "last system" if kind == "system" else "last answer"
                        assert display[gap - 1].strip() == expected, (kind, display)
                    if kind == "paragraph":
                        first = next(y for y, line in enumerate(display) if "paragraph one" in line)
                        assert display[first + 2].strip() == "last answer"
                        assert not display[first + 1].strip()
                    before = display.copy()
                    terminal.send("x" * 160)
                    terminal.send(b"\x15")
                    terminal.wait_for(lambda: terminal.screen.display == before,
                                      "input resizing changed the collapsed margin")
                    # The margin is still present when another content block
                    # follows; collapsing it at the input must not eat it.
                    terminal.send("/power 50\r")
                    display = terminal.screen.display
                    message = next(y for y, line in enumerate(display) if line.strip() == "/power 50")
                    assert not display[message - 2].strip()
                    previous = next(y for y, line in enumerate(display)
                                    if line.strip() == ("last user" if kind == "user" else
                                                        "last system" if kind == "system" else "last answer"))
                    assert message - previous >= (4 if kind == "user" else 3)
                finally:
                    terminal.close()
    print("CLI UI: collapsed input gap for streaming, trailing newlines, user padding, system output, paragraphs and NO_COLOR passed")


def check_newlines_inline(binary):
    terminal = Terminal(binary, no_alt_screen=True, early_input="a中")
    try:
        terminal.wait_for(lambda: terminal.input_text() == "a中", "startup cursor query swallowed typed input")
        assert terminal.screen.display[0].startswith("existing shell output")
    finally:
        terminal.close()
    for inline in (False, True):
        for newline in ("\x1b[13;2u", "\x1b[27;2;13~", "\n"):
            terminal = Terminal(binary, no_alt_screen=inline)
            try:
                terminal.send("/echo alpha")
                for byte in newline.encode(): terminal.send(bytes([byte]), seconds=0.005)
                terminal.send("β中omega")
                y = terminal.screen.cursor.y
                assert terminal.screen.display[y - 1].strip() == "/echo alpha"
                assert terminal.input_text() == "β中omega"
                assert terminal.screen.cursor.x == 10
                terminal.assert_panel(y - 2, 4)
                gap = y - 3
                assert not terminal.screen.display[gap].strip()
                assert all(terminal.screen.buffer[gap][x].bg == "default" for x in range(80))
                assert not any("received:" in line for line in terminal.screen.display)
                terminal.send("\r")
                display = terminal.screen.display
                first = next(y for y, line in enumerate(display) if line.strip() == "/echo alpha")
                assert display[first + 1].strip() == "β中omega"
                answer = next(y for y, line in enumerate(display) if "received: alpha" in line)
                assert answer == first + 4 and display[answer + 1].strip() == "β中omega"
                terminal.send("/echo ab\x1b[D\x1b[13;2u中")
                y = terminal.screen.cursor.y
                assert terminal.screen.display[y - 1].strip() == "/echo a"
                assert terminal.input_text() == "中b" and terminal.screen.cursor.x == 4
                terminal.send("\x1b[D\x7f")
                assert terminal.input_text() == "/echo a中b"
                assert terminal.screen.cursor.y == terminal.screen.lines - 3
                terminal.send(b"\x15")
            finally:
                terminal.close()
    terminal = Terminal(binary, no_alt_screen=True)
    try:
        terminal.send("/slow\r", seconds=0.2)
        for _ in range(6):
            terminal.collect(0.1)
            assert terminal.screen.cursor.y == terminal.screen.lines - 3
            assert "/help" in terminal.screen.display[-1]
            terminal.assert_panel(terminal.screen.lines - 4, 3)
            gap = terminal.screen.lines - 5
            assert not terminal.screen.display[gap].strip()
            assert all(terminal.screen.buffer[gap][x].bg == "default" for x in range(80))
        terminal.resize(60, 24)
        assert terminal.screen.cursor.y == terminal.screen.lines - 3
        assert "/help" in terminal.screen.display[-1]
        terminal.collect(1)
    finally:
        terminal.close()
    print("CLI UI: Shift+Enter/CSI-u/modifyOtherKeys, Ctrl+J, editable newlines, outer gap and pinned inline footer passed")
    for inline in (False, True):
        terminal = Terminal(binary, no_alt_screen=inline)
        try:
            terminal.send("/slow\r", seconds=0.2)
            terminal.send("\x1b[99;5u", seconds=0.2)
            terminal.wait_for(lambda: terminal.screen.display[terminal.screen.cursor.y].strip() == "Message DwarfStar...",
                              "enhanced Ctrl+C did not interrupt generation")
            assert not any("response row 99" in line for line in terminal.screen.display)
        finally:
            terminal.close()


if __name__ == "__main__":
    binary = str(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/test_cli_ui").resolve())
    check_interaction(binary)
    check_capabilities(binary)
    check_screen_modes(binary)
    check_user_panels(binary)
    check_answers(binary)
    check_content_layout(binary)
    check_system_padding(binary)
    check_footer_placeholder(binary)
    check_message_spacing(binary)
    check_input_gap(binary)
    check_scroll_cursor(binary)
    check_newlines_inline(binary)
    check_animation(binary)
