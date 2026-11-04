#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ds4 menubar sidecar — live prefill/decode status in the macOS menu bar.

Independent process: polls the running ds4-server's GET /v1/status once a
second and renders the inference state as menu-bar text. Never imports ds4
code; ds4 upgrades cannot break it.

Usage:
  ds4_menubar.py [--host H] [--port N] [--parent PID]

Normally spawned by `ds4-server --menubar`, which ties the icon's lifetime
to the server. Can also be started by hand (see menubar/README.md).
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.request

from Foundation import NSAttributedString
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSTimer,
)

# NSVariableLengthStatusItem is a C macro, not exported by PyObjC.
NSVariableLengthStatusItem = -1.0

PIDFILE = os.path.expanduser("~/.ds4/menubar/ds4_menubar.pid")
POLL_INTERVAL = 1.0
HTTP_TIMEOUT = 2.5


class StatusClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def fetch(self) -> dict:
        url = f"http://{self.host}:{self.port}/v1/status"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())


def fmt_tokens(n) -> str:
    if not isinstance(n, (int, float)):
        return "?"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_duration(s) -> str:
    if not isinstance(s, (int, float)):
        return ""
    s = max(0, int(round(s)))
    if s >= 60:
        m, r = divmod(s, 60)
        return f"{m}m{r}s" if r else f"{m}m"
    return f"{s}s"


class Poller(threading.Thread):
    def __init__(self, client: StatusClient):
        super().__init__(daemon=True)
        self.client = client
        self.status = None
        self.online = False
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                self.status = self.client.fetch()
                self.online = True
            except Exception:
                if os.environ.get("DS4_MENUBAR_DEBUG"):
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                self.status = None
                self.online = False
            self._stop.wait(POLL_INTERVAL)

    def stop(self):
        self._stop.set()


def _eta(remaining: int, tps: float):
    if tps and tps > 0:
        return remaining / tps
    return None


class MenubarApp:
    def __init__(self, client: StatusClient):
        self.client = client
        self.poller = Poller(client)
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableLengthStatusItem)
        self._last_title = None

    def _live_state(self):
        st = self.poller.status
        if not self.poller.online or st is None:
            return "offline", NSColor.tertiaryLabelColor(), []

        prefills = st.get("prefilling") or []
        decodes = st.get("decoding") or []
        waiting = st.get("waiting") or 0
        active = len(prefills) + len(decodes)
        model = st.get("model") or ""
        slots = st.get("slots") or 0

        if prefills:
            p = prefills[0]
            current = p.get("current") or 0
            total = p.get("total") or 0
            tps = p.get("tps") or 0
            pct = int(round(current / total * 100)) if total else 0
            extra = f" +{active - 1}" if active > 1 else ""
            detail = [model or f"{slots} slot(s)"]
            parts = [f"PP {pct}%  {fmt_tokens(current)}/{fmt_tokens(total)}"]
            if tps > 0:
                parts.append(f"{int(round(tps))} t/s")
                eta = _eta(total - current, tps)
                if eta is not None:
                    parts.append(f"ETA {fmt_duration(eta)}")
            detail.append(" · ".join(parts))
            return f"PP {pct}%{extra}", NSColor.systemBlueColor(), detail

        if decodes:
            g = decodes[0]
            tps = g.get("tps") or 0
            toks = g.get("tokens") or 0
            extra = f" +{active - 1}" if active > 1 else ""
            detail = [model or f"{slots} slot(s)",
                      f"GEN {tps:.1f} t/s · {fmt_tokens(toks)} tok · "
                      f"{fmt_duration(g.get('elapsed'))}"]
            return f"GEN {tps:.1f} t/s{extra}", \
                NSColor.systemGreenColor(), detail

        if waiting > 0:
            return f"WAIT {waiting}", NSColor.systemOrangeColor(), \
                [f"{waiting} request(s) queued"]

        detail = [f"{model} · {slots} slot(s)"] if model else [f"{slots} slot(s)"]
        totals = st.get("totals") or {}
        if totals:
            detail.append(
                f"session avg  PP {totals.get('avg_prefill_tps', 0):.0f} t/s"
                f"  ·  GEN {totals.get('avg_decode_tps', 0):.1f} t/s")
        return "idle", NSColor.secondaryLabelColor(), detail

    def _set_title(self, text: str, color):
        if text == self._last_title:
            return
        self._last_title = text
        if os.environ.get("DS4_MENUBAR_DEBUG"):
            print(f"[menubar] {time.strftime('%H:%M:%S')} {text}",
                  file=sys.stderr, flush=True)
        attrs = {
            NSFontAttributeName: NSFont.menuBarFontOfSize_(0),
            NSForegroundColorAttributeName: color,
        }
        button = self.status_item.button()
        button.setAttributedTitle_(
            NSAttributedString.alloc().initWithString_attributes_(" " + text, attrs))

    def _rebuild_menu(self):
        st = self.poller.status
        online = self.poller.online and st is not None
        menu = NSMenu.alloc().init()

        def add(title, action=None, enabled=True, bold=False):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, None, "")
            item.setEnabled_(bool(enabled) and action is not None)
            if action is not None:
                item.setTarget_(self)
                item.setAction_(action)
            if bold:
                attrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(
                    NSFont.menuBarFontOfSize_(0).pointSize())}
                item.setAttributedTitle_(
                    NSAttributedString.alloc().initWithString_attributes_(
                        title, attrs))
            menu.addItem_(item)
            return item

        host = f"{self.client.host}:{self.client.port}"
        add("ds4 · " + host, bold=True)
        add("---")

        if not online:
            add("server offline")
        else:
            model = st.get("model") or ""
            if model:
                add(model, bold=True)
            prefills = st.get("prefilling") or []
            decodes = st.get("decoding") or []
            for p in prefills:
                current = p.get("current") or 0
                total = p.get("total") or 0
                tps = p.get("tps") or 0
                pct = int(round(current / total * 100)) if total else 0
                line = (f"  slot {p.get('slot')}  PP {pct}%  "
                        f"{fmt_tokens(current)}/{fmt_tokens(total)}")
                if tps > 0:
                    line += f"  ·  {int(round(tps))} t/s"
                    eta = _eta(total - current, tps)
                    if eta is not None:
                        line += f"  ·  ETA {fmt_duration(eta)}"
                add(line)
            for g in decodes[:6]:
                add(f"  slot {g.get('slot')}  GEN {(g.get('tps') or 0):.1f} t/s"
                    f"  ·  {fmt_tokens(g.get('tokens'))} tok"
                    f"  ·  {fmt_duration(g.get('elapsed'))}")
            if len(decodes) > 6:
                add(f"  … {len(decodes) - 6} more decoding")
            if st.get("waiting"):
                add(f"  waiting: {st.get('waiting')}")
            if not (prefills or decodes or st.get("waiting")):
                add("idle — no active requests")
            add("---")

            totals = st.get("totals") or {}
            if totals:
                add(f"session avg  prefill "
                    f"{totals.get('avg_prefill_tps', 0):.0f} t/s  ·  "
                    f"decode {totals.get('avg_decode_tps', 0):.1f} t/s")
                add(f"requests {totals.get('requests', 0)}  ·  prefill "
                    f"{fmt_tokens(totals.get('prefill_tokens', 0))} tok  ·  "
                    f"decode {fmt_tokens(totals.get('decode_tokens', 0))} tok")
                add("---")

        add("Open /v1/status", action="open_status")
        add("Quit Sidecar", action="quit")
        self.status_item.setMenu_(menu)

    def open_status(self, _sender=None):
        import subprocess
        subprocess.Popen(["open",
            f"http://{self.client.host}:{self.client.port}/v1/status"])

    def quit(self, _sender=None):
        NSApplication.sharedApplication().terminate_(None)

    def _tick(self, _timer=None):
        try:
            title, color, _detail = self._live_state()
            self._set_title(title, color)
            self._rebuild_menu()
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)

    def run(self):
        self.poller.start()
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            POLL_INTERVAL, True, lambda _t: self._tick())
        self._tick()
        self.app.run()


def _already_running() -> bool:
    try:
        with open(PIDFILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid != os.getpid()
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="ds4 menubar sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parent", type=int, default=None,
                        help="exit when this PID goes away")
    args = parser.parse_args()

    if _already_running():
        sys.exit(0)
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    if args.parent:
        def watch_parent():
            while True:
                time.sleep(2)
                try:
                    os.kill(args.parent, 0)
                except OSError:
                    os._exit(0)
        threading.Thread(target=watch_parent, daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    MenubarApp(StatusClient(args.host, args.port)).run()


if __name__ == "__main__":
    main()
