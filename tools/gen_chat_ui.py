#!/usr/bin/env python3
"""Regenerate the chat UI embedded in ds4_server.c from chat-ui.html.

The UI exists twice: chat-ui.html is the editable source and can be opened
straight from disk, while ds4_server.c serves the same page from a string
literal so the binary needs no external assets. Only the API base URL differs.

The embedded copy is generated rather than escaped by hand because the
Markdown and LaTeX regexes are backslash-heavy, and one missed backslash
produces a page that still compiles but renders wrongly.

    tools/gen_chat_ui.py            regenerate ds4_server.c in place
    tools/gen_chat_ui.py --check    exit 1 if it is out of date (no write)

After regenerating, rebuild:  make
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "chat-ui.html")
SERVER = os.path.join(ROOT, "ds4_server.c")

# The served copy talks to its own origin; the standalone file needs an absolute URL.
STANDALONE_BASE = "const BASE = 'http://127.0.0.1:8000';"
EMBEDDED_BASE = "const BASE = '';"
TEST_HOOK = "  // exposed for tests\n  window.__ds4 = { renderMarkdown: renderMarkdown };\n"

START = '        http_response(fd, s->enable_cors, 200, "text/html",\n'
END = ');\n        http_request_free(&hr);'


def c_escape(line):
    line = line.replace("\\", "\\\\").replace('"', '\\"')
    return line.replace("??", "?\\?")          # defuse trigraphs


def build_literal():
    html = open(HTML, encoding="utf-8").read()
    if STANDALONE_BASE not in html:
        sys.exit("FAIL: BASE constant not found in chat-ui.html")
    html = html.replace(STANDALONE_BASE, EMBEDDED_BASE)
    html = html.replace(TEST_HOOK, "")         # pointless in the served copy

    lines = html.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    body = "\n".join('            "' + c_escape(ln) + '\\n"' for ln in lines)
    return body + ");\n", len(html)


def splice(src, literal):
    start = src.find(START)
    if start < 0:
        sys.exit("FAIL: could not find the http_response call for GET /")
    body_start = start + len(START)
    end = src.find(END, body_start)
    if end < 0:
        sys.exit("FAIL: could not find the end of the embedded UI")
    return src[:body_start] + literal + src[end + len(');\n'):]


def main():
    check = "--check" in sys.argv[1:]
    literal, size = build_literal()
    src = open(SERVER, encoding="utf-8").read()
    new_src = splice(src, literal)

    if new_src == src:
        print(f"chat UI embedded copy is up to date ({size} bytes)")
        return 0
    if check:
        print("chat UI embedded copy is STALE - run tools/gen_chat_ui.py", file=sys.stderr)
        return 1
    open(SERVER, "w", encoding="utf-8").write(new_src)
    print(f"regenerated embedded chat UI in ds4_server.c ({size} bytes) - now run make")
    return 0


if __name__ == "__main__":
    sys.exit(main())
