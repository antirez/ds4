#!/usr/bin/env python3
"""Claude-Code-shaped requests against a running Inkling server.  usage: claude_compat.py [--port 8000] CTX   (CTX = the server's --ctx)
checks: overflow wording, media/tool_reference in tool results (no 400), count_tokens, a top-level image is still refused."""
import http.client, json, sys
port = 8000
if "--port" in sys.argv:
    i = sys.argv.index("--port"); port = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
ctx = int(sys.argv[1])
def post(path, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    r = c.getresponse(); d = r.read()
    try: return r.status, json.loads(d)
    except Exception: return r.status, d[:200]
fails = 0
def check(name, cond, info=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name, info if not cond else "")
    fails += not cond
png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
def with_result(block):
    return {"model": "inkling-2-mini", "max_tokens": 40, "temperature": 0, "thinking": {"type": "disabled"},
            "tools": [{"name": "Read", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "Read /tmp/x and tell me what it is."},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/tmp/x"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [block]}]}]}
for name, blk in [("image", {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}),
                  ("pdf", {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"}}),
                  ("tool_reference", {"type": "tool_reference", "tool_name": "WebFetch"})]:
    st, j = post("/v1/messages", with_result(blk))
    check("%s inside a tool_result: 200" % name, st == 200 and j.get("type") == "message", (st, str(j)[:200]))
    # and the same conversation with one more turn still works (no sticky error)
st, j = post("/v1/messages", {"model": "inkling-2-mini", "max_tokens": 10, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}]}]})
check("a user image is still refused (no vision)", st == 400, (st, str(j)[:200]))
st, j = post("/v1/messages/count_tokens", with_result({"type": "text", "text": "hello"}))
check("count_tokens", st == 200 and isinstance(j, dict) and j.get("input_tokens", 0) > 20, (st, j))
words = " ".join(["alpha beta gamma delta"] * (ctx // 3))
st, j = post("/v1/messages", {"model": "inkling-2-mini", "max_tokens": 10, "messages": [{"role": "user", "content": words}]})
msg = (j.get("error") or {}).get("message", "") if isinstance(j, dict) else ""
check("overflow says 'prompt is too long: N tokens > M maximum'", st == 400 and msg.startswith("prompt is too long: ") and "tokens > %d maximum" % ctx in msg, (st, msg))
print(msg)
sys.exit(1 if fails else 0)
