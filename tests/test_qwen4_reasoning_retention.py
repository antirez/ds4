#!/usr/bin/env python3
"""Live Qwen reasoning-retention regression.  Requires a model GGUF.

  python3 tests/test_qwen4_reasoning_retention.py --model MODEL [--turns 10]

Four arms, one server run (plus one restart):

A  prompt shape      retention off must remove pre-window reasoning from the
                     rendered prompt and from prompt_tokens.
B  cache stability   turning retention off rewrites the stable prefix, so the
                     first request must miss once and the next must hit.
C  restart reuse     after a restart, a continuation of the retention-off loop
                     -- replayed without round-tripping the frontier reasoning,
                     which is the shape the checkpoint key is built for -- must
                     reuse the disk checkpoint.  This is the pass/fail signal for
                     gating prompt_preserves_reasoning on the retention switch.
D  symptom replay    harness-style notice-only user turns over several turns;
                     counts the "the user has sent / hasn't asked anything"
                     meta-commentary class in returned reasoning, on vs off.

Artifacts (server log, trace, per-request JSON) land in --out so a failure can
be read without re-running the model.
"""

import argparse
import json
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import urllib.request

TRACE_HEADERS = {
    "--- cache decision ---": "cache",
    "--- raw request json ---": "raw",
    "--- rendered prompt ---": "prompt",
    "--- generated text ---": "gen",
    "--- parsed message ---": "parsed",
    "--- client control tokens ---": "client_markers",
}

META = re.compile(
    r"the user (?:has sent|just sent|hasn'?t asked|haven'?t asked)|"
    r"system[- ]instructions? block|no actual task|only the system (?:prompt|instructions)",
    re.I,
)


def wait_ready(proc, base, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                return json.load(response)["data"][0]["id"]
        except (OSError, KeyError, json.JSONDecodeError):
            if time.monotonic() > deadline:
                raise RuntimeError("server did not become ready")
            time.sleep(0.5)


def stop(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def chat(base, model, messages, tools, preserve, effort, max_tokens):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": effort,
        "enable_thinking": effort != "none",
        "preserve_thinking": preserve,
        "chat_template_kwargs": {"preserve_thinking": preserve,
                                 "reasoning_effort": effort},
    }
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    result["_seconds"] = time.monotonic() - started
    return result


def usage_of(result):
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(usage.get("prompt_tokens") or 0), int(details.get("cached_tokens") or 0)


TOOL_NOTE = "archives total 462 segments"


def finished(result):
    """A truncated chain cannot be replayed: the KV holds an unclosed reasoning
    block, so re-rendering it as a closed one diverges at the very first appended
    token and every reuse check after that measures nothing."""
    return (result.get("choices") or [{}])[0].get("finish_reason") != "length"


def extend(messages, result, keep_reasoning=True):
    """Append the frontier turn `result` actually produced, then a tool output.

    This is how an agent loop grows.  keep_reasoning=False drops the reasoning
    body, which is what a client that does not round-trip thinking sends -- and
    the only shape a visible-transcript checkpoint key can match: the key is
    built as the render with an empty think block plus the trimmed content, so a
    replay that carries the reasoning text again never hashes to it."""
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    out = list(messages)
    out.append({"role": "assistant",
                "content": message.get("content") or "",
                "reasoning_content": (message.get("reasoning_content") or "")
                                     if keep_reasoning else "",
                **({"tool_calls": calls} if calls else {})})
    for call in calls:
        out.append({"role": "tool", "tool_call_id": (call or {}).get("id", "c"),
                    "content": TOOL_NOTE})
    return out


def miss_note(tracepath, cache, cached):
    """Why nothing was reused, as far as the artifacts can say.

    A disk hit is found by hashing the first key_bytes of the request render, so
    a render shorter than every key, or a render whose bytes at that length are
    not the stored key, both land on 'reused nothing' with no server-side trace.
    """
    if cached:
        return ""
    rendered = (parse_trace(tracepath) or [{}])[-1].get("prompt", "")
    render_bytes = len(rendered.encode())
    keys = kv_entries(cache)
    note = ("; render=" + str(render_bytes) + "B, keys on disk "
            + (", ".join(f"{name}={nbytes}B" for name, _t, nbytes in keys) or "none"))
    if keys and all(nbytes > render_bytes for _n, _t, nbytes in keys):
        note += " - every key is longer than the render, so none was a candidate"
    return note


KV_FIXED_HEADER = 48


def kv_entries(cache):
    """(name, tokens, key_bytes) per checkpoint file, read from its header.

    A disk hit is found by hashing the first key_bytes of the request render, so
    an entry whose key is longer than that render is never even opened.  Without
    this, a total miss looks like an evicted or corrupt cache."""
    entries = []
    for path in sorted(pathlib.Path(cache).glob("*.kv")):
        try:
            with path.open("rb") as handle:
                head = handle.read(KV_FIXED_HEADER + 4)
        except OSError:
            continue
        if len(head) < KV_FIXED_HEADER + 4:
            continue
        entries.append((path.name[:12],
                        int.from_bytes(head[8:12], "little"),
                        int.from_bytes(head[KV_FIXED_HEADER:KV_FIXED_HEADER + 4],
                                       "little")))
    return entries


def tool_defs():
    return [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up one short fact.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
        },
    }]


def build_history(turns, notice_only=False):
    """Synthetic agent session: every assistant turn carries its own sentinel
    reasoning body, so the rendered prompt shows which bodies survived."""
    msgs = [{
        "role": "system",
        "content": ("You are a coding agent. You MUST call the lookup tool once for every "
                    "step before saying anything; never answer a step without a tool call. "
                    "Keep each answer to one short sentence. " * 6),
    }]
    for i in range(turns):
        if notice_only and i % 2 == 1:
            msgs.append({"role": "user",
                         "content": "<system-notice>\nContinue the unfinished work.\n"
                                    "</system-notice>"})
        else:
            msgs.append({"role": "user", "content": f"Step {i}: check the archive size."})
        sentinel = f"chain-{i:02d} " + f"reasoning body number {i}. " * 14
        msgs.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": sentinel,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "lookup",
                                         "arguments": json.dumps({"query": f"size {i}"})}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f"archive {i} holds 42 segments"})
    # End on the tool output, the way an agent loop does.  A trailing user query
    # would close the retention window and drop the frontier reasoning too, so
    # the arms would measure the wrong thing.
    return msgs


def parse_trace(paths):
    """Header-anchored parse: only the headers ds4 actually writes.  Client text
    contains markdown rules that look like headers, so nothing looser is safe.
    Accepts one path or a list, so evidence from a restarted server (which opens
    its own trace file) stays in request order."""
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    requests = []
    for raw in paths:
        path = pathlib.Path(raw)
        if not path.exists():
            continue
        section = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("===== request "):
                    requests.append({"header": line.strip(), "prompt": "", "cache": ""})
                    section = None
                    continue
                if line.startswith("===== end request"):
                    section = None
                    continue
                stripped = line.rstrip("\n")
                if stripped in TRACE_HEADERS:
                    section = TRACE_HEADERS[stripped]
                    continue
                if section and requests:
                    requests[-1][section] = requests[-1].get(section, "") + line
    return requests


class Checks:
    """PASS/FAIL plus INCONCLUSIVE for arms the machine setup cannot settle."""

    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail="", inconclusive=False):
        state = "INCONCLUSIVE" if inconclusive else ("PASS" if ok else "FAIL")
        self.rows.append((name, ok, detail, inconclusive))
        print(f"  {state}  {name}" + (f"  {detail}" if detail else ""), flush=True)
        return bool(ok)

    @property
    def failed(self):
        return [row for row in self.rows if not row[1] and not row[3]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="ds4-server")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--symptom-turns", type=int, default=6)
    ap.add_argument("--ctx", default="8192")
    ap.add_argument("--kv-disk-space-mb", type=int, default=8192,
                    help="a checkpoint stores the allocated KV state, not just the used "
                         "prefix, so a small budget evicts the arm-C checkpoint before "
                         "the restart and C1 measures nothing")
    ap.add_argument("--effort", default="low", help="reasoning_effort for all arms")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--ready-timeout", type=int, default=900)
    ap.add_argument("--skip-symptom", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    out = (args.out or pathlib.Path(tempfile.mkdtemp(prefix="ds4-retention-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "kv"
    cache.mkdir(exist_ok=True)
    tracepaths = [out / "trace-1.log", out / "trace-2.log"]
    tracepath = tracepaths[0]
    logpath = out / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    for stale in tracepaths:
        if stale.exists():
            stale.unlink()
    cmd = [str((root / args.binary).resolve()), "-m", str(pathlib.Path(args.model).resolve()),
           "--ctx", args.ctx, "--port", str(port), "--prefill-chunk", "1024",
           "--kv-disk-dir", str(cache), "--kv-disk-space-mb",
           str(args.kv_disk_space_mb),
           "--kv-cache-min-tokens", "128", "--kv-cache-cold-max-tokens", "0",
           "--kv-cache-boundary-align-tokens", "128", "--trace", str(tracepath)]
    # Frontier turns must finish: a truncated chain cannot be replayed into the
    # next request, which would leave arms B and C undecidable.
    turn_max = max(args.max_tokens, 192)
    print("Artifacts:", out, flush=True)

    tools = tool_defs()
    history = build_history(args.turns)
    live_sentinel = f"chain-{args.turns - 1:02d}"
    stale_sentinel = "chain-00"
    checks = Checks()
    proc = None
    log = logpath.open("w")
    try:
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=log)
        model = wait_ready(proc, base, args.ready_timeout)

        # ---- A: prompt shape -------------------------------------------------
        print("A: prompt shape", flush=True)
        on = chat(base, model, history, tools, True, args.effort, turn_max)
        off = chat(base, model, history, tools, False, args.effort, turn_max)
        on_tokens, _ = usage_of(on)
        off_tokens, off_cached = usage_of(off)
        checks.add("A1 retention off shrinks the prompt",
                   off_tokens < on_tokens, f"on={on_tokens} off={off_tokens}")
        checks.add("A2 drop is roughly the pre-window reasoning",
                   on_tokens - off_tokens >= (args.turns - 2) * 40,
                   f"dropped={on_tokens - off_tokens} turns={args.turns}")
        rendered = parse_trace(tracepaths[0])
        prompts = [r.get("prompt", "") for r in rendered if r.get("prompt")]
        checks.add("A3 trace captured both renders", len(prompts) >= 2, f"n={len(prompts)}")
        on_prompt, off_prompt = prompts[-2], prompts[-1]
        checks.add("A4 on-arm replays the stale sentinel", stale_sentinel in on_prompt)
        checks.add("A5 off-arm drops the stale sentinel", stale_sentinel not in off_prompt)
        checks.add("A6 both arms keep the frontier reasoning",
                   live_sentinel in on_prompt and live_sentinel in off_prompt)

        # ---- B: one bust, then stable ---------------------------------------
        # Re-sending an equal or shorter prompt is not a reuse path on Qwen: the
        # live session already holds the answer and truncating back to the prompt
        # is the GLM-only rewind, so an identical repeat reports token-mismatch by
        # design.  What retention must not break is the normal growing loop.
        print("B: live reuse with the window", flush=True)
        # Grow the way an agent loop does: answer turn, then a tool output.  A
        # trailing user query would close the retention window here and drop the
        # frontier reasoning, which shortens the prompt below the live frontier
        # and makes it diverge from the checkpointed prefix in the middle.
        # The extension has to be the frontier turn the server actually
        # produced, and that turn has to have finished.  A truncated chain cannot
        # be replayed: the live KV holds an unclosed reasoning block, so
        # re-rendering it as a closed block diverges at the very first appended
        # token and every reuse check after that measures nothing.  A tool call is
        # a finished turn, and it is the shape a real agent loop replays, so only
        # finish=length blocks the arm.
        replayable = finished(off)
        grown = extend(history, off)
        grown_prompt = ""
        nxt = {}
        nxt_tokens = nxt_cached = 0
        if replayable:
            nxt = chat(base, model, grown, tools, False, args.effort, turn_max)
            nxt_tokens, nxt_cached = usage_of(nxt)
            grown_prompt = parse_trace(tracepaths[0])[-1].get("prompt", "")
        reason = ("" if replayable else
                  " - frontier turn was not replayable "
                  f"(finish={off['choices'][0].get('finish_reason')})")
        checks.add("B0 the arm really grew the prompt", replayable and nxt_tokens > off_tokens,
                   f"off={off_tokens} grown={nxt_tokens}{reason}", inconclusive=not replayable)
        checks.add("B1 growing retention-off loop reuses the live prefix",
                   replayable and nxt_cached > 0,
                   f"cached={nxt_cached}/{nxt_tokens} (off arm itself started "
                   f"cold: {off_cached}){reason}", inconclusive=not replayable)
        checks.add("B2 the reused prefix spans the rewritten prompt",
                   replayable and nxt_cached >= off_tokens - 64,
                   f"cached={nxt_cached} off={off_tokens}{reason}", inconclusive=not replayable)
        checks.add("B3 the grown prompt still holds no pre-window reasoning",
                   (not replayable) or (stale_sentinel not in grown_prompt),
                   f"len={len(grown_prompt)}{reason}", inconclusive=not replayable)

        # ---- C: restart reuse -----------------------------------------------
        print("C: restart reuse", flush=True)
        stop(proc)
        cmd[cmd.index("--trace") + 1] = str(tracepaths[1])
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=log)
        model = wait_ready(proc, base, args.ready_timeout)
        log.flush()
        logtext = logpath.read_text(errors="replace")
        budget = re.search(r"KV disk cache .*?budget=(\d+) MiB", logtext)
        thrashed = "disk-cache-full" in logtext
        checks.add("C0 disk budget kept the checkpoint until the restart",
                   not thrashed,
                   f"server reported budget={budget.group(1) if budget else '?'} MiB "
                   f"(asked {args.kv_disk_space_mb})",
                   inconclusive=thrashed)
        # Two things have to be true for this replay to be able to hit, and the
        # first run of this arm failed on both:
        #
        # 1. The request must be longer than the checkpoint's key.  The shutdown
        #    store holds the state the grown request *produced*, keyed by the
        #    transcript that includes the answer it generated, so replaying the
        #    grown request itself presents a render a couple hundred bytes
        #    shorter than the key it is looking for.  find_text_prefix() only
        #    considers keys that fit inside the render, skips everything, opens
        #    no file and logs nothing.
        # 2. The replay must render that frontier turn the way the key renders
        #    it.  build_thinking_visible_text() says so in its own comment: the
        #    key is the render plus an empty think block and the trimmed content.
        #    A client that round-trips thinking replays the reasoning text
        #    instead, and with retention off the renderer keeps the frontier
        #    body, so the hashes part company at the first byte of that body.
        #    That is the shape this arm now does not use: the same continuation
        #    with reasoning_content resent measured cached=0 of 1252 on a fresh
        #    server while reusing the live prefix fine (B1) -- live reuse goes
        #    through the exact-token-prefix tier, which has no such key.
        #
        # So: replay the continuation the way a client that does not round-trip
        # thinking sends it.  If this misses too, the visible-transcript key is
        # unreachable for retention-off sessions and the key kind needs to follow
        # the switch rather than the request flag.
        replayable_after = replayable and finished(nxt)
        after = extend(grown, nxt, keep_reasoning=False)
        cold_tokens = cold_cached = 0
        if replayable_after:
            cold = chat(base, model, after, tools, False, args.effort, turn_max)
            cold_tokens, cold_cached = usage_of(cold)
        detail = (f"cached={cold_cached}/{cold_tokens} of {len(after)} msgs"
                  + miss_note(tracepaths[1], cache, cold_cached))
        # A hit is a hit: the checkpoint key question is settled by the reuse
        # itself.  Nothing reused because the budget evicted the checkpoint, or
        # because there was no replayable frontier to extend, is undecidable
        # rather than a failure.
        undecidable = (not replayable_after) or (thrashed and cold_cached == 0)
        if not replayable_after:
            detail += " - undecidable, no replayable frontier to extend"
        elif thrashed and cold_cached == 0:
            detail += " - undecidable, checkpoint was evicted first"
        checks.add("C1 retention-off continuation reuses the disk checkpoint after restart",
                   replayable_after and cold_cached > 0, detail, inconclusive=undecidable)

        # ---- D: symptom replay ---------------------------------------------
        if not args.skip_symptom:
            print("D: symptom replay", flush=True)
            counts = {}
            steps = 3
            for preserve in (True, False):
                msgs = build_history(args.symptom_turns, notice_only=True)
                hits = calls = 0
                for step in range(steps):
                    result = chat(base, model, msgs, tools, preserve, args.effort, turn_max)
                    message = result["choices"][0]["message"]
                    reasoning = message.get("reasoning_content") or ""
                    hits += len(META.findall(reasoning))
                    got_calls = message.get("tool_calls") or []
                    calls += 1 if (got_calls or (message.get("content") or "").strip()) else 0
                    msgs.append({"role": "assistant",
                                 "content": message.get("content") or "",
                                 "reasoning_content": reasoning,
                                 **({"tool_calls": got_calls} if got_calls else {})})
                    for call in got_calls:
                        msgs.append({"role": "tool",
                                     "tool_call_id": (call or {}).get("id", "c"),
                                     "content": f"notice-loop answer {step}"})
                    msgs.append({"role": "user",
                                 "content": "<system-notice>\nContinue.\n</system-notice>"})
                counts[str(preserve)] = (hits, calls)
                print(f"  preserve={preserve} meta={hits} answered_turns={calls}", flush=True)
            replayed = counts["True"][0] or counts["False"][0]
            checks.add("D1 meta-commentary drops with the window",
                       counts["False"][0] <= counts["True"][0],
                       f"on={counts['True'][0]} off={counts['False'][0]}"
                       + ("" if replayed else " - neither arm replayed the opener"),
                       inconclusive=not replayed)
            checks.add("D2 tool loop still answers every turn",
                       counts["True"][1] == steps and counts["False"][1] == steps,
                       f"expected={steps} on={counts['True'][1]} off={counts['False'][1]}")

        log.flush()
    finally:
        stop(proc)
        log.close()

    text = logpath.read_text(errors="replace")
    checks.add("X1 no KV staging corruption in the server log",
               "KV payload staging failed" not in text and
               "session has no valid checkpoint to stage" not in text)

    print("\nsummary", flush=True)
    for name, ok, detail, inconclusive in checks.rows:
        state = "INCONCLUSIVE" if inconclusive else ("PASS" if ok else "FAIL")
        print(f"  {state}  {name}" + (f"  {detail}" if detail else ""))
    failed = checks.failed
    inconclusive = [row for row in checks.rows if row[3]]
    print(f"\nchecks={len(checks.rows)} failed={len(failed)} "
          f"inconclusive={len(inconclusive)} artifacts={out}")
    print("Interpretation: C1 decides whether to gate prompt_preserves_reasoning "
          "on the retention switch. A pass means restart reuse is fine as-is; a fail "
          "means the checkpoint key must follow the switch -- but read the printed key "
          "lengths first, because a render shorter than every key on disk reuses "
          "nothing for reasons that have nothing to do with the switch.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
