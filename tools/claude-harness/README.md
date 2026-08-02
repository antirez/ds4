# claude-harness — Claude Code on the pulsar stack

Claude Code as the coding harness, with pulsar-server as the main model and a
second box handling the haiku-class background traffic.

```
Claude Code ──all traffic──> pulsar-model-router :8100
                                  │
                 model ~ /haiku/i ├──> ellie.defense.lan:8000  (small/fast)
                 everything else  └──> 127.0.0.1:8000          (pulsar-server)
```

## Why a router

Claude Code sends **all** requests — the main conversation *and* its
high-frequency haiku-class utility calls (summaries, titles, bash-command
descriptions, some subagents) — to one `ANTHROPIC_BASE_URL`. Per-model
endpoints were declined upstream (anthropics/claude-code#25146, #24160). The
only distinguishing bit is the `model` field in the body, so the split has to
happen in front of the servers.

The router is a raw byte relay: it parses the body once (read-only) to find
`model`, then forwards request and response verbatim — no re-serialization, so
the byte-stable replays that pulsar's prefix cache keys on are untouched. This
is also why a general-purpose gateway (LiteLLM etc.) is the wrong tool here:
both backends already speak the Anthropic protocol natively, and a translating
middlebox that normalizes messages (thinking blocks, tool-call ids,
cache_control) would break live-continuation matching.

Keeping the small/fast traffic off pulsar matters beyond throughput: every
request admits a session against the bank pool, and a stream of throwaway
haiku calls would churn the LRU pool and can evict the big cached session —
paying a full re-prefill on the next real turn.

## The 1M context window

Claude Code sizes its context window (and the auto-compact trigger) from the
requested model name's `[1m]` suffix. There is no context-window override env
var. Verified by header dump (2026-08-01): the suffix works on **custom**
names — requesting `deepseek-v4-flash[1m]` sends `model=deepseek-v4-flash`
on the wire *with* the `context-1m-2025-08-07` beta (1M accounting), while
the bare name omits it (Claude Code then assumes a small default window).
pulsar-server never validates the client's model string, so any name works
server-side; keep the `[1m]` suffix on whatever display name you pick.

Known caveat: Claude Code strips the `[1m]` suffix when resolving subagent
models (anthropics/claude-code#45169), so subagents assume a 200K window and
compact earlier. The main loop keeps 1M.

## Usage

On sparky (or wherever, adjust flags):

```bash
tools/claude-harness/pulsar-model-router.py \
    --listen 0.0.0.0:8100 \
    --main 127.0.0.1:8000 \
    --small ellie.defense.lan:8000
```

Then from any machine:

```bash
tools/claude-harness/claude-pulsar            # env-var wrapper, execs claude
```

or point `CLAUDE_PULSAR_URL` elsewhere / pass normal `claude` args through.

To skip the router entirely (single backend, small-fast lands on pulsar too):
`CLAUDE_PULSAR_URL=http://sparky:8000 claude-pulsar`.

## Gaps / notes

- `/v1/messages/count_tokens` is implemented server-side (same render +
  tokenize path as `/v1/messages`, no job/bank involvement, no context-length
  rejection), so Claude Code's pre-flight sizing gets exact counts. Response
  `usage` was already exact, including cache read/creation splits.
- Claude Code sends `cache_control` blocks and `anthropic-beta` headers
  (context-1m etc.); both backends ignore them harmlessly.
- The router treats non-POST requests (e.g. `GET /v1/models`) as main-bound.
