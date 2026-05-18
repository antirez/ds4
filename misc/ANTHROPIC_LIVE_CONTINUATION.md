# Anthropic Live Tool Continuation

## Lessons To Reuse

From `misc/SIMPLIFIED_NO_CANONICALIZATION_MODEL.md`:

- The sampled KV state is the most faithful state DS4 has.
- Client-visible protocol objects should select that sampled state; they should
  not force DS4 to rewrite it unless there is no live or persisted match.
- Tool calls are a natural instance of this rule:

```text
tool_use_id -> exact sampled DSML/KV frontier
```

From `misc/RESPONSE_API.md`:

- Protocol IDs exist to avoid proving that an already-known prefix tokenizes the
  same way again.
- Tool-result requests bound to live call IDs should append only the tool-result
  suffix to the live KV.
- Prefix matching and disk checkpoints remain fallbacks for restart, stateless
  replay, edits, or client/session switches.
- Unknown IDs are not silently repairable if the request does not replay enough
  prior history.

## Anthropic Contract

Anthropic tool loops use `tool_use.id` on assistant output and
`tool_result.tool_use_id` on the following user turn. For a local server with one
live KV session, that ID is enough to prove that the next request is continuing
the just-sampled assistant tool-call frontier, provided the live session is still
at the remembered token position.

The fast path is:

1. The model samples an Anthropic tool call.
2. DS4 assigns/remembers the `toolu_...` IDs and keeps the live KV exactly as
   sampled, including hidden thinking and raw DSML.
3. The next `/v1/messages` request contains `tool_result` blocks with those IDs.
4. DS4 verifies that the live token position and the ID set match.
5. DS4 tokenizes only:

```text
<｜end▁of▁sentence｜><｜User｜><tool_result>...<｜Assistant｜><think-or-/think>
```

and appends it to the live sampled prefix.

This avoids checkpoint canonicalization after the tool call and avoids relying
on a replayed JSON tool block to match the sampled DSML byte-for-byte.

## Fallbacks

If the live Anthropic ID binding is gone, DS4 should use normal replay:

- exact token prefix,
- live rendered-text prefix with suffix retokenization,
- disk string-prefix KV checkpoints,
- cold prefill.

This is valid when the request replays the assistant `tool_use` block before the
`tool_result`, because exact DSML tool memory can restore the sampled tool-call
bytes from the tool ID. If the request contains only a `tool_result` for an
unknown live ID and does not replay the prior assistant call, DS4 should return a
clear 400 asking the client to replay the full history.

## What Not To Do

- Do not add a RAM snapshot cache to paper over live-continuation mismatches.
  The specific Anthropic issue is not lack of a rewind checkpoint; it is that
  the protocol already identifies the live frontier by tool ID.
- Do not canonicalize a live tool-call checkpoint just to match the client JSON.
  If DS4 has raw sampled DSML and a matching live ID, the live state should win.
- Do not reuse already-tokenized suffix tokens from the full rendered request at
  a string boundary. Tokenize the suffix after the cache/continuation decision.

## QA Checklist

- Unit test: Anthropic `tool_result.tool_use_id` is parsed and collected.
- Unit test: Anthropic live suffix renders only EOS, tool result, and assistant
  prefix, not the previous assistant tool call.
- Integration test: a matching live Anthropic ID builds an effective prompt
  whose cached prefix length is the live token count.
- Unit test: a tool-result-only request with an unknown ID is rejected.
- Integration test: Claude Code can execute a multi-turn tool session without
  post-tool checkpoint canonicalization or needless re-prefill.
- Regression check: Responses live continuation still works.

## Current Implementation Notes

- The speculative RAM KV restart cache was removed. It was solving the wrong
  problem for Anthropic tool loops: the protocol already gives DS4 a precise
  live continuation key in `tool_use_id`.
- `chat_msg` now keeps all tool-result IDs seen in a message, not just the first
  or last one. Anthropic can return multiple `tool_result` blocks in one user
  message, and a partial ID set would make live continuation unsafe.
- Anthropic live state stores only the sampled token frontier and generated
  tool-call IDs. Responses still also stores visible replay text because that
  protocol has a visible-prefix continuation mode.
- Anthropic `tool_result` requests are accepted in two cases:
  - the IDs match the current live Anthropic tool-call frontier;
  - the request replays the prior assistant `tool_use` blocks, so normal exact
    DSML replay and prefix matching can reconstruct the prompt.
- A `tool_result` request with neither live state nor replayed prior tool calls
  returns HTTP 400 and asks for full messages.

## QA Log

- `make ds4_test`: pass.
- `./ds4_test --server`: pass.
- `make ds4-server`: pass.
- Claude Code through `~/bin/claude-ds4` against `/v1/messages`: pass.
  - Prompt forced two Bash tool calls in one assistant turn.
  - Server logged:

```text
anthropic live continuation match=tool-output-ids ids=2 cached=24002 prompt=24041
chat ctx=24002..24041:39 TOOLS prompt done 0.873s
```

  - The trace contained no post-tool `tool checkpoint canonicalization` event.
- Negative HTTP check: a `tool_result` with unknown `tool_use_id` and no replayed
  prior assistant call returned:

```text
HTTP/1.1 400 Bad Request
unknown Anthropic tool_result tool_use_id toolu_unknown; replay full messages
```
