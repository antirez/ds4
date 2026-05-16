# Responses API Continuation Plan

This note tracks the design for fixing `/v1/responses` tool-call continuation
and the KV-cache behavior around it.

## Problem

The current implementation handles `/v1/responses` too much like a stateless
chat-completions replay API. After a tool call, it tries to canonicalize the
live KV checkpoint so it matches the next rendered prompt byte-for-byte.

That is wrong for the live Responses protocol path.

In a live Responses tool loop, the next request is not an unrelated prompt. It
is a continuation bound to the previous model output by tool-call ids. The live
KV already contains the true assistant turn, including hidden reasoning that
the client may not replay. Rebuilding the session to match only the visible
client transcript is both slow and less faithful to the state the model actually
produced.

Observed symptom:

```text
tool checkpoint canonicalization needs rebuild ... common=16846 live=16994 canonical=16937
tool checkpoint canonicalized ... via=rebuild
```

This caused a long pause because DS4 rebuilt roughly the whole context before
returning the streamed tool-call response tail.

## Protocol Model

There are two useful id classes:

- `previous_response_id` / `conversation`: response-level server-side state. DS4
  does not currently persist this. If a non-null value arrives, return an error
  asking the client to replay full input.
- `call_id`: tool-call binding. A `function_call_output` or hosted-tool output
  with this id is the continuation of the previous assistant tool call.

The server should distinguish two modes:

- Live continuation: the request contains tool outputs for tool-call ids still
  known by the live server state. Continue from the live KV state and append the
  new tool-result suffix.
- Stateless replay: the request includes the full history. Render and match the
  best prefix, using exact DSML replay from tool-memory when possible, just like
  chat-completions style replay.

Unknown tool-output ids are only valid if the request also includes the matching
prior function-call item in the replayed history. If neither live state nor full
history can explain the id, return an error.

Core mental model:

- The Responses API is designed so a server can avoid matching an already-known
  prefix. If the request is tied to live server state by response/conversation
  state or by immediately returning tool outputs for live call ids, DS4 should
  continue from the live KV instead of proving that the client-visible replay
  tokenizes to the same prefix.
- Prefix matching is only the fallback for stateless replay, cold start,
  server restart, branch/edit, or multi-client cases where the server no longer
  has the relevant live state.
- The fallback should reuse the techniques learned for chat continuation:
  exact token-prefix match first, then rendered string-prefix match with
  suffix retokenization, then disk string-prefix checkpoints. The adaptation
  for Responses is that reasoning/tool-call stateless replay must include the
  reasoning state the protocol requires, not just the visible transcript.

## Official OpenAI Docs Findings

Sources checked:

- Conversation state:
  `https://developers.openai.com/api/docs/guides/conversation-state`
- Function calling:
  `https://developers.openai.com/api/docs/guides/function-calling`
- Responses API reference:
  `https://platform.openai.com/docs/api-reference/responses`

Findings that affect the implementation:

- The Responses API has two explicit server-side state mechanisms:
  `previous_response_id` and `conversation`.
  - `previous_response_id` chains a new response to a stored prior response.
  - `conversation` prepends persisted conversation items and then appends new
    input/output items after the response completes.
  - They are mutually exclusive in the API reference.
- Stateless/manual Responses state is also valid, but the client must preserve
  the model output items. The function-calling guide shows:
  - Start with an `input` list.
  - Call `responses.create()`.
  - Append `response.output` to the same input list.
  - Append `function_call_output` items with the matching `call_id`.
  - Send the resulting input list back.
- For reasoning models, the function-calling guide explicitly says that any
  reasoning items returned in model responses with tool calls must also be
  passed back with tool-call outputs.
- The Responses API reference exposes `include:
  ["reasoning.encrypted_content"]`, described as enabling reasoning items to be
  used in stateless multi-turn conversations when `store=false` or ZDR applies.

Implication for DS4:

- A request containing only a function-call output is not enough to recreate
  stateless reasoning context unless DS4 still has live state for that call id.
- A full stateless replay should include the prior function/custom tool call
  item and, for reasoning/tool-call turns, the reasoning item or an equivalent
  opaque reasoning-state item.
- DS4 currently does not implement durable `previous_response_id` or
  `conversation` state, so non-null values should remain rejected unless we add
  a real response/conversation store.
- The current Codex-style `store=false` / `include=[]` flow can still work as a
  live local server fast path, but if the live state is gone DS4 should reject
  and ask for a full replay that includes the missing reasoning state, rather
  than silently rebuilding a prompt with hidden reasoning removed.

## Current Code Facts

- Exact DSML replay via `tool_memory` / rax is still valid and working. It fixes
  the visible tool-call block by replaying the sampled DSML for known call ids.
- The bad path is `canonicalize_tool_checkpoint()` for Responses tool calls when
  reasoning summaries are not emitted. It drops hidden reasoning and can trigger
  full rebuilds.
- `live_text_prefix_prompt()` is still useful for ordinary replay/cache matching,
  but it is not enough for the live Responses continuation case because the
  client-visible text can omit hidden reasoning that exists in the live KV.
- `ds4_session_save_snapshot()` / `ds4_session_load_snapshot()` exist, but they
  should not be the primary continuation mechanism for this protocol path.

## Implementation Plan

1. Track live Responses continuation state.
   - Remember the latest Responses assistant tool-call turn after generation.
   - Store the generated call ids and enough rendered visible text to recognize
     the next request.
   - Keep the live KV as authoritative; do not canonicalize away hidden
     reasoning for this path.
   - Treat this as an optimization for the current live single-server session,
     not as a substitute for `previous_response_id` / `conversation`.
   - Prefer this path for Codex/pi tool loops when integration tests show those
     clients send tool outputs that are directly bound to the previous live
     call ids.

2. Parse Responses tool outputs with validation.
   - Collect function/custom/hosted tool-output ids during `parse_responses_input()`.
   - Collect function/custom/hosted call ids present in the replayed history.
   - For reasoning-mode tool-call turns, distinguish a true full replay from a
     visible-only replay. A matching prior call id without the reasoning item is
     enough to render visible history, but not enough to reconstruct hidden
     model state if live continuation is unavailable.
   - Reject tool-output ids that are neither known live ids nor present as prior
     calls in the same request history.

3. Add a live continuation fast path before normal cache matching.
   - If the request is a Responses request and its first new semantic item is a
     tool output for the remembered live call ids, build only the suffix that
     must be appended to the current live KV.
   - Tokenize that suffix independently from the live token prefix, as the
     existing text-prefix path does, to avoid BPE boundary mistakes.
   - Set `prompt_for_sync` to `live_tokens + suffix_tokens`, with
     `cached = live_tokens->len`.

4. Keep stateless replay behavior.
   - If the request is not a live continuation, render the full prompt and use
     the existing token-prefix, live text-prefix, and disk text-prefix cache
     paths.
   - Exact DSML tool replay remains available to improve matching.
   - If the request is a stateless reasoning/tool-call replay without reasoning
     items or equivalent opaque reasoning state, reject instead of pretending the
     visible transcript is complete.

5. Remove invalid Responses canonicalization.
   - Do not call `canonicalize_tool_checkpoint()` for Responses live tool-call
     continuations.
   - Remove the Responses-specific reasoning-dropping behavior in
     `build_tool_checkpoint_suffix()`.
   - Keep or narrow toolless thinking canonicalization only where the next
     request is truly a stateless replay and the live hidden reasoning cannot be
     continued by protocol ids.

6. Improve errors.
   - Non-null `previous_response_id` / `conversation`: return a clear 400 error
     unless DS4 later implements persistent response state.
   - Unknown tool-output id without a matching prior function call in the same
     request: return a clear 400 error explaining that DS4 needs full history.
   - Reasoning/tool-call replay that lacks reasoning state and is not live:
     return a clear 400 explaining that DS4 needs either live continuation or a
     full Responses replay with reasoning items / encrypted reasoning content.

## Tests

Automated tests:

- Responses live tool continuation keeps hidden reasoning in live KV and appends
  only the tool-result suffix.
- Responses live continuation does not call tool checkpoint canonicalization.
- Unknown tool-output id with no prior function call is rejected.
- Unknown tool-output id with a prior function call in replayed history is
  accepted as stateless replay only when the replay has enough reasoning state
  for the selected mode.
- Reasoning/tool-call stateless replay without reasoning state is rejected if
  live continuation is unavailable.
- Normal chat-completions exact DSML replay still works.
- Disk KV tool-map restore still works for stateless replay.
- BPE boundary handling remains correct for live-prefix plus text suffix.

Live tests:

- Codex multi-turn session with repeated tool calls.
  - Expected: after each tool call, next request reports a memory-token
    continuation or equivalent live continuation, not a rebuild.
  - Expected: no long pause after `finish=tool_calls`.
  - Verify the actual client behavior: whether Codex sends `previous_response_id`,
    full `response.output`, function-call outputs only, reasoning items, or some
    mixture.
- Pi agent multi-turn session with tools.
  - Expected: tool calls execute and the agent recovers normally from tool
    outputs.
  - Expected: no unexpected cache misses.
  - Verify the actual client behavior independently from Codex. Do not assume
    both clients use the same Responses replay style.
- Stateless replay request with full history and known call ids.
  - Expected: exact DSML replay via rax and best-prefix cache matching.
- Bad continuation request with only a tool output for an unknown id.
  - Expected: HTTP 400 with a direct "replay full history" style message.

## Current Progress

- Confirmed from trace that the pause is synchronous checkpoint
  canonicalization and rebuild after a parsed tool call.
- Confirmed rax exact DSML replay is not the failing part:
  `tool_replay: mem=4 disk=0 canonical=0 missing_ids=0`.
- Confirmed the current Responses parser rejects non-null
  `previous_response_id` / `conversation`, which is the right behavior until
  server-side response persistence exists.
- Checked official OpenAI documentation. The plan was updated to account for
  the documented requirement that reasoning-model tool-call loops preserve
  reasoning items in stateless manual replay.
- Implemented live Responses continuation state and removed Responses tool-call
  checkpoint canonicalization from the main path.
- Added normal server logs for accepted live Responses continuations:
  `responses live continuation match=... cached=... prompt=...`.
  Tool-output continuations now also report the matched id count, for example
  `match=visible-prefix ids=1`.
- Added Responses-visible disk checkpoint keys. The payload is still the exact
  live KV with hidden reasoning, but the file name / lookup text is the visible
  transcript the client can replay. This is required when switching between
  Codex, pi, and synthetic clients, because only one live KV session can be
  resident at a time.

## QA Log

All tests in this section were run on the local M3 Max server with Metal,
`--ctx 65536`, and `/v1/responses` clients configured for
`deepseek-v4-flash`.

Automated checks:

- `make ds4_test && ./ds4_test --server && make ds4-server`: pass.
- Added unit coverage for unknown tool-output ids, reasoning-required
  stateless replay, Responses live-tail rendering, and final-answer visible
  suffix handling.

Official protocol checks:

- `previous_response_id != null`: returns HTTP 400 with
  `previous_response_id is not supported; replay full input instead`.
- `conversation != null`: returns HTTP 400 with
  `conversation is not supported; replay full input instead`.
- Tool output for an unknown `call_id`: returns HTTP 400 with
  `unknown tool output call_id ...; replay full Responses history`.
- Stateless replay with a prior tool call but no reasoning item: returns HTTP
  400 with `Responses replay is missing reasoning state...`.
- Stateless replay with a prior tool call and an explicit reasoning item:
  returns HTTP 200 and generates from the replayed history.

Codex live behavior:

- Codex sends `store=false`, `include=[]`, `prompt_cache_key=<session id>`, and
  full visible replay.
- Exact token matching fails at hidden thinking, as expected:
  `live_prompt_common` stops where live KV has hidden reasoning and the visible
  replay has `</think>`.
- Live visible continuation now catches this path:
  `cache_source: responses-visible`, with short suffix prefill after tool
  outputs and after normal resumed user turns.
- Repeated Codex turns completed with tool execution and no
  `canonicalization needs rebuild` pause.

Pi live behavior:

- Pi uses a custom local provider with `api: openai-responses`.
- Pi sends `reasoning.summary=auto` and `include:
  ["reasoning.encrypted_content"]`.
- Pi replays reasoning summaries for assistant tool-call turns, but not for
  final assistant text answers. The visible-prefix state must therefore include
  tool-call reasoning summaries when present, but omit final-answer hidden
  reasoning.
- Pi first turn, tool result, resumed user turn, and resumed tool result all
  completed. Accepted live continuations print:
  `responses live continuation match=visible-prefix ...`; tool-output turns
  print `ids=1` in the normal server log.

Interleaved session switching:

- Sequence tested: Codex turn, pi turn, Codex resume, pi resume, Codex resume,
  plus synthetic curl traffic between agent turns.
- On switching clients, the live KV miss is expected and logged in orange.
- Before replacing live state, DS4 stores the outgoing Responses session as
  `key=responses-visible`.
- Returning to Codex after pi loaded the Codex checkpoint from disk:
  `kv cache hit text tokens=11772 ...`, then prefills only the new suffix.
- Returning to pi after Codex loaded the pi checkpoint from disk:
  `kv cache hit text tokens=1654 ...`, then prefills only the new suffix.
- Returning to Codex after synthetic curl traffic again loaded the latest Codex
  checkpoint from disk:
  `kv cache hit text tokens=12211 ...`.
- The earlier failure mode was reproduced before the fix: Codex resume after
  pi received `Responses replay is missing reasoning state...` because the
  guard ran before disk recovery and the disk key was raw hidden text. The fix
  is to key Responses checkpoints by visible transcript and accept only
  `KV_EXT_RESPONSES_VISIBLE` disk hits for missing-reasoning replay.

Remaining risks / follow-up:

- DS4 still does not implement durable `previous_response_id` or
  `conversation`; rejecting non-null values remains intentional.
- `reasoning.encrypted_content` is not decrypted. It is useful for real OpenAI
  stateless clients, but DS4 currently relies on live state or visible-key disk
  checkpoints for hidden reasoning recovery.
- The tool-output-only continuation path is unit-tested, but the observed Codex
  and pi clients both used full visible replay in their normal runs.
- Tool-output-only continuation was also exercised directly with curl:
  first request produced `call_3c4891795bf43aca507f4a987707b259`, second
  request sent only `function_call_output` for that id, and the server accepted
  it as `responses live continuation match=tool-output-ids ids=1`.
- Code review follow-up tightened a race-shaped edge case: if a request contains
  only tool outputs, it is marked as requiring the live call-id state. If that
  state no longer matches by worker execution time, DS4 now returns a 400 asking
  for full input instead of cold-prefilling a prompt that starts with a naked
  tool result. Full visible replays in thinking mode also remain marked as
  reasoning-sensitive even if live state existed during parse, because the live
  KV can be replaced before execution.
- Disk cache hit logs now include the key kind:
  `key=responses-visible` for visible-transcript keys that restore hidden KV,
  and `key=token-text` for ordinary rendered-token text keys.
- `make ds4_test && ./ds4_test --server && make ds4-server`: pass after the
  stricter tool-output-only check and log-key change.
- A separate `/tmp/ds4_test_asan` build with
  `-fsanitize=address,undefined` also passes `--server`. Leak detection is not
  supported by Apple ASan on this platform, so the run covered address/UB
  checks but not leak reporting.

Restart / disk recovery test:

- Stopped the old server and started a fresh `ds4-server` with the existing
  `/tmp/kvcache-responses-switch3` directory.
- Resumed the existing Codex session
  `019e23a9-821c-7e40-8c00-122646d7fda7` after process restart.
- Expected behavior occurred immediately:
  `kv cache hit text tokens=12367 ... key=responses-visible`, followed by a
  short suffix prefill and a live Responses tool-output continuation with
  `ids=1`.
- After unrelated pi, curl, and opencode traffic replaced live KV, resuming
  Codex again hit disk with `key=responses-visible` at the later checkpoint and
  then continued the tool result live. This confirms the cross-client/session
  path is not only an in-process memory shortcut.
- Shutdown persistence was also tested: stopping the server logged
  `reason=shutdown key=responses-visible` for the latest Codex live state.
  After a fresh restart, resuming the same Codex session hit that file:
  `kv cache hit text tokens=12897 ... key=responses-visible`, then completed
  the next tool call and tool-result turn normally.

Additional live clients:

- Pi was resumed from the saved session file and completed a real bash tool
  turn. In this invocation pi sent a shorter prompt for the new turn, so DS4
  cold-prefilled that request and then accepted the tool result through
  `responses live continuation match=visible-prefix ids=1`.
- opencode was run for two real shell-tool turns against the configured
  `ds4/deepseek-v4-flash` provider. This path uses the OpenAI-compatible chat
  API rather than `/v1/responses`, and it completed both the initial tool call
  and the resumed tool call. That is useful shared-path regression coverage:
  chat/completions tool checkpoint canonicalization still works while Responses
  skips it.

More protocol negatives:

- Non-null `previous_response_id`: still HTTP 400.
- Non-null `conversation`: still HTTP 400.
- Tool output with an unknown `call_id`: still HTTP 400.
- Stateless replay of a prior tool call plus tool output, without a reasoning
  item in thinking mode: HTTP 400.
- The same stateless replay shape with an explicit `reasoning` item: HTTP 200.
- The short tool-output-only happy path was re-tested after the stricter guard:
  first curl request generated a `bash` function_call, second request sent only
  the matching `function_call_output`, and DS4 logged
  `responses live continuation match=tool-output-ids ids=1`.
- The stale-live-state failure path was also tested with real queued requests:
  create a live `bash` function_call, start a long unrelated request that
  replaces the live KV, then submit the old tool-output-only request while the
  long request is still running. When the queued tool-output-only request
  executed, DS4 returned HTTP 400 with
  `Responses tool output requires live call state; replay full input instead`.
  This verifies the new guard prevents cold-prefilling an orphan tool result.

Post-merge stress pass, 2026-05-14:

- Merged the `responses-api` branch into `main` while preserving its branch
  commits, then re-applied the exact sampled DSML tool-checkpoint gate on top of
  the merged Responses live-continuation path.
- Replaced eager tool-less thinking checkpoint rebuilds with a visible-transcript
  live binding. A no-tool thinking answer now remembers the visible transcript
  that clients replay, while keeping the sampled hidden reasoning in KV. The
  next chat/completions or Anthropic request can continue with
  `cache_source: thinking-visible` if its prompt extends that visible prefix.
- Unit/build checks after the change:
  `make ds4_test`, `./ds4_test --server`, and `make ds4-server` all passed.
- Live stress server:
  `./ds4-server -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 100000 --kv-disk-dir /tmp/ds4-kv-stress --kv-disk-space-mb 8192 --trace /tmp/ds4-response-stress-trace.txt`.
- Pi, using the OpenAI-compatible chat/completions path, completed a multi-tool
  session and a resumed turn. The trace showed normal live-prefix reuse:
  requests after the first cold prompt used `cache_source: memory-token` with
  only short suffix prefills.
- Codex CLI, using `/v1/responses`, completed one tool-heavy turn and a resumed
  turn. Responses continuations used `cache_source: responses-visible`; the
  first resume after the final answer logged `ids=0` and continued from the
  remembered visible prefix, then tool-result turns logged `ids=1`.
- Claude Code, using `/v1/messages`, completed real read/write/bash tool loops.
  One apparent bad miss after a `Read` call was investigated in the rendered
  trace: the client changed its tool schema block between requests
  (`NotebookEdit` disappeared and `LSP` appeared), so the low common prefix was
  a real prompt change, not a DSML replay failure. Disk recovery limited the
  replay to the nearest checkpoint (`cached_tokens=22528` for a 24308-token
  prompt).
- opencode, using OpenAI-compatible chat/completions, completed a tool session
  and a resumed tool session. Most tool-result turns used live memory; a few
  used `cache_source: memory-text` because the exact token stream and rerendered
  byte prefix were text-identical but had different tokenization around a
  boundary, which is the intended text-prefix fallback.
- A direct no-tool chat/completions test exercised the new thinking-visible
  binding: first request logged `thinking live checkpoint remembered`, second
  request logged `thinking live continuation match=visible-prefix cached=54
  prompt=66`, with no rebuild and only the new suffix prefilling.
- No trace line in this run showed `thinking checkpoint canonicalization needs
  rebuild` or a tool checkpoint rebuild caused by exact sampled DSML replay.
